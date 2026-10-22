"""Develop on train/validation, seal choices, then evaluate test in a separate call."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import datetime as dt
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from data import Data, CHANNELS, INPUT_FILES, digest, write_json, metrics, balanced_score
from models import SequenceModel, tensor_inputs, fit_ridge, predict_ridge


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: digest(p) for p in sorted(root.glob('*.py'))}


def predict(model, inputs, channel_std, batch=32):
    model.eval()
    values = []
    with torch.no_grad():
        for begin in range(0, len(inputs[0]), batch):
            values.append(model(inputs[0][begin:begin+batch], inputs[1][begin:begin+batch]).numpy())
    return np.concatenate(values) * channel_std


def develop(args, config, log, check_timeout):
    dest = Path(args.output)
    data = Data(args.data)
    assert len(data.ids) == config['expected_regions']
    write_json(dest/'PROVENANCE.json', {'created_at': now(), 'input_sha256': data.hashes,
               'source_sha256': source_hashes(), 'python': sys.version, 'numpy': np.__version__,
               'torch': torch.__version__, 'device': 'cpu', 'threads': torch.get_num_threads(),
               'data_argument': str(Path(args.data).resolve()), 'test_scored': False})
    train, valid = data.split('train'), data.split('validation')
    np.savez_compressed(dest/'validation_targets.npz', y=valid['y'], origins=valid['origins'], region_indices=data.ids)
    np.savez_compressed(dest/'scaler.npz', region_mean=data.region_mean, region_std=data.region_std,
                        channel_std=data.channel_std, week_means=data.week_means, region_indices=data.ids)
    controls = data.baselines(valid)
    # Ensure identical origin order, geography and causal baseline construction to the accepted dataset.
    old = np.load(Path(args.data)/'baselines_validation/evaluation_targets.npz')
    np.testing.assert_array_equal(old['origin_indices'], valid['origins'])
    np.testing.assert_array_equal(old['region_indices'], data.ids)
    np.testing.assert_allclose(old['y'], valid['y'], rtol=1e-7, atol=1e-4)
    for name, p in controls.items():
        prior = np.load(Path(args.data)/'baselines_validation'/(name+'.npz'))['prediction']
        np.testing.assert_array_equal(p, prior)
    baseline_mae = np.abs(controls['training_hour_of_week_mean'].astype(np.float64)-valid['y']).mean((0,1,2))
    results = {'split': 'validation', 'origins': len(valid['origins']), 'regions': len(data.ids),
               'baseline_mae': baseline_mae.tolist(), 'methods': {}}
    def save(name, p, details):
        np.savez_compressed(dest/(name+'_validation.npz'), prediction=p)
        results['methods'][name] = {'metrics': metrics(valid['y'], p), **details}
        write_json(dest/'validation_metrics.json', results)
    for name, p in controls.items():
        save(name, p, {'kind': 'fixed_statistical_baseline'})
    log({'stage': 'data_ready', 'train': len(train['origins']), 'validation': len(valid['origins']),
         'regions': len(data.ids), 'input_bytes': train['x'].nbytes + valid['x'].nbytes})

    ridge_predictions, states = fit_ridge(train, valid, data, config['ridge_alphas'], log)
    ridge_scores = [balanced_score(valid['y'], p, baseline_mae) for p in ridge_predictions]
    index = int(np.argmin(ridge_scores))
    np.savez_compressed(dest/'ridge_ar.npz', **states[index])
    write_json(dest/'ridge_candidates.json', {'alphas': config['ridge_alphas'], 'scores': ridge_scores,
                                              'selected_alpha': config['ridge_alphas'][index]})
    save('ridge_ar', ridge_predictions[index], {'kind': 'per_region_direct_autoregression',
         'alpha': config['ridge_alphas'][index], 'selection_score': ridge_scores[index]})
    log({'stage': 'ridge_selected', 'alpha': config['ridge_alphas'][index], 'scores': ridge_scores})
    del ridge_predictions, states
    train_t, valid_t = tensor_inputs(train, data), tensor_inputs(valid, data)
    checkpoints = {'ridge_ar': {'file': 'ridge_ar.npz', 'sha256': digest(dest/'ridge_ar.npz')}}
    histories = {}
    for kind in ('mlp', 'gru', 'lstm'):
        family = []
        for seed in config['seeds']:
            check_timeout()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model = SequenceModel(kind, len(data.ids), config, data.region_mean/data.channel_std)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            name = '{}_seed{}'.format(kind, seed)
            best_score, best_epoch, stale = float('inf'), 0, 0
            trace, started = [], time.monotonic()
            for epoch in range(1, config['max_epochs']+1):
                model.train()
                order = torch.randperm(len(train_t[0]), generator=generator)
                total_loss = 0.0
                for begin in range(0, len(order), config['batch_size']):
                    check_timeout()
                    idx = order[begin:begin+config['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(train_t[0][idx], train_t[1][idx])
                    loss = torch.nn.functional.mse_loss(prediction, train_t[2][idx])
                    assert torch.isfinite(loss).item(), 'nonfinite loss'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    total_loss += loss.item()*len(idx)
                p = predict(model, valid_t, data.channel_std, config['batch_size'])
                score = balanced_score(valid['y'], p, baseline_mae)
                row = {'epoch': epoch, 'train_normalized_MSE': total_loss/len(order),
                       'validation_balanced_MAE': score, 'elapsed_seconds': time.monotonic()-started}
                trace.append(row)
                if score < best_score:
                    best_score, best_epoch, stale = score, epoch, 0
                    torch.save(model.state_dict(), dest/(name+'.pt'))
                else:
                    stale += 1
                log({'stage': 'training', 'model': name, **row, 'best_epoch': best_epoch})
                histories[name] = trace
                write_json(dest/'training_history.json', histories)
                if stale >= config['patience']:
                    break
            model.load_state_dict(torch.load(dest/(name+'.pt'), map_location='cpu', weights_only=True))
            p = predict(model, valid_t, data.channel_std, config['batch_size'])
            assert abs(balanced_score(valid['y'], p, baseline_mae)-best_score) < 1e-9
            family.append(p)
            save(name, p, {'kind': kind, 'seed': seed, 'best_epoch': best_epoch,
                          'selection_score': best_score, 'parameter_count': sum(p.numel() for p in model.parameters()),
                          'seconds': time.monotonic()-started})
            checkpoints[name] = {'file': name+'.pt', 'sha256': digest(dest/(name+'.pt')),
                                 'model': kind, 'seed': seed, 'epoch': best_epoch}
            log({'stage': 'model_complete', 'model': name, 'best_epoch': best_epoch, 'score': best_score})
            del model, optimizer
        # Predeclared arithmetic mean; not a validation-fitted ensemble.
        save(kind+'_ensemble', np.mean(family, axis=0), {'kind': kind, 'aggregation': 'mean predictions of all three seeds'})
    selected = min((name for name in results['methods'] if name != 'zero'),
                   key=lambda name: np.mean([results['methods'][name]['metrics'][c]['all_24_steps']['MAE']/baseline_mae[i]
                                             for i, c in enumerate(CHANNELS)]))
    # Includes every family/seed, preventing selection of a lucky seed after seeing test scores.
    seal = {'sealed_at': now(), 'config_sha256': digest(dest/'CONFIG.json'), 'data_sha256': data.hashes,
            'source_sha256': source_hashes(), 'checkpoints': checkpoints,
            'scaler_sha256': digest(dest/'scaler.npz'), 'selected_on_validation': selected,
            'test_evaluation_plan': 'all fixed controls, selected ridge alpha, all three seeds per neural family, their arithmetic ensembles',
            'test_evaluated_before_seal': False, 'retrain_on_validation': False,
            'prior_exposure': config['prior_exposure']}
    write_json(dest/'FINAL_SELECTION.json', seal)
    write_json(dest/'DEVELOPMENT_COMPLETE.json', {'at': now(), 'seal_sha256': digest(dest/'FINAL_SELECTION.json'),
               'models_with_metrics': len(results['methods']), 'test_scored': False})
    log({'stage': 'development_complete', 'validation_selected': selected, 'test_scored': False})


def test(args, config, log, check_timeout):
    dest = Path(args.output)
    assert (dest/'DEVELOPMENT_VERIFIED.json').exists(), 'Independent validation audit required before test scoring'
    assert not (dest/'test_metrics.json').exists(), 'Refusing repeated test scoring'
    receipt = json.loads((dest/'DEVELOPMENT_COMPLETE.json').read_text())
    assert digest(dest/'FINAL_SELECTION.json') == receipt['seal_sha256']
    seal = json.loads((dest/'FINAL_SELECTION.json').read_text())
    assert source_hashes() == seal['source_sha256'], 'Frozen source changed'
    assert digest(dest/'CONFIG.json') == seal['config_sha256']
    assert digest(dest/'scaler.npz') == seal['scaler_sha256']
    for record in seal['checkpoints'].values():
        assert digest(dest/record['file']) == record['sha256']
    data = Data(args.data)
    assert data.hashes == seal['data_sha256']
    bundle = data.split('test')
    tensors = tensor_inputs(bundle, data)
    np.savez_compressed(dest/'test_targets.npz', y=bundle['y'], origins=bundle['origins'], region_indices=data.ids)
    results = {'split': 'test', 'origins': len(bundle['origins']), 'regions': len(data.ids),
               'seal_sha256': digest(dest/'FINAL_SELECTION.json'), 'selected_on_validation': seal['selected_on_validation'],
               'methods': {}}
    def save(name, p):
        np.savez_compressed(dest/(name+'_test.npz'), prediction=p)
        results['methods'][name] = {'metrics': metrics(bundle['y'], p)}
        log({'stage': 'test_prediction_complete', 'model': name})
    for name, p in data.baselines(bundle).items():
        save(name, p)
    save('ridge_ar', predict_ridge(bundle, data, dict(np.load(dest/'ridge_ar.npz'))))
    for kind in ('mlp', 'gru', 'lstm'):
        family = []
        for seed in config['seeds']:
            check_timeout()
            name = '{}_seed{}'.format(kind, seed)
            model = SequenceModel(kind, len(data.ids), config, data.region_mean/data.channel_std)
            model.load_state_dict(torch.load(dest/(name+'.pt'), map_location='cpu', weights_only=True))
            p = predict(model, tensors, data.channel_std, config['batch_size'])
            save(name, p); family.append(p)
        save(kind+'_ensemble', np.mean(family, axis=0))
    write_json(dest/'test_metrics.json', results)
    write_json(dest/'COMPLETE.json', {'at': now(), 'status': 'completed_pending_independent_verification',
               'test_scored': True, 'methods': len(results['methods']), 'seal_sha256': digest(dest/'FINAL_SELECTION.json')})
    log({'stage': 'test_complete'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['develop', 'test'], required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    dest = Path(args.output)
    if args.stage == 'develop':
        dest.mkdir(parents=True, exist_ok=False)
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        write_json(dest/'CONFIG.json', config)
    else:
        config = json.loads((dest/'CONFIG.json').read_text(encoding='utf-8'))
        assert config == json.loads(Path(args.config).read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    def log(value):
        row = {'at': now(), **value}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (dest/'progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
        write_json(dest/'STATE.json', {'pid': os.getpid(), 'stage': args.stage, **row})
    def check_timeout():
        if time.monotonic()-started > config['hard_timeout_seconds']:
            raise TimeoutError('Configured hard timeout reached')
    lock = dest/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'started': now(), 'stage': args.stage}, f)
    try:
        log({'stage': args.stage+'_started', 'pid': os.getpid()})
        (develop if args.stage == 'develop' else test)(args, config, log, check_timeout)
        log({'stage': args.stage+'_finished', 'elapsed_seconds': time.monotonic()-started})
    except Exception as exc:
        write_json(dest/(args.stage+'_FAILED.json'), {'at': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        log({'stage': 'failed', 'error': repr(exc)})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
