"""Independent metric recomputation and saved-weight prediction checks."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import Data, CHANNELS, INPUT_FILES, digest, write_json
from models import SequenceModel, tensor_inputs, predict_ridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--split', choices=['validation', 'test'], required=True)
    args = ap.parse_args()
    root = Path(args.output)
    report_file = root / ('DEVELOPMENT_VERIFIED.json' if args.split == 'validation' else 'VERIFIED.json')
    if report_file.exists():
        raise FileExistsError('Refusing to overwrite verification receipt')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seal = json.loads((root/'FINAL_SELECTION.json').read_text(encoding='utf-8'))
    cfg = json.loads((root/'CONFIG.json').read_text(encoding='utf-8'))
    receipt = json.loads((root/'DEVELOPMENT_COMPLETE.json').read_text())
    assert digest(root/'FINAL_SELECTION.json') == receipt['seal_sha256']
    assert digest(root/'CONFIG.json') == seal['config_sha256']
    assert digest(root/'scaler.npz') == seal['scaler_sha256']
    for name, h in seal['source_sha256'].items():
        assert digest(Path(__file__).parent/name) == h
    for c in seal['checkpoints'].values():
        assert digest(root/c['file']) == c['sha256']
    d = Data(args.data)
    assert d.hashes == seal['data_sha256']
    targets = np.load(root/(args.split+'_targets.npz'))
    y = targets['y'].astype(np.float64)
    origins, ids = targets['origins'], targets['region_indices']
    assert len(origins) == cfg['expected_windows'][args.split]
    np.testing.assert_array_equal(ids, d.ids)
    np.testing.assert_array_equal(origins, d.windows[d.windows.split == args.split].origin_index.to_numpy())
    for i, q in enumerate(origins):
        np.testing.assert_array_equal(y[i], d.y[q:q+24].astype(np.float32).astype(np.float64))
    results = json.loads((root/(args.split+'_metrics.json')).read_text(encoding='utf-8'))
    rows, values_checked = [], 0
    for name, result in results['methods'].items():
        p = np.load(root/(name+'_'+args.split+'.npz'))['prediction'].astype(np.float64)
        assert p.shape == y.shape and np.isfinite(p).all() and np.min(p) >= 0
        for c, label in enumerate(CHANNELS):
            for lead in (None, 1, 3, 6, 12, 24):
                actual = y[..., c] if lead is None else y[:, lead-1, :, c]
                predicted = p[..., c] if lead is None else p[:, lead-1, :, c]
                residual = predicted - actual
                computed = {'MAE': np.mean(np.abs(residual)), 'RMSE': np.linalg.norm(residual.ravel())/np.sqrt(residual.size),
                            'WAPE': np.abs(residual).sum()/actual.sum(),
                            'positive_target_MAE': np.mean(np.abs(residual[actual > 0])), 'bias': residual.mean()}
                saved = result['metrics'][label]['all_24_steps'] if lead is None else result['metrics'][label]['lead_hours'][str(lead)]
                for k, v in computed.items():
                    assert abs(v-saved[k]) < 1e-9, (name, label, lead, k, v, saved[k])
                    values_checked += 1
                if lead is None:
                    rows.append({'model': name, 'target': label, **computed})
            regional = np.mean(np.abs(p[..., c]-y[..., c]), axis=(0,1))
            np.testing.assert_allclose(regional, result['metrics'][label]['region_MAE'], rtol=1e-12, atol=1e-12)
    pd.DataFrame(rows).to_csv(root/(args.split+'_summary.csv'), index=False)

    # Reconstruct three forecast origins independently by applying raw event masks.
    take = [0, len(origins)//2, len(origins)-1]
    q = origins[take]
    b = {'x': np.stack([d.history(int(k)) for k in q]), 'origins': q,
         'past': np.stack([d.calendar[k-168:k] for k in q]),
         'future': np.stack([d.calendar[k:k+24] for k in q]), 'y': y[take].astype(np.float32)}
    # Future-completion energy adversary must not influence the historical input.
    original = d.history(int(q[0]))
    late = (d.starts < q[0]*3600) & (d.ends > q[0]*3600)
    assert late.any()
    original_energy = d.energy.copy()
    d.energy[late] *= 10000
    np.testing.assert_array_equal(original, d.history(int(q[0])))
    d.energy[:] = original_energy
    for name, p in d.baselines(b).items():
        expected = np.load(root/(name+'_'+args.split+'.npz'))['prediction'][take]
        np.testing.assert_array_equal(p, expected)
    p = predict_ridge(b, d, dict(np.load(root/'ridge_ar.npz')))
    np.testing.assert_allclose(p, np.load(root/('ridge_ar_'+args.split+'.npz'))['prediction'][take], rtol=1e-5, atol=1e-4)
    inputs = tensor_inputs(b, d)
    max_difference = 0.0
    history = json.loads((root/'training_history.json').read_text())
    for kind in ('mlp', 'gru', 'lstm'):
        predictions = []
        for seed in cfg['seeds']:
            name = '{}_seed{}'.format(kind, seed)
            m = SequenceModel(kind, len(ids), cfg, d.region_mean/d.channel_std)
            m.load_state_dict(torch.load(root/(name+'.pt'), map_location='cpu', weights_only=True))
            m.eval()
            with torch.no_grad():
                p = m(inputs[0], inputs[1]).numpy()*d.channel_std
            saved = np.load(root/(name+'_'+args.split+'.npz'))['prediction'][take]
            max_difference = max(max_difference, float(np.abs(saved-p).max()))
            np.testing.assert_allclose(p, saved, rtol=1e-5, atol=1e-4)
            predictions.append(np.load(root/(name+'_'+args.split+'.npz'))['prediction'])
            selected_epoch = seal['checkpoints'][name]['epoch']
            best = min(history[name], key=lambda v: v['validation_balanced_MAE'])
            assert selected_epoch == best['epoch']
        np.testing.assert_array_equal(np.mean(predictions, axis=0),
            np.load(root/(kind+'_ensemble_'+args.split+'.npz'))['prediction'])

    files = {p.name: digest(p) for p in sorted(root.iterdir()) if p.suffix in ('.json', '.npz', '.pt', '.csv')
             and p.name not in ('STATE.json', 'VERIFIED.json', 'DEVELOPMENT_VERIFIED.json')}
    write_json(report_file, {'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'passed': True, 'split': args.split, 'metric_values_recomputed': values_checked,
               'methods': len(results['methods']), 'source_and_data_hashes_unchanged': True,
               'saved_weight_reinference_origins_per_model': 3, 'reinference_max_abs_difference': max_difference,
               'future_completion_adversarial_isolation': True, 'selection_matches_validation_minimum': True,
               'test_scored': args.split == 'test', 'artifact_sha256': files})
    print(json.dumps({'passed': True, 'split': args.split, 'metric_values': values_checked,
                      'prediction_recheck_max_difference': max_difference}), flush=True)


if __name__ == '__main__':
    main()
