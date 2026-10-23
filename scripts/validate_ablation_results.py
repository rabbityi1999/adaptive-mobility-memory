"""Score all completed reconstructed ablations, retaining 200 users per arm.

No interim performance is read. Six contrasts were fixed before inference.
Raw responses, per-user results and source paths are not exported.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ablation_adapter import MASKS, NEW_CONDITIONS, SOURCE_CONDITION, digest, read, sha
from validate_llama3_results import exact_discordance, predictions, score


def validate(args):
    status = read(args.run/'status.json')
    if status.get('stage') != 'completed' or not (args.run/'exit.json').exists():
        raise SystemExit('Run is unfinished: no performance read or analysis written')
    assert read(args.run/'exit.json')['exit_code'] == 0
    assert status['finished_queries'] == 200
    assert not args.output.exists()
    meta = read(args.run/'run_metadata.json')
    plan = read(args.run/'ablation_preflight.json')
    assert plan == meta['ablation'] and plan['status'] == 'reconstructed_ablation_preflight_pass'
    assert set(meta['conditions']) == set(MASKS)
    selected = read(args.run/'selected_queries.json')
    assert sha(args.run/'selected_queries.json') == meta['selection_sha256']
    assert len(selected) == len({u for u,t in selected}) == 200
    manifest = read(args.run/'ablation_prompt_manifest.json')
    assert digest(json.dumps(manifest,sort_keys=True)) == plan['prompt_manifest_sha256']
    lookup = {(m['query'],m['condition']):m for m in manifest}
    expected_keys = {(u+':'+t,c) for u,t in selected for c in MASKS}
    assert len(lookup) == len(manifest) == 1200 and set(lookup) == expected_keys
    source = Path(meta['ablation_source'])
    assert sha(source/'run_metadata.json') == plan['source_run_metadata_sha256']
    assert sha(source/'wkg_cache.jsonl') == plan['source_wkg_cache_sha256']
    source_meta = read(source/'run_metadata.json')
    source_fingerprints = {}
    for condition, prompt_type in source_meta['conditions'].items():
        for u,t in selected:
            name = f'llama3-8B-local_{prompt_type}_{u}_{t}_False.json'
            source_fingerprints[condition+'/'+name] = sha(source/'results'/condition/name)
    assert digest(json.dumps(source_fingerprints,sort_keys=True)) == plan['source_result_tree_sha256']
    labels = {}
    for u,t in selected:
        value = read(source/'results'/SOURCE_CONDITION/f'llama3-8B-local_agent_move_v6_{u}_{t}_False.json')
        labels[u+':'+t] = value
    evaluation = read(args.run/'evaluation.json')
    assert evaluation['complete'] and evaluation['denominator'] == 200
    arrays, summaries, fingerprints, completed = {}, {}, {}, set()
    for condition in MASKS:
        folder = args.run/'results'/condition
        expected = {f'llama3-8B-local_agent_move_v6_{u}_{t}_False.json' for u,t in selected}
        assert {p.name for p in folder.glob('*.json')} == expected
        values, diagnostic = [], Counter(execution_errors=0, empty_predictions=0, malformed_id_occurrences=0,
                                         predictions_not_exactly_five=0, raw_response_fallback=0)
        for u,t in selected:
            query = u+':'+t
            path = folder/f'llama3-8B-local_agent_move_v6_{u}_{t}_False.json'
            result = read(path)
            assert result['true'] == labels[query]['true'] and 'output' in result
            assert result['execution_status'] in ['completed','error']
            if condition == 'recon_full':
                assert result == labels[query]
            if result['execution_status'] == 'completed':
                assert digest(result['input']) == lookup[(query,condition)]['original_sha256']
                completed.add((query,condition))
            pred = predictions(result)
            if result['execution_status'] == 'error':
                assert not pred
                diagnostic['execution_errors'] += 1
            diagnostic['empty_predictions'] += not pred
            import re
            diagnostic['malformed_id_occurrences'] += sum(not re.fullmatch('[0-9a-f]{24}',p) for p in pred)
            diagnostic['predictions_not_exactly_five'] += len(pred) != 5
            diagnostic['raw_response_fallback'] += isinstance(result['output'],dict) and 'raw_response' in result['output']
            values.append(score(pred,result['true']))
            fingerprints[condition+'/'+path.name] = sha(path)
        array = np.asarray(values)
        metrics = dict(zip(['Acc1','Acc5','NDCG5','official_ndcg_untruncated'],array.mean(axis=0).tolist()))
        assert all(abs(metrics[k]-evaluation['conditions'][condition][k]) < 1e-12 for k in metrics)
        arrays[condition] = array
        summaries[condition] = dict(n=200, **metrics, **diagnostic, reused=condition=='recon_full')
    pairs = [(c,'recon_base') for c in MASKS if c != 'recon_base'] + [('recon_full','recon_stm_wkg')]
    rng = np.random.default_rng(20261023)
    resamples = rng.integers(0,200,size=(10000,200))
    contrasts = []
    for a,b in pairs:
        delta = arrays[a][:,:3]-arrays[b][:,:3]
        ci = np.quantile(delta[resamples].mean(axis=1),[.025,.975],axis=0)
        a_only = int(((arrays[a][:,0] == 1) & (arrays[b][:,0] == 0)).sum())
        b_only = int(((arrays[b][:,0] == 1) & (arrays[a][:,0] == 0)).sum())
        contrasts.append(dict(a=a,b=b,direction='a minus b',a_only_correct=a_only,b_only_correct=b_only,
            acc1_exact_p=exact_discordance(a_only,b_only),metrics={name:dict(difference=float(delta[:,i].mean()),
            ci95=ci[:,i].tolist()) for i,name in enumerate(['Acc1','Acc5','NDCG5'])}))
    running_p = 0.
    for i,item in enumerate(sorted(contrasts,key=lambda x:x['acc1_exact_p'])):
        running_p = max(running_p,min(1.,(6-i)*item['acc1_exact_p']))
        item['acc1_holm_p_six_contrasts'] = running_p
    costs = defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0,inference_seconds=0.,clipped_calls=0))
    call_keys = set()
    for line in (args.run/'calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        pair = (call['query'],call['stage'])
        assert pair not in call_keys and pair in expected_keys and call['stage'] in NEW_CONDITIONS
        call_keys.add(pair)
        m = lookup[pair]
        assert digest(call['prompt_original']) == m['original_sha256']
        assert digest(call['prompt_effective']) == call['prompt_sha256'] == m['effective_sha256']
        assert call['input_tokens'] == m['input_tokens'] and call['clipped'] == m['clipped']
        assert call['retained_payloads'] == m['retained']
        item = costs[call['stage']]
        item['calls'] += 1
        item['input_tokens'] += call['input_tokens']
        item['output_tokens'] += call['output_tokens']
        item['inference_seconds'] += call['seconds']
        item['clipped_calls'] += call['clipped']
    assert {x for x in completed if x[1] != 'recon_full'} <= call_keys
    assert len(call_keys) == status['model_calls'] == evaluation['model_calls']
    if sum(v['execution_errors'] for v in summaries.values()) == 0:
        assert len(call_keys) == 1000
    result = dict(status='completed_ablation_scores_independently_recomputed',checked_at=time.time(),n_users=200,
        model=meta['model'],conditions=summaries,comparisons=contrasts,new_execution_costs=dict(costs),
        historical_WKG_cost=plan['source_WKG_cost'],historical_reused_full_cost=plan['source_full_prediction_cost'],
        retained_payloads=plan['retention_counts'],bootstrap=dict(seed=20261023,draws=10000,unit='paired user'),
        integrity=dict(exit_code=0,result_files=1200,reused_full_files=200,no_model_rerun=True,
            result_tree_sha256=digest(json.dumps(fingerprints,sort_keys=True)),
            source_result_tree_sha256=plan['source_result_tree_sha256'],selection_sha256=meta['selection_sha256'],
            prompt_manifest_sha256=plan['prompt_manifest_sha256'],runner_sha256=meta['script_sha256'],
            adapter_sha256=plan['adapter_sha256'],validator_sha256=sha(__file__),
            metric_helpers_sha256=sha(Path(__file__).with_name('validate_llama3_results.py'))),
        caveats=['Reconstructed Tokyo ablations, not original author Table2 scripts.',
                 'Observed ordered200-user cohort; no new independent holdout.',
                 'Descriptive percentile intervals assume exchangeable users and are not simultaneous.',
                 'Section removal changes prompt length and clipping; effects are not token-matched.',
                 'Reuse is conditional on fixed inputs/model/cachedWKG; no independent full-condition rerun.',
                 'Add historicalWKG costs to standalone WKG-enabled conditions, not to physical new execution total.'])
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert __debug__, 'Run without -O'
    validate(args)
