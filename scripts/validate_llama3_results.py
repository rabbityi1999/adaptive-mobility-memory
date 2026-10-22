"""Independently score completed fixed-manifest outputs and paired differences.

No model calls. Reject unfinished jobs; never compute interim performance.
Only aggregate results are written. Raw query identities stay in the run folder.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time

import numpy as np

CONDITIONS = ['agent_move_v6_causalCKE', 'llmzs', 'agent_move_v6_officialCKE']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def predictions(value):
    raw = value.get('prediction')
    if isinstance(raw, list):
        return [p.lower() for p in raw if isinstance(p, str)]
    if isinstance(raw, str):
        return re.findall(r'"([0-9a-f]{24})"', raw)
    return []


def score(pred, label):
    rank = pred.index(label.lower()) + 1 if label.lower() in pred else math.inf
    return [float(rank == 1), float(rank <= 5), 1/math.log2(rank+1) if rank <= 5 else 0,
            1/math.log2(rank+1) if math.isfinite(rank) else 0]


def exact_discordance(a_only, b_only):
    n = a_only + b_only
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only)+1)) / 2**n) if n else 1.


def self_test():
    ids = [format(i, '024x') for i in range(7)]
    assert score(ids[:6], ids[5])[:3] == [0., 0., 0]
    assert score(ids[:6], ids[5])[3] > 0
    assert score([], ids[0]) == [0., 0., 0, 0]
    assert score([ids[0]] * 5, ids[0]) == [1., 1., 1., 1.]
    assert predictions({'prediction': [ids[0].upper(), 4, None]}) == [ids[0]]
    assert predictions({'prediction': None}) == []
    assert exact_discordance(0, 0) == exact_discordance(3, 3) == 1
    assert exact_discordance(6, 0) == .03125
    # One failed/missing result stays in the fixed two-query denominator.
    assert np.mean([score([ids[0]], ids[0])[0], score([], ids[1])[0]]) == .5
    print('Synthetic rank, extraction, failure-denominator and exact-discordance checks passed')


def validate(args):
    status = json.loads((args.run / 'status.json').read_text())
    exit_path = args.run / 'exit.json'
    if status.get('stage') != 'completed' or not exit_path.exists():
        raise SystemExit('Run is unfinished: no performance read or analysis written')
    assert json.loads(exit_path.read_text())['exit_code'] == 0
    assert status['finished_queries'] == 200
    meta = json.loads((args.run / 'run_metadata.json').read_text())
    selected_path = args.run / 'selected_queries.json'
    assert sha(selected_path) == meta['selection_sha256']
    assert sha(args.bundle) == meta['bundle_sha256']
    rows = json.loads(args.bundle.read_text())
    selected = json.loads(selected_path.read_text())
    assert [[r['user'], r['trajectory']] for r in rows] == selected
    assert len(rows) == len({r['user'] for r in rows}) == 200
    assert set(meta['conditions']) == set(CONDITIONS)
    evaluation = json.loads((args.run / 'evaluation.json').read_text())
    assert evaluation['complete'] and evaluation['denominator'] == 200
    vectors, summary, fingerprints = {}, {}, {}
    for condition in CONDITIONS:
        folder = args.run / 'results' / condition
        prompt_type = meta['conditions'][condition]
        expected = {f'llama3-8B-local_{prompt_type}_{u}_{t}_False.json' for u, t in selected}
        assert {f.name for f in folder.glob('*.json')} == expected
        scores, errors, empty, invalid, nonfive, fallback = [], 0, 0, 0, 0, 0
        for row in rows:
            path = folder / f'llama3-8B-local_{prompt_type}_{row["user"]}_{row["trajectory"]}_False.json'
            value = json.loads(path.read_text())
            assert value['true'] == row['label']['ground_stay']
            assert value.get('execution_status') in ['completed', 'error']
            assert 'output' in value  # keep official accuracy denominator at200
            pred = predictions(value)
            if value['execution_status'] == 'error':
                assert not pred  # no partial prediction masquerading as completed
                errors += 1
            scores.append(score(pred, value['true']))
            empty += not pred
            invalid += sum(not re.fullmatch('[0-9a-f]{24}', p) for p in pred)
            nonfive += len(pred) != 5
            fallback += isinstance(value['output'], dict) and 'raw_response' in value['output']
            fingerprints[f'{condition}/{path.name}'] = sha(path)
        array = np.asarray(scores, dtype=float)
        vectors[condition] = array
        means = array.mean(axis=0)
        recorded = evaluation['conditions'][condition]
        for key, number in zip(['Acc1', 'Acc5', 'NDCG5', 'official_ndcg_untruncated'], means):
            assert math.isclose(float(number), recorded[key], abs_tol=1e-12), (condition, key)
        summary[condition] = dict(zip(['Acc1', 'Acc5', 'NDCG5', 'official_ndcg_untruncated'], means.tolist()))
        summary[condition].update(n=200, execution_errors=errors, empty_predictions=empty,
                                  invalid_ids=invalid, predictions_not_exactly_five=nonfive,
                                  raw_response_fallback=fallback)
    rng = np.random.default_rng(20261023)
    samples = rng.integers(0, 200, size=(10000, 200))
    comparisons = []
    for a, b in itertools.combinations(CONDITIONS, 2):
        delta = vectors[a][:, :3] - vectors[b][:, :3]
        boot = delta[samples].mean(axis=1)
        ci = np.quantile(boot, [.025, .975], axis=0)
        a_only = int(((vectors[a][:, 0] == 1) & (vectors[b][:, 0] == 0)).sum())
        b_only = int(((vectors[b][:, 0] == 1) & (vectors[a][:, 0] == 0)).sum())
        comparisons.append(dict(a=a, b=b, direction='a minus b',
             metrics={m: dict(difference=float(delta[:, i].mean()), ci95=ci[:, i].tolist())
                      for i, m in enumerate(['Acc1', 'Acc5', 'NDCG5'])},
             a_only_correct=a_only, b_only_correct=b_only,
             acc1_exact_p=exact_discordance(a_only, b_only)))
    running_p = 0.
    for i, item in enumerate(sorted(comparisons, key=lambda c: c['acc1_exact_p'])):
        running_p = max(running_p, min(1., (3-i) * item['acc1_exact_p']))
        item['acc1_holm_p_three_pairs'] = running_p
    costs = defaultdict(lambda: dict(calls=0, input_tokens=0, output_tokens=0,
                                   inference_seconds=0., clipped_calls=0, max_new_tokens_reached=0))
    call_counts = Counter()
    query_keys = {u + ':' + t for u, t in selected}
    for line in (args.run / 'calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        stage, query = call['stage'], call['query']
        assert stage in CONDITIONS + ['WKG'] and query in query_keys
        call_counts[(query, stage)] += 1
        item = costs[stage]
        item['calls'] += 1
        for key in ['input_tokens', 'output_tokens']:
            item[key] += call[key]
        item['inference_seconds'] += call['seconds']
        item['clipped_calls'] += bool(call['clipped'])
        item['max_new_tokens_reached'] += bool(call['max_new_tokens_reached'])
    assert all(n <= (2 if stage == 'WKG' else 1) for (_, stage), n in call_counts.items())
    if not sum(v['execution_errors'] for v in summary.values()):
        assert len(call_counts) == 800
        assert sum(v['calls'] for v in costs.values()) == 1000
    assert sum(v['calls'] for v in costs.values()) == status['model_calls'] == evaluation['model_calls']
    result = dict(status='completed_output_scores_independently_recomputed',
        analysis_time_unix=time.time(), model='Meta-Llama-3-8B-Instruct', n_users=200,
        conditions=summary, comparisons=comparisons, costs_by_stage=dict(costs),
        bootstrap=dict(seed=20261023, draws=10000, unit='paired user', interval='percentile95',
                       limitation='Conditional on ordered benchmark cohort; user exchangeability assumed, not a random population sample. Three metric intervals are descriptive and not simultaneous.'),
        cost_note='WKG cost counted once here; add it to either standalone AgentMove budget, never to LLM-ZS. Shared execution total is not sum of standalone framework budgets.',
        integrity=dict(exit_code=0, manifest_and_ground_truth_matched=True, output_files=600,
            failures_retained=True, metric_matches_runner=True, no_model_rerun=True,
            result_tree_sha256=hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
            selection_sha256=meta['selection_sha256'], bundle_sha256=meta['bundle_sha256'],
            runner_sha256=meta['script_sha256'], validator_sha256=sha(__file__)),
        caveats=['Original static CKE is non-causal; causal variant uses modern addresses and fixed offline membership.',
                 'Local Llama3 is not the paper GPT4omini main-table configuration.',
                 'Ordered200-user cohort; no hyperparameter tuning or independence claim from1000calls.',
                 'Post-output recomputation is not independent retraining or model replication.'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Run without -O')
    if args.self_test:
        self_test()
    else:
        assert args.run and args.bundle and args.output
        validate(args)
