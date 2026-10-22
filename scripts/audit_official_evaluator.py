"""Exercise frozen upstream evaluator on synthetic boundary cases, without APIs."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '718b07d02fabfb95e3248a9dda5e0b7d965158e1db25767187a4cd9d688084eb':
        raise ValueError('Unexpected upstream evaluator source')
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'PredictionEvaluator')
    ns = dict(json=json, os=os, re=re, ast=ast, np=SimpleNamespace(log2=math.log2))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(args.source), 'exec'), ns)
    evaluator = ns['PredictionEvaluator']
    ids = [format(i, '024x') for i in range(1, 8)]
    cases = {
        'rank_6': [{'prediction': ids[:6], 'true': ids[5], 'output': {}}],
        'empty_prediction': [{'prediction': [], 'true': ids[0], 'output': {}}],
        'missing_output_key': [
            {'prediction': [ids[0]], 'true': ids[0], 'output': {}},
            {'prediction': [], 'true': ids[1]}],
        'unreadable_files': [{'prediction': [ids[0]], 'true': ids[0], 'output': {}}],
    }
    results = {}
    for name, rows in cases.items():
        with tempfile.TemporaryDirectory(prefix='agentmove-evaluator-') as directory:
            for i, row in enumerate(rows):
                Path(directory, f'{i}.json').write_text(json.dumps(row))
            intended = len(rows)
            if name == 'unreadable_files':
                Path(directory, 'empty.json').write_text('')
                Path(directory, 'malformed.json').write_text('{')
                intended += 2
            ev = evaluator('rule', directory, False, 'agent_move_v6')
            metrics = ev.compute_combined_top_accuracies()
            strict1 = strict5 = ndcg5 = 0
            for row in rows:
                pred = row['prediction']
                rank = pred.index(row['true']) + 1 if row['true'] in pred else math.inf
                strict1 += rank == 1
                strict5 += rank <= 5
                ndcg5 += 1 / math.log2(rank + 1) if rank <= 5 else 0
            results[name] = dict(
                intended_queries=intended, loaded_rows=len(ev.combined_data),
                official_accuracy_denominator=ev.extract_combined_response_data()[0],
                official_acc1=metrics[0], official_acc5=metrics[2], official_ndcg=metrics[5],
                fixed_denominator_acc1=strict1 / intended,
                fixed_denominator_acc5=strict5 / intended, strict_ndcg5=ndcg5 / intended)
    assert results['rank_6']['official_acc5'] == 0
    assert results['rank_6']['official_ndcg'] > 0
    assert results['rank_6']['strict_ndcg5'] == 0
    assert results['empty_prediction']['official_accuracy_denominator'] == 1
    assert results['missing_output_key']['official_acc1'] == 1
    assert results['missing_output_key']['fixed_denominator_acc1'] == .5
    assert results['unreadable_files']['loaded_rows'] == 1
    assert results['unreadable_files']['fixed_denominator_acc1'] == 1 / 3
    result = dict(status='synthetic_evaluator_boundary_audit_pass', source_sha256=digest,
                  model_calls=0, experimental_prediction_scores=False, cases=results,
                  findings=['Official NDCG is not truncated at rank5.',
                            'Empty predictions with output key stay in accuracy denominator.',
                            'Missing output keys and unreadable files can shrink accuracy denominator.'],
                  required_reporting='Report official-compatible metrics separately from fixed-manifest Acc@1/5 and strict NDCG@5; count failed/missing planned predictions as misses.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
