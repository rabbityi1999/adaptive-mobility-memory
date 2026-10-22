"""Audit pinned prompt composition, cached WKG and actual clipped inputs.

Read-only with respect to completed runs. No model calls or network. Only
aggregate diagnostics and hashes are exported; prompts and trajectories stay
in their existing private locations. Run with the original pandas environment.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import time

PINNED = {
    'models/prompts.py': '14b44b5b216825561eed907dbc8dcd44fea3d53e134b7ce9581f785d3f476ff4',
    'models/personal_memory.py': 'aef9e9538ae767f55d981b9814c8929b0bdab24c8e79214f950b57eed0782fd9',
    'models/world_model.py': '7b251f5237b7b87fa70f7061ec26c1a88031798752048210c8dc6f7803d71f5c',
    'agent.py': 'b03c77b10350213dabf6363613039eba519fefec9b41ad3807d7df7281cf83cf',
}
HEADINGS = [
    ('WKG', '## The potential places from the global spatial view:\n'),
    ('CKE', '## The nearby places visited by other users with similar mobility pattern:\n'),
    ('STM', '## The personal profile and long memory:\n'),
    ('history', '## The history data:\n'),
    ('output', '## Output'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def payload_spans(prompt):
    positions = []
    for name, heading in HEADINGS:
        assert prompt.count(heading) == 1, name
        positions.append((name, prompt.index(heading), len(heading)))
    assert [p[1] for p in positions] == sorted(p[1] for p in positions)
    spans = {}
    for i, (name, start, length) in enumerate(positions):
        end = positions[i+1][1] if i+1 < len(positions) else len(prompt)
        left, right = start+length, end
        while left < right and prompt[left].isspace():
            left += 1
        while right > left and prompt[right-1].isspace():
            right -= 1
        spans[name] = (left, right)
    return spans


def retained_state(start, end, cutoff):
    if start == end:
        return 'empty_source', 0
    kept = max(0, end-max(start, cutoff))
    return ('missing' if not kept else 'full' if kept == end-start else 'partial'), kept


def self_test():
    assert retained_state(5, 10, 0) == ('full', 5)
    assert retained_state(5, 10, 7) == ('partial', 3)
    assert retained_state(5, 10, 10) == ('missing', 0)
    assert retained_state(5, 10, 20) == ('missing', 0)
    assert retained_state(5, 5, 20) == ('empty_source', 0)
    example = ''.join(heading+' abc \n' for _, heading in HEADINGS)
    spans = payload_spans(example)
    assert all(example[a:b] == 'abc' for a, b in spans.values())
    print('Payload-span and clipping-boundary checks passed')


def audit(args):
    import pandas as pd
    def no_network(*unused, **kwargs):
        raise RuntimeError('No network in read-only prompt audit')
    socket.socket.connect = no_network
    socket.getaddrinfo = no_network
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
        os.nice(19)
    assert not args.output.exists()
    meta = read(args.run/'run_metadata.json')
    verified = read(args.run/'validated_summary.json')
    assert read(args.run/'status.json')['stage'] == 'completed'
    assert read(args.run/'exit.json')['exit_code'] == 0
    assert verified['status'] == 'completed_output_scores_independently_recomputed'
    assert sha(args.bundle) == meta['bundle_sha256'] == verified['integrity']['bundle_sha256']
    assert sha(args.run/'selected_queries.json') == verified['integrity']['selection_sha256']
    for name, fingerprint in PINNED.items():
        assert sha(args.upstream/name) == meta['upstream_sha256'][name] == fingerprint
    # Load only hash-verified pure definitions; skip package/provider imports.
    prompt_tree = ast.parse((args.upstream/'models/prompts.py').read_text())
    nodes = [n for n in prompt_tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-prompts>', 'exec'), scope)
    mem_tree = ast.parse((args.upstream/'models/personal_memory.py').read_text())
    memory_class = next(n for n in mem_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Memory')
    mem_scope = {'pd': pd}
    exec(compile(ast.Module(body=[memory_class], type_ignores=[]), '<pinned-memory>', 'exec'), mem_scope)
    world_tree = ast.parse((args.upstream/'models/world_model.py').read_text())
    world_class = next(n for n in world_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpatialWorld')
    world_formatter = next(n for n in world_class.body if isinstance(n, ast.FunctionDef) and n.name == 'get_world_info')
    world_scope = {}
    exec(compile(ast.Module(body=[world_formatter], type_ignores=[]), '<pinned-world-format>', 'exec'), world_scope)
    from types import SimpleNamespace
    selected = read(args.run/'selected_queries.json')
    bundle = read(args.bundle)
    assert [[r['user'], r['trajectory']] for r in bundle] == selected
    assert len(selected) == len({u for u, _ in selected}) == 200
    cached_rows = rows(args.run/'wkg_cache.jsonl')
    cache = {r['query']: r for r in cached_rows}
    assert len(cache) == len(cached_rows) == 200
    calls = rows(args.run/'calls.jsonl')
    by_key = defaultdict(list)
    for call in calls:
        by_key[(call['query'], call['stage'])].append(call)
    assert len(calls) == 1000 and len(by_key) == 800
    expected_keys = {(u+':'+t, stage) for u, t in selected for stage in [*meta['conditions'], 'WKG']}
    assert set(by_key) == expected_keys
    stats = {c: {name: Counter(full=0, partial=0, missing=0, empty_source=0) for name, _ in HEADINGS}
             for c in meta['conditions'] if c != 'llmzs'}
    fingerprints = {}
    prompt_hashes = {}
    rebuilt, wkg_matched, zs_differs = 0, 0, 0
    for row in bundle:
        query = row['user']+':'+row['trajectory']
        wkg_calls = by_key[(query, 'WKG')]
        assert len(wkg_calls) == 2
        cached = cache[query]
        assert cached['world_model'] == dict(subdistrict=wkg_calls[0]['output'], poi=wkg_calls[1]['output'])
        spatial = world_scope['get_world_info'](SimpleNamespace(world_model=cached['world_model'], max_lens=1000))
        assert spatial == cached['world_info']
        wkg_matched += 1
        ex = copy.deepcopy(row['inputs'])
        memory = mem_scope['Memory'](ex['historical_stays_long'], ex['context_stays'], memory_lens=15)
        memory_info = memory.read_memory(row['user'], ex['target_stay'])
        for condition, prompt_type in meta['conditions'].items():
            records = by_key[(query, condition)]
            assert len(records) == 1
            call = records[0]
            original, effective = call['prompt_original'], call['prompt_effective']
            assert call['clipped'] == (call['gpt_token_count_before'] > 2000)
            assert effective == (original[-6000:] if call['clipped'] else original)
            assert hashlib.sha256(effective.encode()).hexdigest() == call['prompt_sha256']
            path = args.run/'results'/condition/f'llama3-8B-local_{prompt_type}_{row["user"]}_{row["trajectory"]}_False.json'
            output = read(path)
            assert output['input'] == original and output['execution_status'] == 'completed'
            fingerprints[f'{condition}/{path.name}'] = sha(path)
            prompt_hashes[query+'/'+condition] = call['prompt_sha256']
            if condition == 'llmzs':
                assert scope['prompt_generator_llmzs'](ex) == original
                zs_differs += original != scope['prompt_generator_agent'](ex, 'agent_move_v6', '',
                    {'historical_info': '', 'user_profile': ''}, '')
                continue
            spans = payload_spans(original)
            start, end = spans['CKE']
            social = row['social_world_info'] if condition == 'agent_move_v6_causalCKE' else original[start:end]
            regenerated = scope['prompt_generator_agent'](ex, 'agent_move_v6', spatial, memory_info, social)
            # Original social block may have outer whitespace; retain its exact slice.
            if condition == 'agent_move_v6_officialCKE':
                begin = original.index(HEADINGS[1][1])+len(HEADINGS[1][1])
                finish = original.index(HEADINGS[2][1])
                social = original[begin:finish]
                assert social.endswith('\n\n')
                social = social[:-2]
                regenerated = scope['prompt_generator_agent'](ex, 'agent_move_v6', spatial, memory_info, social)
            assert regenerated == original, condition
            rebuilt += 1
            cutoff = len(original)-len(effective)
            for name, (a, b) in spans.items():
                state, _ = retained_state(a, b, cutoff)
                stats[condition][name][state] += 1
    tree_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    assert tree_hash == verified['integrity']['result_tree_sha256']
    agent_fn = next(n for n in prompt_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prompt_generator_agent')
    branches = sorted({cmp.comparators[0].value for cmp in ast.walk(agent_fn)
                       if isinstance(cmp, ast.Compare) and isinstance(cmp.left, ast.Name)
                       and cmp.left.id == 'prompt_type' and len(cmp.comparators) == 1
                       and isinstance(cmp.comparators[0], ast.Constant)})
    assert branches == ['agent_move_v6']
    memory_fields = sorted({n.slice.value for n in ast.walk(agent_fn)
                            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                            and n.value.id == 'memory_info' and isinstance(n.slice, ast.Constant)})
    result = dict(status='ablation_input_audit_pass', checked_at=time.time(), model_calls=0,
        users=200, completed_prediction_files_checked=600, full_prompts_reconstructed=rebuilt,
        wkg_cache_rows_matched_to_calls_and_formatter=wkg_matched,
        upstream_agent_prompt_branches=branches, upstream_memory_fields_in_full_prompt=memory_fields,
        llmzs_differs_from_empty_module_full_template_queries=zs_differs,
        retained_payloads_after_recorded_clipping=stats,
        definition='Character-span retention of each original section payload, excluding heading and outer whitespace. Empty source is distinct from clipping loss. Missing means zero characters from an originally nonempty span remain, not absence of overlapping facts elsewhere.',
        source_sha256=PINNED, audit_script_sha256=sha(__file__),
        input_sha256={name: sha(args.run/name) for name in ['run_metadata.json','validated_summary.json','calls.jsonl','wkg_cache.jsonl']},
        bundle_sha256=sha(args.bundle), result_tree_sha256=tree_hash,
        effective_prompt_map_sha256=hashlib.sha256(json.dumps(prompt_hashes,sort_keys=True).encode()).hexdigest(),
        caveats=['No new model inference or accuracy computation.',
                 'Original CKE payload is read from the accepted output; no new graph reconstruction in this audit.',
                 'Public pinned prompt generator has no separate Table2 ablation branches; this does not establish what unpublished author scripts did.',
                 'Payload retention does not prove attention, comprehension or causal score impact.'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    for key in ['run','bundle','upstream','output']:
        parser.add_argument('--'+key, type=Path)
    args = parser.parse_args()
    assert __debug__, 'Run without -O'
    if args.self_test:
        self_test()
    else:
        assert all(getattr(args,k) for k in ['run','bundle','upstream','output'])
        audit(args)
