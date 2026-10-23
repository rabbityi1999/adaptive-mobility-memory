"""Frozen section-deletion ablations; preserve the accepted full prompt exactly.

These are reconstructed ablations, not unpublished original Table 2 prompts.
All raw prompts and per-user provenance remain in the external run directory.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

HEADINGS = {
    'WKG': '## The potential places from the global spatial view:\n',
    'CKE': '## The nearby places visited by other users with similar mobility pattern:\n',
    'STM': '## The personal profile and long memory:\n',
    'history': '## The history data:\n',
    'output': '## Output',
}
INSTRUCTIONS = {
    'WKG': '3. The potential places that users may visit based on an overall analysis of multi-level urban spaces.\n',
    'STM': '4. The personal profile and memory info extracted from the long trajectory history of each user.\n',
}
MASKS = {
    'recon_base': (),
    'recon_stm': ('STM',),
    'recon_wkg': ('WKG',),
    'recon_cke': ('CKE',),
    'recon_stm_wkg': ('STM', 'WKG'),
    'recon_full': ('STM', 'WKG', 'CKE'),
}
NEW_CONDITIONS = tuple(c for c in MASKS if c != 'recon_full')
SOURCE_CONDITION = 'agent_move_v6_causalCKE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def key(row):
    return row['user'] + ':' + row['trajectory']


def transform(full, condition):
    enabled = MASKS[condition]
    offsets = [full.index(h) for h in HEADINGS.values()]
    assert all(full.count(h) == 1 for h in HEADINGS.values())
    assert offsets == sorted(offsets)
    removals = []
    for index, module in enumerate(['WKG', 'CKE', 'STM']):
        if module not in enabled:
            removals.append((offsets[index], offsets[index+1]))
    for module, text in INSTRUCTIONS.items():
        assert full.count(text) == 1
        if module not in enabled:
            start = full.index(text)
            removals.append((start, start+len(text)))
    prompt = full
    for start, end in sorted(removals, reverse=True):
        prompt = prompt[:start]+prompt[end:]
    assert prompt[prompt.index(HEADINGS['history']):] == full[offsets[3]:]
    for module in ['WKG', 'CKE', 'STM']:
        assert (HEADINGS[module] in prompt) == (module in enabled)
    if condition == 'recon_full':
        assert prompt == full
    return prompt


def retention(prompt, effective):
    """Report source payload survival, with disabled/empty distinguished."""
    cutoff = len(prompt)-len(effective)
    assert effective == prompt[cutoff:]
    result = {}
    active = [(name, prompt.index(h), len(h)) for name, h in HEADINGS.items() if h in prompt]
    assert [a[1] for a in active] == sorted(a[1] for a in active)
    for module in ['WKG', 'CKE', 'STM']:
        if module not in [x[0] for x in active]:
            result[module] = 'disabled'
    for i, (name, start, length) in enumerate(active):
        a = start+length
        b = active[i+1][1] if i+1 < len(active) else len(prompt)
        while a < b and prompt[a].isspace():
            a += 1
        while b > a and prompt[b-1].isspace():
            b -= 1
        kept = max(0, b-max(a,cutoff))
        result[name] = 'empty_source' if a == b else 'missing' if not kept else 'full' if kept == b-a else 'partial'
    return result


def self_test():
    full = 'TASK\n'+''.join(INSTRUCTIONS.values())+'\n'
    for name, heading in HEADINGS.items():
        full += heading+'PAYLOAD_'+name+'\n\n'
    for condition, mask in MASKS.items():
        prompt = transform(full, condition)
        assert prompt.startswith('TASK\n') and prompt.endswith('PAYLOAD_output\n\n')
        for name in ['WKG','CKE','STM']:
            assert ('PAYLOAD_'+name in prompt) == (name in mask)
        assert all(state in ['full', 'disabled'] for state in retention(prompt,prompt).values())
    assert transform(full,'recon_full') == full
    assert retention(full, full[-1:])['STM'] == 'missing'
    print('All six masks, full identity, preserved history/output and clipping tests passed')


def prepare(source, rows, metadata, token_count, tokenizer):
    """Verify source provenance and all 200 identities before any model load."""
    from models.personal_memory import Memory
    from models.prompts import prompt_generator_agent
    from models.world_model import SpatialWorld
    from types import SimpleNamespace
    source = Path(source)
    old = read(source/'run_metadata.json')
    validated = read(source/'validated_summary.json')
    audited = read(source/'ablation_input_audit.json')
    assert read(source/'status.json')['stage'] == 'completed'
    assert read(source/'exit.json')['exit_code'] == 0
    assert validated['status'] == 'completed_output_scores_independently_recomputed'
    assert audited['status'] == 'ablation_input_audit_pass'
    for field in ['model', 'source_commit', 'upstream_sha256', 'selection_sha256', 'bundle_sha256',
                  'model_files', 'weight_files', 'external_weight_manifest_sha256', 'decoding',
                  'model_seed', 'torch', 'transformers']:
        assert metadata[field] == old[field], field
    assert audited['bundle_sha256'] == metadata['bundle_sha256']
    for filename, fingerprint in audited['input_sha256'].items():
        assert sha(source/filename) == fingerprint, filename
    selected = read(source/'selected_queries.json')
    assert sha(source/'selected_queries.json') == metadata['selection_sha256']
    assert [[r['user'], r['trajectory']] for r in rows] == selected
    assert len(rows) == len({r['user'] for r in rows}) == 200
    fingerprints = {}
    for condition, prompt_type in old['conditions'].items():
        folder = source/'results'/condition
        expected = {f'llama3-8B-local_{prompt_type}_{u}_{t}_False.json' for u,t in selected}
        assert {p.name for p in folder.glob('*.json')} == expected
        for name in expected:
            fingerprints[condition+'/'+name] = sha(folder/name)
    tree_hash = digest(json.dumps(fingerprints, sort_keys=True))
    assert tree_hash == validated['integrity']['result_tree_sha256'] == audited['result_tree_sha256']
    cached = [json.loads(x) for x in (source/'wkg_cache.jsonl').read_text().splitlines()]
    cache = {r['query']: r for r in cached}
    assert len(cached) == len(cache) == 200
    calls = [json.loads(x) for x in (source/'calls.jsonl').read_text().splitlines()]
    full_calls = {c['query']: c for c in calls if c['stage'] == SOURCE_CONDITION}
    assert len(full_calls) == 200
    plans, reused, manifests = {}, {}, []
    counts = {c: {m: Counter() for m in HEADINGS} for c in MASKS}
    for row in rows:
        query = key(row)
        ex = copy.deepcopy(row['inputs'])
        assert ex['target_stay'][2:] == ['<next_place_id>', '<next_place_address>']
        memory = Memory(ex['historical_stays_long'], ex['context_stays'], memory_lens=15)
        memory_info = memory.read_memory(row['user'], ex['target_stay'])
        spatial = SpatialWorld.get_world_info(SimpleNamespace(world_model=cache[query]['world_model'], max_lens=1000))
        assert spatial == cache[query]['world_info']
        full = prompt_generator_agent(ex,'agent_move_v6',spatial,memory_info,row['social_world_info'])
        path = source/'results'/SOURCE_CONDITION/f'llama3-8B-local_agent_move_v6_{row["user"]}_{row["trajectory"]}_False.json'
        value = read(path)
        call = full_calls[query]
        assert value['execution_status'] == 'completed' and value['true'] == row['label']['ground_stay']
        assert full == value['input'] == call['prompt_original']
        plans[query] = {}
        reused[query] = value
        for condition in MASKS:
            prompt = transform(full, condition)
            count = token_count(prompt)
            effective = prompt[-6000:] if count > 2000 else prompt
            rendered = tokenizer.apply_chat_template([dict(role='user',content=effective)], tokenize=False, add_generation_prompt=True)
            input_tokens = len(tokenizer(rendered,add_special_tokens=False)['input_ids'])
            assert input_tokens+1000 <= 8192
            if condition == 'recon_full':
                assert digest(effective) == call['prompt_sha256']
                assert effective == call['prompt_effective'] and input_tokens == call['input_tokens']
                assert count == call['gpt_token_count_before']
            retained = retention(prompt,effective)
            for module, state in retained.items():
                counts[condition][module][state] += 1
            plans[query][condition] = prompt
            manifests.append(dict(query=query,condition=condition,original_sha256=digest(prompt),
                                  effective_sha256=digest(effective),gpt_token_count=count,input_tokens=input_tokens,
                                  clipped=count>2000,retained=retained))
    report = dict(status='reconstructed_ablation_preflight_pass', queries=200,
        conditions=list(MASKS), prompt_variants_checked=len(manifests), full_identity_matches=200,
        expected_new_prediction_calls=1000, reused_full_predictions=200, new_WKG_calls=0,
        source_result_tree_sha256=tree_hash, source_run_metadata_sha256=sha(source/'run_metadata.json'),
        source_wkg_cache_sha256=sha(source/'wkg_cache.jsonl'), adapter_sha256=sha(__file__),
        prompt_manifest_sha256=digest(json.dumps(manifests,sort_keys=True)), retention_counts=counts,
        source_WKG_cost=validated['costs_by_stage']['WKG'],
        source_full_prediction_cost=validated['costs_by_stage'][SOURCE_CONDITION],
        conditions_are_reconstructed=True, source_scores_not_retuned=True)
    return plans, reused, manifests, report


if __name__ == '__main__':
    self_test()
