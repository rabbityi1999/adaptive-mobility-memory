"""Prepare frozen post-hoc development controls; no weights, labels or responses.

Private outputs contain prompts and identities. Only result.json is public.
Existing reader code and all accepted artifacts remain immutable.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from prepare_memory_reader import canonical, messages, sha


def render(rows):
    if not rows:
        return ''
    return 'source: candidate_id | support_count | source_share | days_since_latest_support\n' + '\n'.join(
        f"{r['source']}: {r['candidate_id']} | {r['support_count']} | {r['source_share']:.6f} | {r['days_since_latest_support']:.6f}"
        for r in rows)


def transform(item, condition, cfg):
    """Accept only target-free original input schema. Never change shared fields."""
    assert set(item) == {'user', 'trajectory', 'cutoff_ns', 'context', 'candidates', 'shared_sha256', 'memory'}
    assert condition in cfg['conditions']
    original = 'fixed_fusion' if condition == 'fusion_population_rebound' else 'personal'
    rows = copy.deepcopy(item['memory'][original]['rows'])
    assert render(rows) == item['memory'][original]['text']
    valid = sorted(c['candidate_id'] for c in item['candidates'])
    assert len(set(valid)) == len(valid) and len(valid) >= 5
    seed = f"{cfg['protocol']}:{item['user']}:{item['trajectory']}:{condition}"
    offset = 1 + int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (len(valid)-1)
    mapping = dict(zip(valid, valid[offset:] + valid[:offset]))
    assert all(k != v for k, v in mapping.items())
    changed = 0
    if condition == 'personal6':
        rows = rows[:6]
        assert rows == [r for r in item['memory']['fixed_fusion']['rows'] if r['source'] == 'personal']
        changed = len(item['memory']['personal']['rows']) - len(rows)
    else:
        for row in rows:
            if condition == 'personal_rebound' or row['source'] == 'population':
                row['candidate_id'] = mapping[row['candidate_id']]
                changed += 1
    result = copy.deepcopy(item)
    result['memory'] = {condition: dict(rows=rows, text=render(rows))}
    assert {k: v for k, v in result.items() if k != 'memory'} == {k: v for k, v in item.items() if k != 'memory'}
    return result, original, changed, offset


def main():
    import resource
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('input-root', 'accepted-inputs', 'original-prompts', 'original-preflight', 'original-results', 'reader-config', 'config', 'model', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args()
    assert __debug__ and not a.output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]); os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    from transformers import AutoTokenizer
    start = time.time()
    cfg = json.loads(a.config.read_text()); rcfg = json.loads(a.reader_config.read_text())
    accepted = json.loads(a.accepted_inputs.read_text())
    pre = json.loads(a.original_preflight.read_text()); old = json.loads(a.original_results.read_text())
    assert cfg['phases'] == rcfg['phases'] == {'fit': 400, 'development': 200}
    assert cfg['conditions'] == ['personal6', 'personal_rebound', 'fusion_population_rebound']
    assert accepted['status'] == 'fit_and_development_inputs_independently_validated'
    assert old['status'] == 'memory_reader_results_independently_recomputed' and old['model_calls'] == 2400
    assert pre['accepted_report_sha256'] == sha(a.accepted_inputs)
    assert pre['config_sha256'] == old['config_sha256'] == sha(a.reader_config)
    assert pre['script_sha256'] == sha(Path(__file__).with_name('prepare_memory_reader.py'))
    assert pre['model_config_sha256'] == sha(a.model/'config.json')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    a.output.mkdir()
    report = dict(status='controls_cpu_preflight_pass_pending_independent_validation',
        protocol=cfg['protocol'], prior_results_observed=True, model_calls=0, model_loaded=False,
        label_files_opened=0, response_files_opened=0, reserve_queries=0,
        script_sha256=sha(__file__), renderer_sha256=pre['script_sha256'],
        config_sha256=sha(a.config), reader_config_sha256=sha(a.reader_config),
        accepted_inputs_sha256=sha(a.accepted_inputs), original_preflight_sha256=sha(a.original_preflight),
        original_results_sha256=sha(a.original_results), phases={})
    for phase, n in cfg['phases'].items():
        source = a.input_root/phase/'inputs.jsonl'
        assert sha(source) == accepted['phases'][phase]['build']['artifacts']['inputs.jsonl'] == pre['phases'][phase]['inputs_sha256']
        prior_path = a.original_prompts/(phase+'_prompts.jsonl')
        assert sha(prior_path) == pre['phases'][phase]['prompts_sha256']
        for name, value in pre['phases'][phase]['tokenizer_sha256'].items():
            assert sha(a.model/name) == value
        prior = [json.loads(x) for x in prior_path.read_text().splitlines()]
        items = [json.loads(x) for x in source.read_text().splitlines()]
        assert len(items) == n and len(prior) == 4*n
        stats = {c: dict(prompts=0, generate=0, reuse=0, null_interventions=0, changed_rows=0, total_input_tokens=0, max_input_tokens=0) for c in cfg['conditions']}
        path = a.output/(phase+'_prompts.jsonl')
        identities = set()
        with path.open('x', encoding='utf-8') as stream:
            for i, item in enumerate(items):
                key = (item['user'], item['trajectory']); assert key not in identities
                identities.add(key)
                shared = {k: v for k, v in item.items() if k not in ('shared_sha256','memory')}
                assert hashlib.sha256(canonical(shared).encode()).hexdigest() == item['shared_sha256']
                block = prior[i*4:(i+1)*4]
                assert [r['condition'] for r in block] == rcfg['conditions']
                assert all((r['user'], r['trajectory']) == key for r in block)
                by_condition = {r['condition']: r for r in block}
                for c in rcfg['conditions']:
                    assert messages(item, c, rcfg) == by_condition[c]['messages']
                for condition in cfg['conditions']:
                    changed, original, count, offset = transform(item, condition, cfg)
                    local_cfg = dict(rcfg, conditions=cfg['conditions'])
                    msg = messages(changed, condition, local_cfg)
                    ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
                    fingerprint = hashlib.sha256(canonical(ids).encode()).hexdigest()
                    original_row = by_condition[original]
                    assert msg[0] == original_row['messages'][0]
                    assert msg[1]['content'].split('\nAdditional memory:\n')[0] == original_row['messages'][1]['content'].split('\nAdditional memory:\n')[0]
                    assert len(ids)+rcfg['max_new_tokens'] <= rcfg['max_sequence_tokens']
                    if condition in cfg['exact_token_match_required']:
                        assert len(ids) == original_row['input_tokens'], 'Exact token matching failed; no alternative offset search'
                    reuse = next((r for r in block if r['messages'] == msg and r['input_token_ids_sha256'] == fingerprint), None)
                    saved = dict(user=item['user'], trajectory=item['trajectory'], condition=condition,
                        messages=msg, input_tokens=len(ids), input_token_ids_sha256=fingerprint,
                        valid_ids=original_row['valid_ids'], source_shared_sha256=item['shared_sha256'],
                        original_condition=original, changed_rows=count, offset=None if condition=='personal6' else offset,
                        action='reuse' if reuse else 'generate',
                        reuse_condition=reuse['condition'] if reuse else None,
                        original_response_file_sha256=old['phases'][phase]['response_sha256'])
                    stream.write(canonical(saved)+'\n')
                    v=stats[condition]; v['prompts']+=1; v[saved['action']]+=1
                    v['null_interventions']+=count==0; v['changed_rows']+=count
                    v['total_input_tokens']+=len(ids); v['max_input_tokens']=max(v['max_input_tokens'],len(ids))
        assert len(identities)==n
        report['phases'][phase]=dict(queries=n, prompts=n*3, inputs_sha256=sha(source),
            original_prompts_sha256=sha(prior_path), prompts_sha256=sha(path), conditions=stats)
    report.update(elapsed_seconds=time.time()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['planned_new_calls']=sum(v['generate'] for phase in report['phases'].values() for v in phase['conditions'].values())
    report['planned_reuse']=1800-report['planned_new_calls']
    assert report['planned_new_calls'] <= cfg['new_model_calls_max']
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
