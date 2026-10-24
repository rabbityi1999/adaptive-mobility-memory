"""Seal actual Llama3 chat inputs without loading weights or reading labels.

Raw prompts/identities stay in the private output. Aggregate report is public.
This module also defines the frozen response parser for the future runner.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def messages(item, condition, cfg):
    assert set(item) == {'user', 'trajectory', 'cutoff_ns', 'context', 'candidates', 'shared_sha256', 'memory'}
    assert condition in cfg['conditions']
    lookup = {c['poi']: c['candidate_id'] for c in item['candidates']}
    assert len(lookup) == len(item['candidates'])
    unknown = {}
    def observed_label(poi):
        if poi in lookup:
            return lookup[poi]
        if poi not in unknown:
            unknown[poi] = f'H{len(unknown)+1:02d}'
        return unknown[poi]
    context = []
    for row in item['context']:
        assert row['time_ns'] <= item['cutoff_ns']
        dt = datetime.fromtimestamp(row['time_ns'] / 1e9, tz=timezone.utc)
        context.append(dict(location_id=observed_label(row['poi']), category=row['category'],
                            time_utc=dt.isoformat(), weekday_utc=dt.strftime('%A')))
    candidates = [{k: c[k] for k in ('candidate_id', 'category', 'distance_km')} for c in item['candidates']]
    memory = item['memory'][condition]['text']
    text = (cfg['shared_header'] + '\nObserved check-ins (oldest first):\n' + canonical(context)
            + '\nCandidates (display order is not a ranking):\n' + canonical(candidates)
            + '\nAdditional memory:\n' + (memory if memory else '(none)')
            + '\nReturn the JSON array of five candidate IDs:')
    return [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': text}]


def parse_predictions(response, valid_ids):
    """Strict full-response JSON. Every invalid response has no predictions."""
    try:
        value = json.loads(response.strip())
    except (json.JSONDecodeError, AttributeError):
        return {'valid': False, 'predictions': [], 'reason': 'invalid_json'}
    if not isinstance(value, list) or len(value) != 5 or any(not isinstance(v, str) for v in value):
        return {'valid': False, 'predictions': [], 'reason': 'wrong_shape'}
    if len(set(value)) != 5:
        return {'valid': False, 'predictions': [], 'reason': 'duplicate_ids'}
    if not set(value) <= set(valid_ids):
        return {'valid': False, 'predictions': [], 'reason': 'invalid_ids'}
    return {'valid': True, 'predictions': value, 'reason': None}


def main():
    import resource
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'accepted-report', 'config', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if not __debug__:
        raise RuntimeError('Do not use -O')
    assert not a.output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    from transformers import AutoTokenizer
    started = time.time()
    cfg = json.loads(a.config.read_text())
    accepted = json.loads(a.accepted_report.read_text())
    assert accepted['status'] == 'fit_and_development_inputs_independently_validated'
    assert cfg['conditions'] == ['context', 'personal', 'population', 'fixed_fusion']
    assert cfg['phases'] == {'fit': 400, 'development': 200}
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    assert tok.eos_token_id == cfg['pad_token_id'] == 128001
    assert tok.convert_tokens_to_ids('<|eot_id|>') == 128009
    model_cfg = json.loads((a.model / 'config.json').read_text())
    assert cfg['max_sequence_tokens'] <= model_cfg['max_position_embeddings']
    a.output.mkdir()
    report = dict(status='memory_reader_cpu_preflight_pass', model_loaded=False, model_calls=0,
                  config_sha256=sha(a.config), script_sha256=sha(__file__),
                  accepted_report_sha256=sha(a.accepted_report), model_config_sha256=sha(a.model/'config.json'),
                  phases={}, label_files_opened=0, reserve_queries=0, truncated_prompts=0)
    for phase, size in cfg['phases'].items():
        root = a.input_root / phase
        r = json.loads((root/'result.json').read_text())
        v = json.loads((root/'validation.json').read_text())
        assert r == accepted['phases'][phase]['build'] and v == accepted['phases'][phase]['validation']
        assert sha(root/'result.json') == v['result_sha256']
        assert sha(root/'inputs.jsonl') == r['artifacts']['inputs.jsonl']
        for name, fingerprint in r['tokenizer_sha256'].items():
            assert sha(a.model/name) == fingerprint
        lengths = {condition: [] for condition in cfg['conditions']}
        memory_counts = Counter(); memory_tokens = Counter(); identities = set()
        target = a.output / (phase + '_prompts.jsonl')
        with (root/'inputs.jsonl').open() as f, target.open('x', encoding='utf-8') as out:
            for line in f:
                x = json.loads(line)
                key = (x['user'], x['trajectory']); assert key not in identities
                identities.add(key)
                shared = {k: val for k, val in x.items() if k not in ('shared_sha256', 'memory')}
                assert hashlib.sha256(canonical(shared).encode()).hexdigest() == x['shared_sha256']
                ids = [c['candidate_id'] for c in x['candidates']]
                assert len(ids) == len(set(ids)) and 5 <= len(ids) <= 48
                reference = None
                for condition in cfg['conditions']:
                    msg = messages(x, condition, cfg)
                    shared_text = msg[1]['content'].split('\nAdditional memory:\n')[0]
                    if reference is None:
                        reference = shared_text
                    assert shared_text == reference
                    tokens = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
                    assert len(tokens) + cfg['max_new_tokens'] <= cfg['max_sequence_tokens'], 'Prompt too long: reject, never truncate'
                    lengths[condition].append(len(tokens))
                    memory_counts[condition] += not x['memory'][condition]['rows']
                    memory_tokens[condition] += x['memory'][condition]['tokens']
                    saved = dict(user=x['user'], trajectory=x['trajectory'], condition=condition,
                                 messages=msg, input_tokens=len(tokens), valid_ids=ids,
                                 input_token_ids_sha256=hashlib.sha256(canonical(tokens).encode()).hexdigest(),
                                 source_shared_sha256=x['shared_sha256'])
                    out.write(canonical(saved)+'\n')
        assert len(identities) == size and all(len(vv) == size for vv in lengths.values())
        assert dict(memory_counts) == r['empty_memory_queries']
        assert dict(memory_tokens) == r['memory_token_totals']
        report['phases'][phase] = dict(queries=size, prompts=size*4,
            inputs_sha256=r['artifacts']['inputs.jsonl'], prompts_sha256=sha(target),
            validation_sha256=sha(root/'validation.json'), tokenizer_sha256=r['tokenizer_sha256'],
            tokens={k: dict(total=sum(vv), min=min(vv), max=max(vv), mean=sum(vv)/len(vv)) for k, vv in lengths.items()},
            shared_context_candidates_equal=True, memory_aggregates_recomputed=True)
    report.update(elapsed_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (a.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
