"""Offline sealed-prompt runner. No labels, prompt changes, retries or resume.

Use --preflight-only for CPU verification; normal execution requires a fresh
GPU gate and the same cooperative project GPU lock used by AgentMove jobs.
"""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import traceback
import hashlib

from prepare_memory_reader import canonical, parse_predictions, sha


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def hardware(gpu, cfg):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, free, util = [x.strip() for x in raw.split(',')]
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert int(free) >= cfg['minimum_free_gpu_memory_mib'] and int(util) < cfg['maximum_gpu_utilization_exclusive'], 'GPU resource gate not met'
    assert available >= cfg['minimum_available_host_ram_gib']*1024**2, 'Host RAM gate not met'
    return dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(util), available_host_kib=available, checked_at=time.time())


def verify_seals(args, cfg):
    pre = json.loads((args.prompts/'result.json').read_text())
    assert pre == json.loads(args.preflight_report.read_text())
    assert pre['status'] == 'memory_reader_cpu_preflight_pass' and pre['config_sha256'] == sha(args.config)
    assert pre['script_sha256'] == sha(Path(__file__).with_name('prepare_memory_reader.py'))
    assert pre['model_config_sha256'] == sha(args.model/'config.json')
    plans = {}
    for phase, n in cfg['phases'].items():
        seal = pre['phases'][phase]
        path = args.prompts/(phase+'_prompts.jsonl')
        assert sha(path) == seal['prompts_sha256']
        for name, fingerprint in seal['tokenizer_sha256'].items():
            assert sha(args.model/name) == fingerprint
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows) == n*len(cfg['conditions'])
        identities = []
        for k in range(n):
            block = rows[k*4:(k+1)*4]
            assert [r['condition'] for r in block] == cfg['conditions']
            key = (block[0]['user'], block[0]['trajectory']); identities.append(key)
            assert all((r['user'], r['trajectory']) == key for r in block)
            assert len({r['source_shared_sha256'] for r in block}) == 1
            assert all(r['valid_ids'] == block[0]['valid_ids'] for r in block)
        assert len(set(identities)) == len({u for u, _ in identities}) == n
        plans[phase] = rows
    weights = json.loads(args.weight_manifest.read_text())
    assert weights['status'] == 'local_full_weight_fingerprints_recorded'
    index = json.loads((args.model/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values()) == set(weights['files'])
    for name, evidence in weights['files'].items():
        current = (args.model/name).stat()
        assert current.st_size == evidence['bytes'] and current.st_mtime_ns == evidence['mtime_ns']
    for name, evidence in weights['metadata_files'].items():
        assert sha(args.model/name) == evidence['sha256']
    return pre, plans, weights


def execute(args):
    import fcntl
    cfg = json.loads(args.config.read_text())
    assert cfg['conditions'] == ['context', 'personal', 'population', 'fixed_fusion']
    assert cfg['phases'] == {'fit':400, 'development':200} and not cfg['do_sample']
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.preflight_only else str(args.gpu),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]); os.nice(19)
    def no_network(*a, **k):raise RuntimeError('Network disabled in reader worker')
    socket.socket.connect = no_network; socket.socket.connect_ex = no_network; socket.getaddrinfo = no_network
    started = time.time(); progress = dict(stage='verifying', model_calls=0, completed_queries=0, pid=os.getpid())
    def status(stage):
        progress.update(stage=stage, checked_at=time.time())
        write(args.output/'status.json', progress)
    status('verifying')
    pre, plans, weights = verify_seals(args, cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for rows in plans.values():
        for r in rows:
            ids = tokenizer.apply_chat_template(r['messages'], tokenize=True, add_generation_prompt=True)
            assert len(ids) == r['input_tokens'] and len(ids)+cfg['max_new_tokens'] <= cfg['max_sequence_tokens']
            assert hashlib.sha256(canonical(ids).encode()).hexdigest() == r['input_token_ids_sha256']
    metadata = dict(protocol=cfg['protocol'], config_sha256=sha(args.config), preflight_report_sha256=sha(args.preflight_report),
        runner_sha256=sha(__file__), parser_sha256=sha(Path(__file__).with_name('prepare_memory_reader.py')),
        weight_manifest_sha256=sha(args.weight_manifest), model_metadata=weights['metadata_files'],
        weight_shards=weights['files'], expected_calls=2400, model_calls=0, label_files_opened=0,
        weight_check='Prior full shard hashes retained; current sizes/mtimes and metadata hashes checked; no full shard rehash this run',
        prompts={p:v['prompts_sha256'] for p,v in pre['phases'].items()}, mode='preflight' if args.preflight_only else 'inference')
    if args.preflight_only:
        write(args.output/'run_metadata.json', metadata)
        write(args.output/'summary.json', dict(status='runner_cpu_preflight_pass', prompts=2400,
            model_calls=0, model_loaded=False, elapsed_seconds=time.time()-started, metadata_sha256=sha(args.output/'run_metadata.json')))
        status('preflight_pass');return
    assert args.gpu in cfg['authorized_gpus']
    lock = Path(f'/tmp/adaptive_mobility_gpu_{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata['hardware'] = hardware(args.gpu, cfg)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, GenerationConfig
    torch.set_num_threads(2); random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    metadata.update(torch_version=torch.__version__, transformers_version=transformers.__version__)
    status('loading_model')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
        device_map={'':'cuda:0'}, local_files_only=True).eval()
    metadata['attention_implementation'] = model.config._attn_implementation
    write(args.output/'run_metadata.json', metadata)
    generation = GenerationConfig(do_sample=False, num_beams=cfg['num_beams'], max_new_tokens=cfg['max_new_tokens'],
        eos_token_id=cfg['eos_token_ids'], pad_token_id=cfg['pad_token_id'], use_cache=True)
    torch.cuda.reset_peak_memory_stats()
    total_input=total_output=errors=0
    for phase, rows in plans.items():
        with (args.output/(phase+'_responses.jsonl')).open('x') as stream:
            for i, row in enumerate(rows):
                progress.update(phase=phase, phase_prompts_done=i)
                status('generating')
                ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
                tensor = torch.tensor([ids], dtype=torch.long, device='cuda:0')
                torch.cuda.synchronize(); tic=time.perf_counter()
                with torch.inference_mode():
                    result=model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=generation)
                torch.cuda.synchronize()
                generated=result[0,len(ids):].tolist()
                raw=tokenizer.decode(generated, skip_special_tokens=True)
                parsed=parse_predictions(raw, row['valid_ids'])
                saved=dict(user=row['user'], trajectory=row['trajectory'], condition=row['condition'], raw=raw,
                    parsed=parsed, input_tokens=len(ids), output_tokens=len(generated), generated_ids=generated,
                    seconds=time.perf_counter()-tic, max_new_tokens_reached=len(generated)>=cfg['max_new_tokens'],
                    input_token_ids_sha256=row['input_token_ids_sha256'], source_shared_sha256=row['source_shared_sha256'])
                stream.write(canonical(saved)+'\n');stream.flush()
                progress['model_calls']+=1;progress['completed_queries']+=int((i+1)%4==0)
                total_input+=len(ids);total_output+=len(generated);errors+=not parsed['valid']
                del result,tensor
                progress['phase_prompts_done']=i+1;status('generating')
    assert progress['model_calls']==2400 and progress['completed_queries']==600
    write(args.output/'summary.json', dict(status='completed', model_calls=2400, queries=600, format_failures=errors,
        total_input_tokens=total_input, total_output_tokens=total_output, elapsed_seconds=time.time()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), metadata_sha256=sha(args.output/'run_metadata.json'),
        responses={p:sha(args.output/(p+'_responses.jsonl')) for p in plans}))
    status('completed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prompts','preflight-report','config','model','weight-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=range(4),default=0)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if not __debug__:raise RuntimeError('Do not use -O')
    assert not a.output.exists(), 'Refuse overwrite or automatic rerun/resume'
    a.output.mkdir()
    try:
        execute(a)
    except BaseException:
        write(a.output/'failure.json',dict(traceback=traceback.format_exc(), at=time.time()))
        write(a.output/'exit.json',dict(exit_code=1));raise
    write(a.output/'exit.json',dict(exit_code=0))


if __name__=='__main__':main()
