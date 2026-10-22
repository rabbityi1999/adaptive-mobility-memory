"""Bounded, offline Llama3 paired baseline using pinned AgentMove modules.

The causal CKE variant is deliberately labelled separately from the original
static-graph protocol. Raw data, prompts and responses remain in --output.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

CONDITIONS = {'agent_move_v6_causalCKE': 'agent_move_v6', 'llmzs': 'llmzs'}
MODEL_NAME = 'llama3-8B-local'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('upstream', 'inputs', 'model', 'output', 'token-cache'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--causal-bundle', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--gpu-lock-dir', type=Path, default=Path('/tmp'))
    parser.add_argument('--weight-manifest', type=Path)
    parser.add_argument('--include-official-cke', action='store_true',
                        help='Add explicitly non-causal static-graph official-code control')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Validate hashes, imports, tokenizer and GPT cache without loading model or inference')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    result = parser.parse_args()
    for key, value in vars(result).items():
        if isinstance(value, Path):
            setattr(result, key, value.resolve())
    return result


def gpu_snapshot(gpu):
    fields = 'index,uuid,name,memory.free,memory.total,utilization.gpu'
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=' + fields,
                                   '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, name, free, total, utilization = [x.strip() for x in raw.split(',')]
    result = dict(index=int(index), uuid=uuid, name=name, free_mib=int(free),
                  total_mib=int(total), utilization=int(utilization), checked_at=time.time())
    if result['free_mib'] < 19456 or result['utilization'] >= 10:
        raise RuntimeError('GPU does not meet free-memory/utilization gate: ' + str(result))
    return result


def command(mode):
    args = [x for x in sys.argv[1:] if x not in ('--launch', '--supervise', '--worker')]
    return [sys.executable, '-u', str(Path(__file__).resolve()), *args, mode]


def preflight(args):
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))
    if available < 32 * 1024**2:
        raise RuntimeError('At least 32 GiB available host RAM is required')
    return gpu_snapshot(args.gpu)


def run(args):
    import fcntl
    if args.include_official_cke:
        CONDITIONS['agent_move_v6_officialCKE'] = 'agent_move_v6'
    # Cooperating project jobs share this non-blocking lock; other jobs are
    # protected by a fresh hardware check immediately before model loading.
    lock = (args.gpu_lock_dir / ('adaptive_mobility_gpu_%d.lock' % args.gpu)).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hardware = preflight(args)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(19)
    # No RLIMIT_AS: CUDA reserves large virtual mappings unrelated to host RSS.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      TIKTOKEN_CACHE_DIR=str(args.token_cache),
                      nominatim_deploy_server_address='unused-cached-addresses')
    def no_network(*a, **kw):
        raise RuntimeError('Network disabled in baseline worker')
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.getaddrinfo = no_network
    started = time.time()
    progress = dict(stage='verifying_inputs', started_at=started, pid=os.getpid(),
                    finished_queries=0, model_calls=0, gpu=args.gpu)
    def status(stage, **extra):
        progress.update(stage=stage, checked_at=time.time(), **extra)
        write(args.output / 'status.json', progress)
        print(json.dumps(progress), flush=True)
    status('verifying_inputs')
    accepted = json.loads((args.inputs / 'result.json').read_text())
    assert accepted['status'] == 'prediction_inputs_and_graph_pass'
    for name, evidence in accepted['artifacts'].items():
        path = args.inputs / name
        if not path.exists():
            path = args.inputs / 'processed' / name
        assert path.stat().st_size == evidence['bytes'] and sha(path) == evidence['sha256'], name
    for name, fingerprint in accepted['upstream_sha256'].items():
        assert sha(args.upstream / name) == fingerprint, name
    sidecar = json.loads((args.causal_bundle.parent / 'result.json').read_text())
    assert sidecar['status'] == 'causal_bundle_pass'
    assert sha(args.causal_bundle) == sidecar['bundle_sha256']
    assert sidecar['query_selection_sha256'] == sha(args.inputs / 'query_selection.json')
    rows = json.loads(args.causal_bundle.read_text())
    selected = json.loads((args.inputs / 'query_selection.json').read_text())
    assert [[r['user'], r['trajectory']] for r in rows] == selected
    assert len(rows) == len({r['user'] for r in rows}) == 200
    assert all(isinstance(r['social_world_info'], str) for r in rows)
    assert all(e['edge_observed_ns'] <= r['cutoff_ns'] and e['attribute_source_ns'] <= r['cutoff_ns']
               for r in rows for e in r['evidence'])
    assert all(r['inputs']['target_stay'][2:] == ['<next_place_id>', '<next_place_address>'] for r in rows)
    source_files = ['agent.py', 'utils.py', 'config.py', 'models/world_model.py',
                    'models/personal_memory.py', 'models/prompts.py', 'models/llm_api.py',
                    'evaluate/evaluations.py', 'processing/data.py']
    metadata = dict(model=args.model.name, model_directory=str(args.model), model_seed=100,
                    source_commit='61a3e47667f2862c81478bfa442595b766f2f4d6',
                    upstream_sha256={name: sha(args.upstream / name) for name in source_files},
                    script_sha256=sha(__file__), input_result_sha256=sha(args.inputs / 'result.json'),
                    bundle_sha256=sha(args.causal_bundle), causal_audit=sidecar,
                    selection_sha256=sha(args.inputs / 'query_selection.json'),
                    hardware=hardware, conditions=CONDITIONS, denominator=200,
                    limits=dict(cpu_cores=2, nice=19, outer_wall_seconds=21600,
                                gpu_memory_fraction=.8, host_available_gate_gib=32,
                                host_address_space_limit=None),
                    decoding=dict(dtype='float16', sampling=False, batch_size=1, max_new_tokens=1000,
                                  eos_token_ids=[128001, 128009], gpt_token_limit=2000, character_tail=6000),
                    caveat='Time-filtered CKE + local Llama3 variant; not paper-level strict replication',
                    model_files={f.name: dict(bytes=f.stat().st_size, sha256=sha(f))
                                 for f in args.model.iterdir() if f.suffix == '.json'},
                    weight_files={f.name: dict(bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
                                  for f in args.model.iterdir() if f.suffix in ('.bin', '.safetensors')})
    if args.weight_manifest:
        metadata['external_weight_manifest_sha256'] = sha(args.weight_manifest)
        weight_evidence = json.loads(args.weight_manifest.read_text())
        assert weight_evidence['status'] == 'local_full_weight_fingerprints_recorded'
        for name, evidence in weight_evidence['files'].items():
            current = (args.model / name).stat()
            assert current.st_size == evidence['bytes'] and current.st_mtime_ns == evidence['mtime_ns'], name
        index = json.loads((args.model / 'model.safetensors.index.json').read_text())
        assert set(index['weight_map'].values()) == set(weight_evidence['files'])
        for name, evidence in weight_evidence['metadata_files'].items():
            assert sha(args.model / name) == evidence['sha256'], name
        metadata['external_weight_manifest_note'] = 'Weight shard sizes/mtimes and all model metadata hashes match prior full-hash manifest; weight shards not rehashed this run'
    write(args.output / 'run_metadata.json', metadata)
    write(args.output / 'selected_queries.json', selected)
    sys.path.insert(0, str(args.upstream))
    import config
    config.EXP_CITIES = ['Tokyo']
    config.PROCESSED_DIR = str(args.inputs / 'processed')
    from utils import token_count
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    metadata.update(torch=torch.__version__, transformers=transformers.__version__)
    write(args.output / 'run_metadata.json', metadata)
    random.seed(100)
    torch.manual_seed(100)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    class LocalLLM:
        stage = 'initialization'
        query = ''
        calls = 0

        def __init__(self, model_name=MODEL_NAME, platform=None):
            self.model_name = model_name

        def get_response(self, prompt_text):
            original = prompt_text
            count = token_count(original)
            if count > 2000:
                prompt_text = prompt_text[-6000:]
            rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt_text)],
                                                     tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda:0')
            length = encoded.input_ids.shape[1]
            record = dict(query=self.query, stage=self.stage, prompt_original=original,
                          prompt_effective=prompt_text, prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
                          gpt_token_count_before=count, clipped=count > 2000, input_tokens=length,
                          started_at=time.time(), batch_size=1)
            append(args.output / 'call_attempts.jsonl', record)
            assert length + 1000 <= model.config.max_position_embeddings
            torch.cuda.synchronize()
            tic = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**encoded, do_sample=False, max_new_tokens=1000,
                                         eos_token_id=[128001, 128009], pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            generated = outputs[0, length:].tolist()
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            record.update(output=answer, output_tokens=len(generated), seconds=time.perf_counter()-tic,
                          max_new_tokens_reached=len(generated) >= 1000)
            append(args.output / 'calls.jsonl', record)
            LocalLLM.calls += 1
            status(self.stage, model_calls=LocalLLM.calls, query=self.query)
            return answer

    import models.llm_api as api
    api.LLMWrapper = LocalLLM
    import models.world_model as world
    import agent
    from models.personal_memory import Memory
    from evaluate.evaluations import PredictionEvaluator
    world.LLMWrapper = LocalLLM
    agent.LLMWrapper = LocalLLM
    official_social = None
    if args.include_official_cke:
        # SocialWorld finds the accepted GML and loads it; empty dataset cannot
        # supply an alternative graph. This is diagnostic, non-causal evidence.
        assert (args.inputs / 'processed' / 'Tokyo_graph.gml').is_file()
        official_social = world.SocialWorld(SimpleNamespace(get_generated_datasets=lambda: ({}, {})),
                          str(args.inputs / 'processed'), 'Tokyo', khop=1, max_neighbors=10)
    token_count('offline cache check')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert tokenizer.eos_token_id == 128001
    assert tokenizer.convert_tokens_to_ids('<|eot_id|>') == 128009
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if args.preflight_only:
        status('preflight_pass', model_loaded=False, model_calls=0)
        return

    def result_path(condition, row):
        return args.output / 'results' / condition / (f'{MODEL_NAME}_{CONDITIONS[condition]}_{row["user"]}_{row["trajectory"]}_False.json')

    for condition in CONDITIONS:
        (args.output / 'results' / condition).mkdir(parents=True)
        for row in rows:
            write(result_path(condition, row), dict(input='', output={}, prediction=[], reason='',
                  true=row['label']['ground_stay'], execution_status='not_attempted'))

    def evaluate(complete):
        report = dict(complete=complete, denominator=200, conditions={}, elapsed_seconds=time.time()-started,
                      model_calls=LocalLLM.calls, causal_CKE=True,
                      caveat=metadata['caveat'])
        for condition, prompt_type in CONDITIONS.items():
            evaluator = PredictionEvaluator('rule', str(args.output / 'results' / condition), False, prompt_type)
            assert len(evaluator.combined_data) == 200
            official = evaluator.compute_combined_top_accuracies()
            ranks, errors, empty, pending, invalid, nonfive = [], 0, 0, 0, 0, 0
            for row in rows:
                value = json.loads(result_path(condition, row).read_text())
                evaluator.combined_data['_current_row'] = value
                predictions = evaluator.get_prediction_values('_current_row', value, False)
                del evaluator.combined_data['_current_row']
                label = str(value['true']).lower()
                ranks.append(predictions.index(label)+1 if label in predictions else math.inf)
                errors += value.get('execution_status') == 'error'
                pending += value.get('execution_status') == 'not_attempted'
                empty += not predictions
                invalid += sum(not re.fullmatch('[a-f0-9]{24}', p) for p in predictions)
                nonfive += len(predictions) != 5
            report['conditions'][condition] = dict(n=200, Acc1=sum(r==1 for r in ranks)/200,
                Acc5=sum(r<=5 for r in ranks)/200,
                NDCG5=sum(1/math.log2(r+1) for r in ranks if r<=5)/200,
                official_ndcg_untruncated=official[5], official_metrics=official,
                execution_errors=errors, not_attempted=pending, empty_predictions=empty,
                invalid_ids=invalid, predictions_not_exactly_five=nonfive)
        calls_path = args.output / 'calls.jsonl'
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        report['cost'] = dict(input_tokens=sum(r['input_tokens'] for r in calls),
                            output_tokens=sum(r['output_tokens'] for r in calls),
                            inference_seconds=sum(r['seconds'] for r in calls),
                            clipped_calls=sum(r['clipped'] for r in calls),
                            WKG_calls_shared_between_conditions=sum(r['stage']=='WKG' for r in calls))
        write(args.output / ('evaluation.json' if complete else 'partial_evaluation.json'), report)

    def systemic(error):
        return any(s in str(error).lower() for s in ('cuda', 'out of memory', 'cublas', 'cudnn', 'device-side'))

    def failure(condition, row, error):
        path = result_path(condition, row)
        value = json.loads(path.read_text())
        value.update(execution_status='error', error_type=type(error).__name__, error=str(error))
        write(path, value)
        append(args.output / 'errors.jsonl', dict(query=[row['user'], row['trajectory']],
               condition=condition, stage=LocalLLM.stage, error_type=type(error).__name__, error=str(error)))

    try:
        status('loading_model')
        metadata['gpu_immediately_before_load'] = gpu_snapshot(args.gpu)
        write(args.output / 'run_metadata.json', metadata)
        torch.cuda.set_per_process_memory_fraction(.8, 0)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
        for index, row in enumerate(rows):
            LocalLLM.query = row['user'] + ':' + row['trajectory']
            LocalLLM.stage = 'WKG'
            ex = copy.deepcopy(row['inputs'])
            try:
                memory = Memory(ex['historical_stays_long'], ex['context_stays'], memory_lens=15)
                social = SimpleNamespace(get_world_info=lambda *unused, text=row['social_world_info']: text)
            except Exception as error:
                for condition in CONDITIONS:
                    failure(condition, row, error)
                if systemic(error):
                    raise
                status('query_error', finished_queries=index+1)
                continue
            wkg_failed = False
            try:
                before = LocalLLM.calls
                spatial = world.SpatialWorld('vllm', MODEL_NAME, 'Tokyo', ex, explore_num=5)
                assert LocalLLM.calls-before == 2
                append(args.output / 'wkg_cache.jsonl', dict(query=LocalLLM.query,
                       world_model=spatial.world_model, world_info=spatial.get_world_info()))
            except Exception as error:
                for condition in CONDITIONS:
                    if condition != 'llmzs':
                        failure(condition, row, error)
                if systemic(error):
                    raise
                wkg_failed = True
                spatial = SimpleNamespace(get_world_info=lambda: '')
            for condition, prompt_type in CONDITIONS.items():
                if wkg_failed and condition != 'llmzs':
                    continue
                LocalLLM.stage = condition
                try:
                    selected_social = official_social if condition == 'agent_move_v6_officialCKE' else social
                    predictor = agent.Agent('Tokyo', 'vllm', MODEL_NAME, spatial, selected_social, memory,
                        prompt_type, str(args.output / 'results' / condition), False, 'address')
                    predictor.predict(row['user'], row['trajectory'], ex, ex['target_stay'], row['label'], {})
                    path = result_path(condition, row)
                    value = json.loads(path.read_text())
                    value['execution_status'] = 'completed'
                    write(path, value)
                except Exception as error:
                    failure(condition, row, error)
                    if systemic(error):
                        raise
            status('query_completed', finished_queries=index+1)
        evaluate(True)
        status('completed', peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
    except BaseException:
        evaluate(False)
        status('failed', traceback=traceback.format_exc())
        raise


def main():
    if not __debug__:
        raise RuntimeError('Run without -O')
    args = arguments()
    if args.launch:
        preflight(args)
        args.output.mkdir(parents=False, exist_ok=False)
        environment = dict(os.environ, PYTHONHASHSEED='100')
        with (args.output / 'run.log').open('x') as log:
            process = subprocess.Popen(command('--supervise'), stdin=subprocess.DEVNULL,
                          stdout=log, stderr=log, start_new_session=True, env=environment, cwd=args.output)
        launch = dict(supervisor_pid=process.pid, started_at=time.time(), script_sha256=sha(__file__),
                      command=command('--supervise'))
        write(args.output / 'launch.json', launch)
        print(json.dumps(launch))
    elif args.supervise:
        proc = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30s', '21600s', *command('--worker')])
        code = proc.wait()
        write(args.output / 'exit.json', dict(exit_code=code, ended_at=time.time(), timeout=code in (124, 137)))
        sys.exit(code)
    else:
        if not args.worker:
            raise RuntimeError('Use --launch so timeout, fresh output and deterministic hash seed apply')
        assert os.environ.get('PYTHONHASHSEED') == '100'
        assert not (args.output / 'status.json').exists()
        try:
            run(args)
        except BaseException:
            write(args.output / 'fatal.json', dict(ended_at=time.time(), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
