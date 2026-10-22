"""Resume frozen AgentMove Dataset from its verified Tokyo session checkpoint.

Use --launch for a detached, bounded CPU job. No LLM calls. Only load the
trusted, checksum-pinned checkpoint; pickle is executable serialization.
"""
import argparse
import ast
from collections import Counter
import glob
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import shutil
import socket
import subprocess
import sys
import time
import types

if not __debug__:
    raise RuntimeError('Run without -O')
p = argparse.ArgumentParser()
for name in ['checkpoint', 'upstream', 'output']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--launch', action='store_true')
a = p.parse_args()
a.checkpoint, a.upstream, a.output = [x.resolve() for x in [a.checkpoint, a.upstream, a.output]]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()

assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))) >= 48 * 1024**2
assert shutil.disk_usage(a.output.parent).free >= 10 * 1024**3
if a.launch:
    a.output.mkdir(exist_ok=False)
    cmd = ['timeout', '--signal=TERM', '--kill-after=30s', '9000s', sys.executable,
           '-u', str(Path(__file__).resolve()), '--checkpoint', str(a.checkpoint),
           '--upstream', str(a.upstream), '--output', str(a.output)]
    with (a.output / 'run.log').open('x') as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True, cwd=a.output)
    launch = dict(supervisor_pid=proc.pid, started_at=time.time(), command=cmd,
                  script_sha256=sha(Path(__file__)))
    (a.output / 'launch.json').write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch))
    sys.exit(0)

started = time.time()
assert a.output.is_dir() and not (a.output / 'status.json').exists()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(19)
resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')

def no_network(*args, **kw):
    raise RuntimeError('No network in CPU preparation worker')

socket.socket.connect = no_network
socket.getaddrinfo = no_network
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
assert (pd.__version__, np.__version__, nx.__version__) == ('2.1.4', '1.26.4', '3.3')

def status(stage):
    v = dict(stage=stage, pid=os.getpid(), started_at=started, checked_at=time.time())
    (a.output / 'status.json').write_text(json.dumps(v, indent=2))
    print(json.dumps(v), flush=True)

status('checking_checkpoint_and_source')
assert sha(a.checkpoint / 'session_rows.pkl') == 'da1b75d8b62de6f9697ccee0d612d108d5a516afbcedf5263035c103de5d940a'
assert sha(a.checkpoint / 'session_manifest.csv') == '64c95b59c36ff9e86c098ed5d7e1bcdfcc985c9176651cf77c78194473c34e48'
expected_sources = {
    'processing/data.py': 'de6bd5b410f3c4e7f402869a16a4e0d3e419a9c52d24d4bf4e754def071a07b8',
    'utils.py': '8eb71891071e804bf69cbdf6aa371c267d6c2e73e030512e8fb91824a70abff4',
    'models/world_model.py': '7b251f5237b7b87fa70f7061ec26c1a88031798752048210c8dc6f7803d71f5c',
    'agent.py': 'b03c77b10350213dabf6363613039eba519fefec9b41ad3807d7df7281cf83cf',
}
trees = {}
for name, expected in expected_sources.items():
    assert sha(a.upstream / name) == expected, name
    trees[name] = ast.parse((a.upstream / name).read_text())

def execute(nodes, ns):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, 'frozen_upstream', 'exec'), ns)
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()

ns = dict(pd=pd, np=np, os=os, glob=glob, json=json, random=random, tqdm=tqdm,
          nx=nx, itertools=itertools, Counter=Counter)
utility_ns = dict(os=os)
execute([n for n in trees['utils.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'create_dir'], utility_ns)
ns['utils'] = types.SimpleNamespace(create_dir=utility_ns['create_dir'])
cls = next(n for n in trees['processing/data.py'].body if isinstance(n, ast.ClassDef) and n.name == 'Dataset')
execute([cls], ns)
obj = ns['Dataset'].__new__(ns['Dataset'])
obj.__dict__.update(dataset_name='Tokyo', trajectory_mode='trajectory_split', base_name='AgentMove',
                    history_stays=15, context_stays=6, hour_bins=72, traj_min_len=3, traj_max_len=100,
                    max_sample_trajectories=100, sample_one_traj_of_user=True, prompt_num=200,
                    use_int_venue=False, train_sample=.7, test_size=.2, seed=1234,
                    save_dir=str(a.output / 'processed'), test_dictionary={}, true_locations={}, align_dictionary={})
obj.data = pd.read_pickle(a.checkpoint / 'session_rows.pkl')
manifest = pd.read_csv(a.checkpoint / 'session_manifest.csv')
assert len(obj.data) == 1030105 and len(manifest) == 277509
n_by_user = manifest.groupby('user_id').size()
count_audit = dict(paper_table4_trajectories=112942, raw_sessions=len(manifest),
                  all_sessions_at_least4=int((manifest.stays >= 4).sum()),
                  sessions_at_least4_of_users_at_least5_raw_sessions=int(
                      ((manifest.stays >= 4) & (manifest.user_id.map(n_by_user) >= 5)).sum()),
                  interpretation='Descriptive counts only; no protocol changed to match paper')
(a.output / 'count_audit.json').write_text(json.dumps(count_audit, indent=2))
ns.update(self=obj, DL_train_trajectory_ids=[], DL_val_trajectory_ids=[], DL_test_trajectory_ids=[],
          traj_id_map={(int(r.user_id), int(r.traj_id)): int(r.DL_traj_id) for r in manifest.itertuples()},
          traj_user_map={int(r.DL_traj_id): int(r.user_id) for r in manifest.itertuples()})
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_trajectories')
branch = method.body[1].orelse[0]
split_start = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name) and n.target.id == 'user_id')
random.seed(1234)
status('official_prediction_dictionary_construction')
suffix_hash = execute(branch.body[split_start:] + method.body[2:], ns)
status('auditing_all_prediction_inputs')
cared = ['hour', 'weekday', 'venue_category_name', 'venue_id', 'admin', 'subdistrict', 'poi', 'street']
addr = ['admin', 'subdistrict', 'poi', 'street']
audited = 0
history_times = []
target_times = {}
for uid, rows in obj.data.groupby('user_id', sort=False):
    instances = obj.test_dictionary[str(uid)]
    ids = rows.traj_id.unique()
    cut = int(.8 * len(ids))
    history = rows[rows.traj_id.isin(ids[:cut])]
    sessions = {str(tid): group for tid, group in rows.groupby('traj_id', sort=False)}
    expected_ids = [str(tid) for tid in ids[cut:] if len(sessions[str(tid)]) >= 4] if len(ids) >= 3 else []
    assert list(instances) == expected_ids
    if instances:
        history_times.extend(history.datetime.astype('int64').tolist())
    for tid, instance in instances.items():
        seq = sessions[tid]
        context, target = seq.iloc[:-1], seq.iloc[-1]
        assert instance['historical_stays_long'] == history[cared].values.tolist()
        assert instance['historical_stays'] == history[cared].values.tolist()[-15:]
        assert instance['historical_addr_long'] == history[addr].values.tolist()
        assert instance['historical_addr'] == history[addr].values.tolist()[-15:]
        assert instance['historical_pos'] == history[['longitude', 'latitude']].values.tolist()[-15:]
        assert instance['context_stays'] == context[cared].values.tolist()
        assert instance['context_addr'] == context[addr].values.tolist()
        assert instance['context_pos'] == context[['longitude', 'latitude']].values.tolist()
        assert instance['target_stay'] == [target.hour, target.weekday, '<next_place_id>', '<next_place_address>']
        assert set(instance) == {'historical_stays', 'historical_pos', 'historical_addr', 'historical_stays_long',
                                 'historical_addr_long', 'context_stays', 'context_pos', 'context_addr', 'target_stay'}
        ground = obj.true_locations[str(uid)][tid]
        assert ground == dict(ground_stay=target.venue_id, ground_pos=[target.longitude, target.latitude],
                              ground_addr=target[addr].tolist())
        assert history.datetime.max() <= seq.datetime.min()
        target_times[(str(uid), tid)] = int(target.datetime.value)
        audited += 1
assert audited == 10917
social_cls = next(n for n in trees['models/world_model.py'].body if isinstance(n, ast.ClassDef) and n.name == 'SocialWorld')
execute([social_cls], ns)
status('official_collective_graph_construction')
social = ns['SocialWorld'](obj, obj.save_dir, 'Tokyo', khop=1, max_neighbors=10)
expected_edges = set()
for instances in obj.test_dictionary.values():
    if instances:
        history = next(iter(instances.values()))['historical_stays_long']
        ids = [x[3] for x in history]
        expected_edges.update(frozenset(pair) for pair in zip(ids[:-1], ids[1:]))
assert not social.graph.is_directed()
assert {frozenset(e) for e in social.graph.edges()} == expected_edges
reloaded = nx.read_gml(social.graph_file_path)
assert set(reloaded) == set(social.graph)
assert {frozenset(e) for e in reloaded.edges()} == expected_edges
for u, v, attributes in social.graph.edges(data=True):
    assert reloaded[u][v] == attributes
for n, attrs in social.graph.nodes(data=True):
    assert reloaded.nodes[n] == attrs
agent_cls = next(n for n in trees['agent.py'].body if isinstance(n, ast.ClassDef) and n.name == 'Agents')
sampler = next(n for n in agent_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'trajs_sampling')
execute([sampler], ns)
sample = types.SimpleNamespace(city_name='Tokyo', traj_min_len=3, traj_max_len=50, prompt_num=200,
                               max_sample_trajectories=100, sample_one_traj_of_user=True,
                               trajectories=[], trajectory_groups=[], known_stays={})
ns['trajs_sampling'](sample, obj.test_dictionary)
selection = [[u, t] for u, t, _ in sample.trajectories]
assert len(selection) == len({u for u, t in selection}) == 200
selection_file = a.output / 'query_selection.json'
selection_file.write_text(json.dumps(selection, indent=2))
ordered_times = np.sort(np.asarray(history_times, dtype=np.int64))
future_counts = [int(len(ordered_times)-np.searchsorted(ordered_times, target_times[(u, t)], side='right'))
                 for u, t in selection]
files = list((a.output / 'processed').iterdir()) + [selection_file, a.output / 'count_audit.json']
result = dict(status='prediction_inputs_and_graph_pass', checked_at=time.time(),
              model_calls=0, prediction_scores_available=False, audited_test_instances=audited,
              dictionary_users=len(obj.test_dictionary), users_with_test=sum(bool(v) for v in obj.test_dictionary.values()),
              selected_queries=len(selection), selected_users=200, target_masking='all instances passed',
              graph_nodes=social.graph.number_of_nodes(), graph_edges=social.graph.number_of_edges(),
              graph_directed=False, graph_edge_set_matches_history=True, graph_reload_verified=True,
              collective_input_history_records=len(ordered_times),
              selected_queries_with_later_collective_history=sum(n > 0 for n in future_counts),
              later_collective_history_records_range=[min(future_counts), max(future_counts)],
              collective_time_note='Official per-user splits retained; later records from other users may enter CKE. Not a globally time-causal evaluation.',
              count_audit=count_audit, dataset_user_min_sessions=3, paper_appendix_user_min_sessions=5,
              checkpoint_sha256=sha(a.checkpoint / 'session_rows.pkl'), upstream_sha256=expected_sources,
              script_sha256=sha(Path(__file__)), suffix_ast_sha256=suffix_hash,
              artifacts={f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in files},
              elapsed_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              limits=dict(cpu_cores=2, address_space_gib=24, cpu_seconds=7200, outer_wall_seconds=9000))
(a.output / 'result.json').write_text(json.dumps(result, indent=2))
status('completed')
print(json.dumps(result), flush=True)
