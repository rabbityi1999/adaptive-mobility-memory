"""Run the frozen official Tokyo session prefix, then independently audit it.

Produces a local trusted pickle checkpoint, not prediction-ready dictionaries.
Run on Linux with pandas 2.1.4 / numpy 1.26.4, without Python -O.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time
import types

if not __debug__:
    raise RuntimeError('Run without -O')
p = argparse.ArgumentParser()
for name in ['source', 'upstream', 'output']:
    p.add_argument('--' + name, type=Path, required=True)
args = p.parse_args()
started = time.time()
assert int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemAvailable:'))) >= 32 * 1024**2
assert shutil.disk_usage(args.output.parent).free >= 5 * 1024**3
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(19)
resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')

def no_network(*a, **kw):
    raise RuntimeError('Network disabled in session worker')

socket.socket.connect = no_network
socket.getaddrinfo = no_network
import pandas as pd
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()

assert pd.__version__ == '2.1.4' and np.__version__ == '1.26.4'
assert sha(args.source) == '088b29d30d3f1177c3556e00a24cff4a374441ac7f644fa5e5daf911a25dd184'
sources = {
    'processing/data.py': 'de6bd5b410f3c4e7f402869a16a4e0d3e419a9c52d24d4bf4e754def071a07b8',
    'utils.py': '8eb71891071e804bf69cbdf6aa371c267d6c2e73e030512e8fb91824a70abff4',
}
for name, expected in sources.items():
    assert sha(args.upstream / name) == expected, name
args.output.mkdir(exist_ok=False)

def status(stage):
    value = dict(stage=stage, pid=os.getpid(), started_at=started,
                 checked_at=time.time(), elapsed_seconds=time.time()-started)
    (args.output / 'status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)

status('loading_official_city_data')
tree = ast.parse((args.upstream / 'processing/data.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Dataset')
loader = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_dataset')
trajectories = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_trajectories')
branch = trajectories.body[1].orelse[0]
assert ast.unparse(branch.test) == "self.trajectory_mode == 'trajectory_split'"
split_start = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name) and n.target.id == 'user_id')
# Verbatim official sorting, anchored-window grouping, globally unique session IDs,
# address normalization, and initialization of DL split lists; no API imports.
prefix = ast.Module(body=[trajectories.body[0]] + branch.body[:split_start], type_ignores=[])
prefix = ast.fix_missing_locations(prefix)
utils_tree = ast.parse((args.upstream / 'utils.py').read_text())
day_function = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'int_to_days')
utility_ns = {}
exec(compile(ast.Module(body=[day_function], type_ignores=[]), 'official_utils', 'exec'), utility_ns)
ns = dict(pd=pd, np=np, os=os, utils=types.SimpleNamespace(int_to_days=utility_ns['int_to_days']),
          CITY_DATA_DIR=str(args.source.parent))
exec(compile(ast.Module(body=[loader], type_ignores=[]), 'official_loader', 'exec'), ns)
obj = types.SimpleNamespace(dataset_name='Tokyo', trajectory_mode='trajectory_split', hour_bins=72)
obj.data = ns['get_dataset'](obj)
assert len(obj.data) == 1030105
status('official_session_grouping')
ns['self'] = obj
exec(compile(prefix, 'official_session_prefix', 'exec'), ns)
df = obj.data
status('independent_session_audit')
assert len(df) == 1030105 and df['user_id'].nunique() == 12464
assert df['venue_id'].nunique() == 83190
session_rows = []
boundary_violations = 0
for user, rows in df.groupby('user_id', sort=False):
    times = rows['datetime'].tolist()
    actual = rows['traj_id'].tolist()
    expected = []
    anchor, sid = times[0], 0
    for t in times:
        if t > anchor + pd.Timedelta(hours=72):
            sid += 1
            anchor = t
        expected.append(sid)
    assert times == sorted(times)
    assert actual == expected, 'Official session IDs disagree with anchored-window reference'
    sessions = list(rows.groupby('traj_id', sort=False))
    n = len(sessions)
    cut70, cut80 = int(.7*n), int(.8*n)
    for i, (tid, records) in enumerate(sessions):
        assert records['DL_traj_id'].nunique() == 1
        split = 'excluded_user' if n < 3 else 'train' if i < cut70 else 'validation' if i < cut80 else 'test'
        session_rows.append(dict(user_id=int(user), traj_id=int(tid),
                                 DL_traj_id=int(records['DL_traj_id'].iloc[0]),
                                 stays=len(records), start=str(records['datetime'].iloc[0]),
                                 end=str(records['datetime'].iloc[-1]), split=split,
                                 llm_history=(n >= 3 and i < cut80),
                                 eligible_test=(split == 'test' and len(records) >= 4)))
    if n >= 3:
        history = rows[rows['traj_id'] < cut80]
        test = rows[rows['traj_id'] >= cut80]
        boundary_violations += int(history['datetime'].max() > test['datetime'].min())
assert boundary_violations == 0
table = pd.DataFrame(session_rows)
assert table['DL_traj_id'].is_unique
assert int(table['stays'].sum()) == len(df)
assert table['DL_traj_id'].nunique() == df['DL_traj_id'].nunique()
status('saving_verified_checkpoint')
checkpoint = args.output / 'session_rows.pkl'
df.to_pickle(checkpoint)
table_path = args.output / 'session_manifest.csv'
table.to_csv(table_path, index=False)
result = dict(status='official_session_checkpoint_pass', checked_at=time.time(),
              checkins=len(df), users=int(df['user_id'].nunique()), pois=int(df['venue_id'].nunique()),
              sessions=len(table), split_sessions=table['split'].value_counts().to_dict(),
              eligible_test_sessions=int(table['eligible_test'].sum()),
              users_with_eligible_test=int(table.loc[table['eligible_test'], 'user_id'].nunique()),
              llm_history_sessions=int(table['llm_history'].sum()),
              temporal_boundary_violations=boundary_violations,
              anchored_window_reference_match=True, model_calls=0,
              prediction_dictionaries_completed=False, collective_graph_completed=False,
              source_sha256=sha(args.source), upstream_sha256=sources,
              script_sha256=sha(Path(__file__)), checkpoint_sha256=sha(checkpoint),
              checkpoint_bytes=checkpoint.stat().st_size, manifest_sha256=sha(table_path),
              prefix_ast_sha256=hashlib.sha256(ast.dump(prefix).encode()).hexdigest(),
              elapsed_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              limits=dict(cpu_cores=2, address_space_gib=16, cpu_seconds=1800),
              pandas_version=pd.__version__, numpy_version=np.__version__)
(args.output / 'result.json').write_text(json.dumps(result, indent=2))
status('completed')
print(json.dumps(result), flush=True)
