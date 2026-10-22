"""Join verified L4 outputs with Tokyo check-ins using frozen upstream AST.

No model calls. Run on Linux in the existing CPU preprocessing environment.
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

if not __debug__:
    raise RuntimeError('Run without -O')

parser = argparse.ArgumentParser()
for name in ['source', 'upstream', 'seed', 'chunks', 'output']:
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
started = time.time()
available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:')))
assert available >= 16 * 1024 * 1024, 'Need 16 GiB available memory'
assert shutil.disk_usage(args.output.parent).free >= 2 * 1024**3
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(19)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')

def no_network(*a, **kw):
    raise RuntimeError('Network disabled in CPU join worker')

socket.socket.connect = no_network
socket.getaddrinfo = no_network
import pandas as pd

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert sha(args.source) == 'ace610a71cd5d1669cb46fdd29cc08a50be6f96b27790f906b8e835f8eb67d8d'
assert sha(args.upstream) == 'c0d13994cd1cd54af280404b5fb09efd3925726e0c5bce1f1fdb74d4a02d2512'
assert pd.__version__ == '2.1.4', pd.__version__
args.output.mkdir(exist_ok=False)
(args.output / 'started.json').write_text(json.dumps({'pid': os.getpid(), 'started_at': started}))
folders = [args.seed] + [args.chunks / f'chunk{i:03d}' for i in range(1, 18)]
addresses, joined, manifest = {}, {}, {}
for i, folder in enumerate(folders):
    result = json.loads((folder / 'run_result.json').read_text())
    assert result['status'] == ('completed_100' if i == 0 else 'chunk_complete')
    path = folder / 'parsed_responses.jsonl'
    n = 0
    with path.open() as f:
        for line in f:
            assert line.endswith('\n'), 'Incomplete JSONL'
            row = json.loads(line)
            key = 'Tokyo_' + row['venue_id']
            assert key not in addresses
            assert not row['official_process_error'] and not row['official_join_error']
            addresses[key] = row['official_repaired']
            joined[key] = row['official_joined']
            n += 1
    assert n == result['responses_completed'] == (100 if i == 0 else 3090 if i == 17 else 5000)
    manifest['seed' if i == 0 else f'chunk{i:03d}'] = {'rows': n, 'parsed_sha256': sha(path)}
assert len(addresses) == 83190
source = pd.read_csv(args.source)
assert len(source) == 1030105 and source['venue_id'].nunique() == 83190
assert source['user'].nunique() == 12464
assert set('Tokyo_' + x for x in source['venue_id']) == set(addresses)
address_file = args.output / 'Tokyo_addr_dict.json'
with address_file.open('x', encoding='utf-8') as f:
    json.dump(addresses, f, ensure_ascii=False)

# Execute exactly the upstream normalization/join/export statements, excluding
# all API imports, generation loops and existence-based skip logic.
tree = ast.parse(args.upstream.read_text(encoding='utf-8'))
normalize = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'get_normalize_city_name')
main = next(n for n in tree.body if isinstance(n, ast.If))
loop = next(n for n in main.body if isinstance(n, ast.For))
start = next(i for i, n in enumerate(loop.body)
             if isinstance(n, ast.Assign) and ast.unparse(n).startswith('city_addr_dict = json.load('))
module = ast.fix_missing_locations(ast.Module(body=[normalize] + loop.body[start:], type_ignores=[]))
ns = dict(os=os, pd=pd, json=json, city='Tokyo', DATASET='TIST2015',
          NO_ADDRESS_TRAJ_DIR=str(args.source.parent), CITY_DATA_DIR=str(args.output),
          file_path=str(address_file))
exec(compile(module, str(args.upstream), 'exec'), ns)
data = ns['city_data']
assert len(data) == len(source)
pd.testing.assert_frame_equal(data[list(source.columns)], source)
mapping = {'admin': 'administrative', 'subdistrict': 'subdistrict', 'poi': 'poi', 'street': 'street'}
for key, value in ns['city_addr_dict'].items():
    assert {col: value.get(field, '') for col, field in mapping.items()} == joined[key]
output_csv = args.output / 'Tokyo_filtered.csv'
roundtrip = pd.read_csv(output_csv)
pd.testing.assert_frame_equal(roundtrip[list(source.columns)], source, check_exact=False,
                              rtol=1e-12, atol=1e-12)
assert len(roundtrip) == 1030105 and roundtrip['venue_id'].nunique() == 83190
result = dict(status='full_city_join_pass', checked_at=time.time(), checkins=len(data),
              pois=data['venue_id'].nunique(), users=data['user'].nunique(), dropped_checkins=0,
              source_columns_preserved=True, address_entries=len(addresses),
              model_calls=0, session_construction_completed=False,
              elapsed_seconds=time.time()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              pandas_version=pd.__version__, source_sha256=sha(args.source),
              upstream_sha256=sha(args.upstream), script_sha256=sha(Path(__file__)),
              input_batches=manifest, output_csv_sha256=sha(output_csv),
              output_csv_bytes=output_csv.stat().st_size, address_dict_sha256=sha(address_file),
              join_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
              limits=dict(cpu_cores=2, address_space_gib=8, cpu_seconds=900))
with (args.output / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
