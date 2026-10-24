"""Inventory candidate users against registered prior cohorts; no predictions.

Publish only aggregate JSON. The private output contains IDs and must stay off
GitHub. This checks query/donor identity overlap, not exclusion from historical
population memory. It neither selects targets nor freezes a new test set.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def user_hash(user):
    return hashlib.sha256(str(user).encode()).hexdigest()[:16]


def fingerprint(users):
    return hashlib.sha256('\n'.join(sorted(users)).encode()).hexdigest()


def identities(value):
    """Include top-level query IDs and nested donor IDs, but no location IDs."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('user', 'donor_user') or (key == 'donor' and isinstance(item, str)):
                if not isinstance(item, str) or len(item) != 16:
                    raise ValueError('Unexpected hashed user schema')
                int(item, 16)
                found.add(item)
            elif isinstance(item, (dict, list)):
                found.update(identities(item))
    elif isinstance(value, list):
        for item in value:
            found.update(identities(item))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['pilot', 'inventory', 'output', 'private-output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.private_output.exists()
    files = ['prepared/inputs.jsonl', 'retrieval_followup/val_rows.jsonl',
             'retrieval_followup/test_rows.jsonl', 'ranking_v2/fit_inputs.jsonl',
             'ranking_v2/tune_inputs.jsonl',
             'ranking_v2/validation_v2_1/fit_inputs.jsonl',
             'ranking_v2/validation_v2_1/tune_inputs.jsonl',
             'memory_suite/data/fit_inputs.jsonl', 'memory_suite/data/tune_inputs.jsonl',
             'memory_suite/data/test_inputs.jsonl',
             'memory_suite/matched_followup/data/test_inputs.jsonl',
             'memory_suite/permutation_followup/original/data/fit_inputs.jsonl',
             'memory_suite/permutation_followup/original/data/tune_inputs.jsonl',
             'memory_suite/permutation_followup/rotated/data/fit_inputs.jsonl',
             'memory_suite/permutation_followup/rotated/data/tune_inputs.jsonl']
    groups, evidence = {}, {}
    for rel in files:
        path = a.pilot/rel
        users = set()
        row_count = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row_count += 1
                    users.update(identities(json.loads(line)))
        assert users or row_count == 0, rel
        groups[rel] = users
        evidence[rel] = dict(sha256=sha(path), users=len(users), rows=row_count)
    rel = 'ranking_v2/user_hashes.json'
    reserved = json.loads((a.pilot/rel).read_text())
    groups[rel] = {u for phase in reserved.values() for users in phase.values() for u in users}
    evidence[rel] = dict(sha256=sha(a.pilot/rel), users=len(groups[rel]))
    for name in ['code_queries.json', 'paper_queries.json']:
        path = a.inventory/name
        pairs = json.loads(path.read_text())
        assert len(pairs) == 200 and len({u for u,t in pairs}) == 200
        groups[name] = {user_hash(u) for u,t in pairs}
        evidence[name] = dict(sha256=sha(path), users=len(groups[name]))
    # Validate raw string-ID normalization and hash uniqueness against the
    # existing two-city cache. Use only user identity fields from this cache.
    source = a.pilot/'data/two_cities.json'
    data = json.loads(source.read_text())
    raw_users = {row[0] for rows in data['events'].values() for row in rows}
    assert all(isinstance(u,str) and str(int(u)) == u for u in raw_users)
    assert len({user_hash(u) for u in raw_users}) == len(raw_users)
    tokyo = {row[0] for row in data['events']['Tokyo']}
    del data
    path = a.inventory/'paper_session_manifest.csv'
    assert sha(path) == '91d408472a3eab5724504d555f29e341c68d7658ae6197408b4867baf6598f0f'
    with path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 88611
    retained = {r['user_id'] for r in rows}
    counts = Counter(r['user_id'] for r in rows if r['eligible_test'] == 'True')
    eligible = {u for u,n in counts.items() if 3 <= n <= 50}
    assert len(retained) == 7360 and len(eligible) == 2409
    excluded = set.union(*groups.values())
    remaining = {u for u in eligible if user_hash(u) not in excluded}
    # Require at least one validation session for any prospective dev query.
    val_users = {r['user_id'] for r in rows if r['split'] == 'validation'}
    available = remaining & val_users
    assert not {user_hash(u) for u in available} & excluded
    result = dict(status='registered_cohort_identity_inventory_pass', model_calls=0,
        new_targets_selected=0, new_test_set_frozen=False,
        retained_users=len(retained), eligible_users=len(eligible),
        registered_excluded_hashes=len(excluded),
        eligible_excluded=len(eligible-remaining), remaining_eligible_users=len(remaining),
        remaining_with_validation_session=len(available),
        eligible_not_in_legacy_tokyo=len(eligible-tokyo),
        remaining_users_sha256=fingerprint(available),
        per_source_eligible_overlap={k:len({user_hash(u) for u in eligible}&v) for k,v in groups.items()},
        sources=evidence, session_manifest_sha256=sha(path), identity_cache_sha256=sha(source),
        script_sha256=sha(Path(__file__)),
        limitations=['Only listed cohorts and their recorded donors/reservations were excluded; external historical studies are not certified.',
                    'Population-memory and previous city preprocessing may include these users; not a strict background-data holdout.',
                    'Availability does not establish habit drift, source complementarity, statistical power or a final split.',
                    'Eligibility uses total session counts as in reproduction; prospective sampling must be separately defined.'])
    a.private_output.write_text(json.dumps(dict(available_users=sorted(available,key=int), excluded_hashes=sorted(excluded)),indent=2))
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','per_source_eligible_overlap')},indent=2))


if __name__ == '__main__':
    main()
