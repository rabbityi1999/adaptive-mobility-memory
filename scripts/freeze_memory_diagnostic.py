"""Freeze users and session IDs from audited metadata without opening events.

Private identities remain outside the repository. No labels, candidate coverage,
model outputs, or target times are read. Reserve users get no query-session IDs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(users):
    return hashlib.sha256('\n'.join(sorted(users)).encode()).hexdigest()


def split_users(users, config):
    assert len(users) == len(set(users)) == sum(config['split'].values())
    key = lambda u: (hashlib.sha256((config['protocol']+':'+u).encode()).hexdigest(), int(u))
    ordered = sorted(users, key=key)
    groups, start = {}, 0
    for name in ('fit', 'development', 'reserve'):
        count = config['split'][name]
        groups[name] = ordered[start:start+count]
        start += count
    assert start == len(ordered)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'inventory-report', 'private-inventory', 'session-manifest', 'output', 'private-output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.private_output.exists()
    config = json.loads(a.config.read_text())
    report = json.loads(a.inventory_report.read_text())
    private = json.loads(a.private_inventory.read_text())
    assert report['status'] == 'registered_cohort_identity_inventory_pass'
    assert report['remaining_with_validation_session'] == 1633
    assert report['registered_excluded_hashes'] == 3562
    assert fingerprint(private['available_users']) == report['remaining_users_sha256']
    assert len(set(private['excluded_hashes'])) == 3562
    assert sha(a.session_manifest) == report['session_manifest_sha256']
    assert all(hashlib.sha256(u.encode()).hexdigest()[:16] not in private['excluded_hashes'] for u in private['available_users'])
    groups = split_users(private['available_users'], config)
    # Independent enumeration order must not alter the split.
    assert split_users(list(reversed(private['available_users'])), config) == groups
    with a.session_manifest.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    first = {}
    for row in rows:
        if row['split'] != 'validation':
            continue
        u = row['user_id']
        if u not in first or int(row['retained_rank']) < int(first[u]['retained_rank']):
            first[u] = row
    queries = {}
    for group in ('fit', 'development'):
        queries[group] = []
        for u in groups[group]:
            row = first[u]
            assert int(row['stays']) >= 4
            queries[group].append(dict(user=u, trajectory=row['traj_id'],
                retained_rank=int(row['retained_rank']),
                session_start=row['start'], personal_history_before=row['start']))
    assert 'reserve' not in queries
    payload = dict(protocol=config['protocol'], users=groups, queries=queries)
    text = json.dumps(payload, indent=2)+'\n'
    result = dict(status='memory_diagnostic_users_and_sessions_frozen',
        model_calls=0, target_event_rows_read=0, targets_materialized=0,
        counts={k:len(v) for k,v in groups.items()},
        query_session_counts={k:len(v) for k,v in queries.items()},
        reserve_queries_selected=0,
        all_pairwise_user_overlaps={a+'_'+b:len(set(groups[a])&set(groups[b]))
            for a,b in [('fit','development'),('fit','reserve'),('development','reserve')]},
        group_sha256={k:fingerprint(v) for k,v in groups.items()},
        manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),
        config_sha256=sha(a.config), inventory_report_sha256=sha(a.inventory_report),
        session_manifest_sha256=sha(a.session_manifest), script_sha256=sha(Path(__file__)),
        order_invariance_checked=True,
        limitations=config['limitations'],
        next_gate=config['next_gate'])
    with a.private_output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    assert sha(a.private_output) == result['manifest_sha256']
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
