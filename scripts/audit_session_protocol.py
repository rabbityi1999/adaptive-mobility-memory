"""Audit filtering order from an existing session manifest; no model or labels.

The paper-order reconstruction is a separate diagnostic, never a replacement
for the frozen benchmark. Only aggregate counts and hashes are exported.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

EXPECTED_MANIFEST = '64c95b59c36ff9e86c098ed5d7e1bcdfcc985c9176651cf77c78194473c34e48'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cohort(groups, minimum, filter_before_split=False, minimum_after_filter=False):
    selected, histories = [], {}
    eligible_users = eligible_sessions = kept_sessions = kept_users = 0
    splits = Counter()
    for user, original in sorted(groups.items()):
        if len(original) < minimum:
            continue
        rows = [r for r in original if r['stays'] >= 4] if filter_before_split else original
        if minimum_after_filter and len(rows) < minimum:
            continue
        if not rows:
            continue
        kept_users += 1
        kept_sessions += len(rows)
        cut70, cut80 = int(.7 * len(rows)), int(.8 * len(rows))
        splits.update(train=cut70, validation=cut80-cut70, test=len(rows)-cut80)
        tests = [r for r in rows[cut80:] if r['stays'] >= 4]
        eligible_sessions += len(tests)
        if not 3 <= len(tests) <= 50:
            continue
        eligible_users += 1
        if len(selected) < 200:
            q = (user, tests[0]['traj_id'])
            selected.append(q)
            histories[q] = frozenset(r['traj_id'] for r in rows[:cut80])
    return dict(kept_users=kept_users, kept_sessions=kept_sessions,
                split_sessions=dict(splits), eligible_test_sessions=eligible_sessions,
                users_with_3_to_50_eligible_test_sessions=eligible_users,
                selected_queries=len(selected),
                selection_sha256=sha(json.dumps(selected, separators=(',', ':')).encode())), selected, histories


def compare(reference, alternate):
    _, rq, rh = reference
    _, aq, ah = alternate
    shared = set(rq) & set(aq)
    return dict(shared_users=len({u for u, _ in rq} & {u for u, _ in aq}),
                shared_queries=len(shared),
                identical_ordered_selection=rq == aq,
                shared_queries_with_changed_history=sum(rh[q] != ah[q] for q in shared))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True,
                   help='Frozen selected_queries.json from the completed baseline')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.manifest.read_bytes()
    assert sha(raw) == EXPECTED_MANIFEST, 'Unexpected frozen manifest'
    groups = defaultdict(list)
    for r in csv.DictReader(raw.decode().splitlines()):
        groups[int(r['user_id'])].append(dict(traj_id=int(r['traj_id']), stays=int(r['stays']),
                                              split=r['split'], eligible_test=r['eligible_test']))
    for rows in groups.values():
        rows.sort(key=lambda x: x['traj_id'])
        assert [r['traj_id'] for r in rows] == list(range(len(rows)))
        for i, row in enumerate(rows):
            expected = 'excluded_user' if len(rows)<3 else 'train' if i<int(.7*len(rows)) else 'validation' if i<int(.8*len(rows)) else 'test'
            assert row['split'] == expected
            assert (row['eligible_test'] == 'True') == (expected == 'test' and row['stays'] >= 4)
    variants = {
        'fixed_code_user3_split_then_test_length4': cohort(groups, 3),
        'user5_only_split_then_test_length4': cohort(groups, 5),
        'paper_order_user5_then_length4_then_split': cohort(groups, 5, True),
        'sensitivity_length4_then_user5_then_split': cohort(groups, 5, True, True),
    }
    fixed = variants['fixed_code_user3_split_then_test_length4']
    assert fixed[0]['eligible_test_sessions'] == 10917
    assert fixed[0]['selected_queries'] == 200
    selection_raw = args.selection.read_bytes()
    assert sha(selection_raw) == '526e73882604a4ebdb6632c3f3b0bfcf42ef842953299d5ee82a29fad1bddefb'
    assert fixed[1] == [tuple(map(int, pair)) for pair in json.loads(selection_raw)]
    report = dict(status='session_protocol_count_audit_pass', verification='ANALYZED',
                  manifest_sha256=sha(raw), script_sha256=sha(Path(__file__).read_bytes()),
                  frozen_selection_sha256=sha(selection_raw), frozen_selection_exact_match=True,
                  raw_users=len(groups), raw_sessions=sum(map(len, groups.values())),
                  raw_checkins=sum(r['stays'] for rows in groups.values() for r in rows),
                  paper_table4_trajectories=112942,
                  variants={name:dict(counts=value[0], versus_fixed=compare(fixed,value))
                            for name,value in variants.items()},
                  model_calls=0, elapsed_seconds=time.time()-started,
                  caveats=[
                      'Paper-order reconstruction uses existing anchored72h sessions; it does not infer another window rule.',
                      'Paper filtering order is audited separately; no thresholds are searched to fit Table4.',
                      '70/10/20 split, first200 ordered users, first eligible test session; agent-stage3to50 filter unchanged.',
                      'History comparison uses preceding80percent session membership, not prompt text or score.',
                      'No predictions, target venue IDs, raw trajectories, or user identifiers are exported.',
                      'Counts do not establish the authors actual unpublished preprocessing or a new model result.',
                  ])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
