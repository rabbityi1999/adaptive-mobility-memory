"""Build a separately labelled paper-order Tokyo input checkpoint, without LLMs.

Filter users by >=5 RAW sessions, remove sessions with <4 stays, then split
retained sessions70/10/20. Preserve original IDs and the code's80% LLM history.
Only checksum-pinned local trusted pickle input is accepted. No original files
are modified. Raw artifacts must stay off GitHub; publish only result.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if not __debug__:
        raise RuntimeError('Run without -O')
    assert not a.output.exists(), 'Refuse to overwrite an existing attempt'
    assert sha(a.checkpoint/'session_rows.pkl') == 'da1b75d8b62de6f9697ccee0d612d108d5a516afbcedf5263035c103de5d940a'
    assert sha(a.checkpoint/'session_manifest.csv') == '64c95b59c36ff9e86c098ed5d7e1bcdfcc985c9176651cf77c78194473c34e48'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def no_network(*args, **kwargs):
        raise RuntimeError('Network disabled')
    socket.socket.connect = no_network
    import pandas as pd
    assert pd.__version__ == '2.1.4'
    started = time.time()
    a.output.mkdir()
    def status(stage):
        (a.output/'status.json').write_text(json.dumps(dict(stage=stage, pid=os.getpid(), checked_at=time.time())))
    status('loading_trusted_checkpoint')
    audit = json.loads(a.audit.read_text())
    assert audit['status'] == 'session_protocol_count_audit_pass'
    assert audit['frozen_selection_exact_match']
    expected = audit['variants']['paper_order_user5_then_length4_then_split']['counts']
    df = pd.read_pickle(a.checkpoint/'session_rows.pkl')
    manifest = pd.read_csv(a.checkpoint/'session_manifest.csv')
    assert len(df) == 1030105 and len(manifest) == 277509
    assert not manifest.DL_traj_id.duplicated().any()
    raw_counts = manifest.groupby('user_id').size()
    kept = manifest[(manifest.user_id.map(raw_counts)>=5) & (manifest.stays>=4)].copy()
    kept = kept.sort_values(['user_id','traj_id'])
    # Rank only retained sessions; original traj_id is provenance, not rank.
    kept['retained_rank'] = kept.groupby('user_id').cumcount()
    sizes = kept.groupby('user_id').size()
    counts = kept.user_id.map(sizes)
    cut70, cut80 = (counts*.7).astype(int), (counts*.8).astype(int)
    kept['split'] = 'test'
    kept.loc[kept.retained_rank<cut80,'split'] = 'validation'
    kept.loc[kept.retained_rank<cut70,'split'] = 'train'
    kept['llm_history'] = kept.retained_rank<cut80
    kept['eligible_test'] = kept['split']=='test'
    filtered = df[df.DL_traj_id.isin(kept.DL_traj_id)].copy()
    assert len(filtered) == int(kept.stays.sum())
    actual_sizes = filtered.groupby('DL_traj_id').size()
    assert actual_sizes.to_dict() == kept.set_index('DL_traj_id').stays.to_dict()
    assert len(kept)==expected['kept_sessions'] and len(sizes)==expected['kept_users']
    assert kept['split'].value_counts().to_dict() == expected['split_sessions']
    test_counts = kept[kept.eligible_test].groupby('user_id').size()
    users = sorted(test_counts[(test_counts>=3)&(test_counts<=50)].index)
    assert len(users)==expected['users_with_3_to_50_eligible_test_sessions']
    selected = [(int(u), int(kept[(kept.user_id==u)&kept.eligible_test].iloc[0].traj_id)) for u in users[:200]]
    assert hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest() == expected['selection_sha256']
    status('building_selected_inputs_and_auditing_boundaries')
    cared = ['hour','weekday','venue_category_name','venue_id','admin','subdistrict','poi','street']
    addr = ['admin','subdistrict','poi','street']
    coord = ['longitude','latitude']
    dictionaries, truths, boundaries = {}, {}, []
    groups = {int(u): rows for u,rows in filtered.groupby('user_id',sort=False)}
    history_ids = set(kept.loc[kept.llm_history,'DL_traj_id'])
    for u,t in selected:
        rows = groups[u]
        history = rows[rows.DL_traj_id.isin(history_ids)]
        seq = rows[rows.traj_id==t]
        assert len(seq)>=4 and len(history)>0
        assert history.datetime.is_monotonic_increasing and seq.datetime.is_monotonic_increasing
        assert history.datetime.max()<=seq.datetime.min()
        assert not set(history.DL_traj_id)&set(seq.DL_traj_id)
        context, target = seq.iloc[:-1],seq.iloc[-1]
        data = dict(historical_stays=history[cared].values.tolist()[-15:],
                    historical_stays_long=history[cared].values.tolist(),
                    historical_addr=history[addr].values.tolist()[-15:],
                    historical_addr_long=history[addr].values.tolist(),
                    historical_pos=history[coord].values.tolist()[-15:],
                    context_stays=context[cared].values.tolist(),
                    context_addr=context[addr].values.tolist(),
                    context_pos=context[coord].values.tolist(),
                    target_stay=[target.hour,target.weekday,'<next_place_id>','<next_place_address>'])
        truth = dict(ground_stay=target.venue_id,ground_pos=target[coord].tolist(),ground_addr=target[addr].tolist())
        assert data['context_stays']==seq[cared].values.tolist()[:-1]
        assert data['target_stay'][2:]==['<next_place_id>','<next_place_address>']
        assert data['historical_stays']==data['historical_stays_long'][-15:]
        assert set(data).isdisjoint({'ground_stay','ground_pos','ground_addr'})
        dictionaries[str(u)]={str(t):data}
        truths[str(u)]={str(t):truth}
        boundaries.append(dict(user=str(u),trajectory=str(t),
                               cutoff=int(context.iloc[-1].datetime.value),target_time=int(target.datetime.value)))
    assert len(dictionaries)==len(truths)==len(selected)==200
    status('saving_separate_checkpoint')
    artifacts = {
        'test_dictionary_Tokyo_paper_order.json':dictionaries,
        'true_locations_Tokyo_paper_order.json':truths,
        'query_selection.json':[[str(u),str(t)] for u,t in selected],
        'query_boundaries.json':boundaries,
    }
    for name,value in artifacts.items():
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    filtered.to_pickle(a.output/'filtered_session_rows.pkl')
    kept.to_csv(a.output/'session_manifest.csv',index=False)
    paths = list(artifacts)+['filtered_session_rows.pkl','session_manifest.csv']
    report=dict(status='paper_order_selected_inputs_pass',verification='ANALYZED',
                condition='paper_order_user5_then_length4_then_split',model_calls=0,
                selected_users=200,selected_queries=200,total_eligible_test_sessions=int(kept.eligible_test.sum()),
                retained_users=len(sizes),retained_sessions=len(kept),retained_checkins=len(filtered),
                split_sessions=kept['split'].value_counts().to_dict(),
                selected_input_audits_passed=200,personal_time_boundary_violations=0,
                llm_history_sessions=len(history_ids),
                llm_history_records=int(filtered.DL_traj_id.isin(history_ids).sum()),
                collective_graph_completed=False,collective_time_audit_completed=False,
                prompt_identity_preflight_completed=False,prediction_scores_available=False,
                checkpoint_sha256=sha(a.checkpoint/'session_rows.pkl'),audit_sha256=sha(a.audit),
                script_sha256=sha(Path(__file__)),
                artifacts={name:dict(bytes=(a.output/name).stat().st_size,sha256=sha(a.output/name)) for name in paths},
                elapsed_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                limits=dict(cpu_cores=2,address_space_gib=16,cpu_seconds=600),
                caveats=['Separate paper-text reconstruction, not a change to the completed code baseline.',
                         'Only selected200 inputs materialized; full21032 test index preserved in manifest.',
                         'Retains current addresses and anchored72h sessions; Table4 count remains unresolved.',
                         '178 users and22 queries overlap observed baseline; not an independent holdout.',
                         'History uses preceding80percent as in LLM code; DL train is70percent.',
                         'No CKE graph or causal evidence built; do not launch model inference from this checkpoint yet.'])
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    status('completed')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
