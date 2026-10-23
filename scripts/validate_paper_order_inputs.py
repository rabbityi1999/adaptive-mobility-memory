"""Read back saved paper-order inputs and check them against trusted raw rows.

No models or scoring. Raw input/output stays local; writes aggregate report only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','checkpoint','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not __debug__:
        raise RuntimeError('Run without -O')
    assert not a.output.exists()
    os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    import pandas as pd
    result=json.loads((a.inputs/'result.json').read_text())
    assert result['status']=='paper_order_selected_inputs_pass'
    assert json.loads((a.inputs/'status.json').read_text())['stage']=='completed'
    for name,meta in result['artifacts'].items():
        assert sha(a.inputs/name)==meta['sha256'],name
    assert sha(a.checkpoint/'session_rows.pkl')==result['checkpoint_sha256']=='da1b75d8b62de6f9697ccee0d612d108d5a516afbcedf5263035c103de5d940a'
    raw=pd.read_pickle(a.checkpoint/'session_rows.pkl')
    tests=json.loads((a.inputs/'test_dictionary_Tokyo_paper_order.json').read_text())
    truth=json.loads((a.inputs/'true_locations_Tokyo_paper_order.json').read_text())
    selection=json.loads((a.inputs/'query_selection.json').read_text())
    boundaries=json.loads((a.inputs/'query_boundaries.json').read_text())
    assert len(selection)==len(tests)==len(truth)==len(boundaries)==200
    assert set(tests)==set(truth)=={u for u,t in selection}
    cared=['hour','weekday','venue_category_name','venue_id','admin','subdistrict','poi','street']
    addr=['admin','subdistrict','poi','street']; pos=['longitude','latitude']
    raw_groups={str(u):r for u,r in raw.groupby('user_id',sort=False)}
    for (u,t),boundary in zip(selection,boundaries):
        rows=raw_groups[u]
        sessions=[(sid,r) for sid,r in rows.groupby('traj_id',sort=True)]
        assert len(sessions)>=5
        retained=[(sid,r) for sid,r in sessions if len(r)>=4]
        cut=int(.8*len(retained))
        eligible=retained[cut:]
        assert 3<=len(eligible)<=50 and int(t)==eligible[0][0]
        history=pd.concat([r for sid,r in retained[:cut]])
        seq=eligible[0][1]; context=seq.iloc[:-1]; target=seq.iloc[-1]
        assert history.datetime.max()<=seq.datetime.min()
        expected={'target_stay':[target.hour,target.weekday,'<next_place_id>','<next_place_address>']}
        for key,columns in [('stays',cared),('addr',addr),('pos',pos)]:
            expected['historical_'+key]=history[columns].values.tolist()[-15:]
            expected['context_'+key]=context[columns].values.tolist()
            if key!='pos':expected['historical_'+key+'_long']=history[columns].values.tolist()
        assert set(tests[u])==set(truth[u])=={t}
        assert tests[u][t]==expected, 'Serialized input differs from raw-row reference'
        assert truth[u][t]==dict(ground_stay=target.venue_id,ground_pos=target[pos].tolist(),ground_addr=target[addr].tolist())
        assert boundary==dict(user=u,trajectory=t,cutoff=int(context.iloc[-1].datetime.value),target_time=int(target.datetime.value))
    out=dict(status='serialized_paper_order_inputs_reference_pass',verification='ANALYZED',
             checked_inputs=200,checked_artifacts=len(result['artifacts']),model_calls=0,
             result_sha256=sha(a.inputs/'result.json'),validator_sha256=sha(Path(__file__)),
             checks=['saved artifact hashes','raw-user count before filtering','retained session order',
                     'first eligible test query per selected user','all input and truth fields',
                     'target placeholders','context cutoff timestamps','personal time boundary'],
             limitation='No collective graph, rendered prompts, or new predictions verified.')
    with a.output.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
