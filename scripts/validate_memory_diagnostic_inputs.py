"""Validate saved diagnostic inputs against source events without generation.

All queries: identity/time/catalog/memory/token and coverage recomputation.
First/middle/last: independent Counter/deque retrieval reconstruction.
No builder import; only an aggregate report is emitted.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def encode(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def stats(frame):
    counts=Counter(frame.venue_id)
    latest={}
    for p,t in zip(frame.venue_id,frame.time_ns):latest[p]=max(latest.get(p,t),t)
    return {p:(n,int(latest[p])) for p,n in counts.items()}


def order(s):
    return sorted(s,key=lambda p:(-s[p][0],-s[p][1],p))


def main():
    import resource
    p=argparse.ArgumentParser()
    for k in ('run','inputs','config','split','model'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if not __debug__:raise RuntimeError('Do not run with -O')
    assert not (a.run/'validation.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1500,1500))
    import pandas as pd
    from transformers import AutoTokenizer
    started=time.time();r=json.loads((a.run/'result.json').read_text())
    assert r['status']=='memory_inputs_built_pending_independent_validation'
    assert json.loads((a.run.parent/(a.run.name+'.exit.json')).read_text())['exit_code']==0
    for name,digest in r['artifacts'].items():assert sha(a.run/name)==digest
    assert sha(a.config)==r['config_sha256'] and sha(a.split)==r['split_sha256']
    assert sha(a.inputs/'filtered_session_rows.pkl')==r['source_pickle_sha256']
    assert sha(a.inputs/'session_manifest.csv')==r['session_manifest_sha256']
    for name,digest in r['tokenizer_sha256'].items():assert sha(a.model/name)==digest
    cfg=json.loads(a.config.read_text());split=json.loads(a.split.read_text());phase=r['phase']
    inputs=[json.loads(x) for x in (a.run/'inputs.jsonl').read_text().splitlines()]
    labels=[json.loads(x) for x in (a.run/'labels.jsonl').read_text().splitlines()]
    audits=[json.loads(x) for x in (a.run/'audit.jsonl').read_text().splitlines()]
    queries=split['queries'][phase];assert len(inputs)==len(labels)==len(audits)==len(queries)==r['queries']
    df=pd.read_pickle(a.inputs/'filtered_session_rows.pkl').reset_index(drop=True)
    df['row_id']=df.index;df['time_ns']=df.datetime.astype('int64')
    mf=pd.read_csv(a.inputs/'session_manifest.csv');train_ids=set(mf.loc[mf.split=='train','DL_traj_id'])
    ends=dict(zip(mf.DL_traj_id,pd.to_datetime(mf['end'],utc=True).astype('int64')))
    train=df[df.DL_traj_id.isin(train_ids)].sort_values(['user_id','traj_id','time_ns','row_id'],kind='stable').copy()
    train['end_ns']=train.DL_traj_id.map(ends)
    train['previous_poi']=train.groupby('DL_traj_id').venue_id.shift(1)
    reserve={int(u) for u in split['users']['reserve']}
    pool=train[~train.user_id.isin(reserve)]
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    totals=Counter();full_indices={0,len(queries)//2,len(queries)-1};max_tokens=Counter()
    for index,(x,y,audit,q) in enumerate(zip(inputs,labels,audits,queries)):
        assert set(x)=={'user','trajectory','cutoff_ns','context','candidates','shared_sha256','memory'}
        assert x['user']==y['user']==q['user'] and x['trajectory']==y['trajectory']==q['trajectory']
        u,t=int(q['user']),int(q['trajectory']);assert u not in reserve
        seq=df[(df.user_id==u)&(df.traj_id==t)].sort_values(['time_ns','row_id'],kind='stable')
        target=seq.iloc[-1];prefix=seq.iloc[:-1];cut=int(prefix.iloc[-1].time_ns)
        assert y['target']==target.venue_id and y['target_time_ns']==int(target.time_ns)>=cut
        assert x['cutoff_ns']==cut and q['session_start']==str(seq.iloc[0].datetime)
        context=[dict(poi=v.venue_id,category=str(v.venue_category_name),time_ns=int(v.time_ns)) for v in prefix.tail(5).itertuples()]
        assert x['context']==context
        shared={k:v for k,v in x.items() if k not in ('shared_sha256','memory')}
        assert x['shared_sha256']==hashlib.sha256(encode(shared).encode()).hexdigest()
        start=int(pd.Timestamp(q['personal_history_before']).value)
        own=train[(train.user_id==u)&(train.time_ns<start)]
        pop=pool[(pool.user_id!=u)&(pool.end_ns<cut)]
        assert not set(pop.user_id)&(reserve|{u}) and pop.time_ns.lt(cut).all()
        assert own.time_ns.lt(start).all() and set(pop.DL_traj_id)<=train_ids
        assert len(own)==audit['personal_rows'] and len(pop)==audit['population_rows']
        for name,frame in [('personal',own),('population',pop)]:
            assert audit[name+'_row_ids_sha256']==hashlib.sha256(encode(frame.row_id.tolist()).encode()).hexdigest()
        transition=pop[pop.previous_poi==context[-1]['poi']]
        pstat,gstat=stats(own),stats(transition)
        catalog=sorted(set(own.venue_id)|set(pop.venue_id))
        assert len(catalog)==audit['catalog_size'] and hashlib.sha256(encode(catalog).encode()).hexdigest()==audit['catalog_sha256']
        c96=audit['candidates96'];assert len(c96)==len(set(c96))<=96 and set(c96)<=set(catalog)
        chosen=sorted(c96[:48],key=lambda p:hashlib.sha256(f"{cfg['protocol']}:{q['user']}:{q['trajectory']}:{p}".encode()).hexdigest())
        assert [v['poi'] for v in x['candidates']]==chosen
        ids={v['poi']:v['candidate_id'] for v in x['candidates']}
        assert list(ids.values())==[f'C{i+1:02d}' for i in range(len(chosen))]
        expected={}
        for name,s,total in [('personal',pstat,len(own)),('population',gstat,len(transition))]:
            expected[name]=[dict(source=name,candidate_id=ids[p],support_count=s[p][0],source_share=round(s[p][0]/total,6),days_since_latest_support=round((cut-s[p][1])/86400e9,6)) for p in order(s) if p in ids]
        fused=[]
        for k in range(6):
            for source in ('personal','population'):
                if k<len(expected[source]):fused.append(expected[source][k])
        expected.update(context=[],fixed_fusion=fused)
        assert set(x['memory'])==set(cfg['reader_conditions'])
        for name,v in x['memory'].items():
            rows=expected[name][:12]
            def render(rows):
                if not rows:return ''
                return 'source: candidate_id | support_count | source_share | days_since_latest_support\n'+'\n'.join(f"{a['source']}: {a['candidate_id']} | {a['support_count']} | {a['source_share']:.6f} | {a['days_since_latest_support']:.6f}" for a in rows)
            while len(tok.encode(render(rows),add_special_tokens=False))>1024:rows.pop()
            assert rows==v['rows'] and render(rows)==v['text']
            assert len(tok.encode(v['text'],add_special_tokens=False))==v['tokens']<=1024
            assert len(rows)<=12;max_tokens[name]=max(max_tokens[name],v['tokens'])
        if index in full_indices:
            combined=pd.concat([own,pop]).sort_values(['time_ns','row_id'],kind='stable')
            meta=combined.drop_duplicates('venue_id',keep='last').set_index('venue_id')
            lat1,lon1=map(math.radians,(float(prefix.iloc[-1].latitude),float(prefix.iloc[-1].longitude)))
            def dist(p):
                lat2,lon2=map(math.radians,(float(meta.loc[p,'latitude']),float(meta.loc[p,'longitude'])))
                h=math.sin((lat1-lat2)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon1-lon2)/2)**2
                return 12742*math.asin(math.sqrt(min(1,max(0,h))))
            distances={p:dist(p) for p in catalog}
            queues=[deque(order(pstat)),deque(order(gstat)),deque(sorted(catalog,key=lambda p:(distances[p],p))),deque(order(stats(pop)))]
            reference=[]
            while any(queues) and len(reference)<96:
                for queue in queues:
                    while queue and queue[0] in reference:queue.popleft()
                    if queue:reference.append(queue.popleft())
                    if len(reference)==96:break
            assert reference==c96
            for v in x['candidates']:
                assert v['category']==str(meta.loc[v['poi'],'venue_category_name'])
                assert abs(v['distance_km']-distances[v['poi']])<=.00000051
        totals['catalog_hit']+=y['target'] in set(catalog)
        for k in (24,48,96):totals[f'hit{k}']+=y['target'] in c96[:k]
    for key,value in totals.items():assert value==r['coverage'][key]['hits']
    result=dict(status='independent_memory_input_validation_pass',phase=phase,queries=len(queries),model_calls=0,
        all_time_and_source_exclusions_pass=True,all_memory_rows_and_tokens_recomputed=True,
        all_labels_and_contexts_matched=True,all_catalogs_recomputed=True,
        full_retrieval_reference_indices=sorted(full_indices),coverage_matches=True,
        result_sha256=sha(a.run/'result.json'),validator_sha256=sha(__file__),elapsed_seconds=time.time()-started,
        limitation='Full queue order and candidate attributes independently reconstructed for first/middle/last; all queries validated for catalog, history provenance, memory and budget.')
    (a.run/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
