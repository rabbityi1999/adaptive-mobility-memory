"""CPU-only construction under frozen memory-diagnostic-v1; no model weights.

Raw inputs, labels and evidence stay outside GitHub. Publish aggregate result
only after independent validation. Target is not an argument to retrieval.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def round_robin(queues, limit=96):
    positions = [0]*len(queues)
    result, seen = [], set()
    while len(result) < limit:
        added = False
        for i, queue in enumerate(queues):
            while positions[i] < len(queue) and queue[positions[i]] in seen:
                positions[i] += 1
            if positions[i] < len(queue):
                item = queue[positions[i]]
                positions[i] += 1
                seen.add(item); result.append(item); added = True
                if len(result) == limit:
                    break
        if not added:
            break
    return result


def aggregate(frame):
    if not len(frame):
        return {}
    counts = frame.groupby('venue_id').agg(n=('time_ns','size'), latest=('time_ns','max'))
    return {p:(int(n),int(t)) for p,n,t in counts.itertuples()}


def ranked(stats):
    return sorted(stats, key=lambda p:(-stats[p][0], -stats[p][1], p))


def render_memory(rows):
    if not rows:
        return ''
    return 'source: candidate_id | support_count | source_share | days_since_latest_support\n'+'\n'.join(
        f"{r['source']}: {r['candidate_id']} | {r['support_count']} | {r['source_share']:.6f} | {r['days_since_latest_support']:.6f}" for r in rows)


def build_query(query, prefix, personal, population, tokenizer, config):
    """No target or target attributes are accepted by this function."""
    import numpy as np
    import pandas as pd
    cutoff = int(prefix.iloc[-1].time_ns)
    last_poi = prefix.iloc[-1].venue_id
    pstats, visits = aggregate(personal), aggregate(population)
    transitions = population[population.previous_poi == last_poi]
    gstats = aggregate(transitions)
    catalog = sorted(set(pstats)|set(visits))
    allowed = pd.concat([personal, population]).sort_values(['time_ns','row_id'],kind='stable')
    attrs = allowed.drop_duplicates('venue_id',keep='last').set_index('venue_id')
    assert set(attrs.index) == set(catalog)
    coords = attrs.loc[catalog, ['latitude','longitude']].to_numpy(float)
    origin = prefix.iloc[-1][['latitude','longitude']].to_numpy(float)
    lat, lon = np.radians(coords[:,0]), np.radians(coords[:,1])
    la, lo = np.radians(origin)
    angle = np.sin((lat-la)/2)**2 + np.cos(lat)*np.cos(la)*np.sin((lon-lo)/2)**2
    distance = 6371*2*np.arcsin(np.sqrt(np.clip(angle,0,1)))
    distances = dict(zip(catalog, distance.tolist()))
    nearest = sorted(catalog,key=lambda p:(distances[p],p))
    candidates96 = round_robin([ranked(pstats), ranked(gstats), nearest, ranked(visits)])
    chosen = candidates96[:48]
    u,t = query['user'],query['trajectory']
    chosen.sort(key=lambda p:hashlib.sha256(f"{config['protocol']}:{u}:{t}:{p}".encode()).hexdigest())
    names = {p:f'C{i+1:02d}' for i,p in enumerate(chosen)}
    candidates = [dict(candidate_id=names[p], poi=p,
                       category=str(attrs.loc[p,'venue_category_name']), distance_km=round(distances[p],6)) for p in chosen]
    def memory(stats, source, total):
        return [dict(source=source,candidate_id=names[p],support_count=stats[p][0],
                     source_share=round(stats[p][0]/total,6),
                     days_since_latest_support=round((cutoff-stats[p][1])/86400e9,6))
                for p in ranked(stats) if p in names]
    pm = memory(pstats,'personal',len(personal))
    gm = memory(gstats,'population',len(transitions))
    fusion=[]
    for i in range(6):
        for source in (pm,gm):
            if i < len(source):fusion.append(source[i])
    variants={}
    for name,rows in [('context',[]),('personal',pm[:12]),('population',gm[:12]),('fixed_fusion',fusion)]:
        rows=list(rows)
        text=render_memory(rows)
        while len(tokenizer.encode(text,add_special_tokens=False)) > 1024:
            rows.pop();text=render_memory(rows)
        variants[name]=dict(rows=rows,text=text,tokens=len(tokenizer.encode(text,add_special_tokens=False)))
    shared=dict(user=u,trajectory=t,cutoff_ns=cutoff,
        context=[dict(poi=r.venue_id,category=str(r.venue_category_name),time_ns=int(r.time_ns)) for r in prefix.tail(5).itertuples()],
        candidates=candidates)
    output=dict(**shared,shared_sha256=hashlib.sha256(encode(shared).encode()).hexdigest(),memory=variants)
    audit=dict(user=u,trajectory=t,cutoff_ns=cutoff,
        personal_rows=len(personal),population_rows=len(population),population_transition_rows=len(transitions),
        personal_latest_ns=int(personal.time_ns.max()) if len(personal) else None,
        population_latest_ns=int(population.time_ns.max()) if len(population) else None,
        population_latest_session_end_ns=int(population.end_ns.max()) if len(population) else None,
        personal_row_ids_sha256=hashlib.sha256(encode(personal.row_id.tolist()).encode()).hexdigest(),
        population_row_ids_sha256=hashlib.sha256(encode(population.row_id.tolist()).encode()).hexdigest(),
        candidates96=candidates96, catalog_size=len(catalog),
        catalog_sha256=hashlib.sha256(encode(catalog).encode()).hexdigest())
    return output,audit,set(catalog)


def main():
    import resource
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','config','split','split-report','model','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--phase',choices=['fit','development'],required=True)
    p.add_argument('--accepted-fit',type=Path)
    a=p.parse_args()
    if not __debug__:raise RuntimeError('Do not run with -O')
    assert not a.output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(1500,1500))
    import pandas as pd
    from transformers import AutoTokenizer
    started=time.time()
    cfg=json.loads(a.config.read_text());split=json.loads(a.split.read_text());seal=json.loads(a.split_report.read_text())
    assert sha(a.config)==seal['config_sha256'] and sha(a.split)==seal['manifest_sha256']
    assert sha(a.inputs/'session_manifest.csv')==seal['session_manifest_sha256']
    assert sha(a.inputs/'filtered_session_rows.pkl')=='b44b72bcd0f4d17d921156449c6aad13dea19535ec9b0b38deff63ed68d65d4f'
    if a.phase=='development':
        assert a.accepted_fit
        previous=json.loads((a.accepted_fit/'result.json').read_text())
        v=json.loads((a.accepted_fit/'validation.json').read_text())
        assert v['status']=='independent_memory_input_validation_pass'
        assert v['result_sha256']==sha(a.accepted_fit/'result.json')
        assert previous['script_sha256']==sha(__file__) and previous['config_sha256']==sha(a.config)
    a.output.mkdir()
    def status(stage,done=0):
        (a.output/'status.json').write_text(json.dumps(dict(stage=stage,phase=a.phase,queries_done=done,pid=os.getpid(),checked_at=time.time())))
    status('loading_trusted_inputs')
    df=pd.read_pickle(a.inputs/'filtered_session_rows.pkl').reset_index(drop=True)
    df['row_id']=df.index;df['time_ns']=df.datetime.astype('int64')
    mf=pd.read_csv(a.inputs/'session_manifest.csv')
    ends=dict(zip(mf.DL_traj_id,pd.to_datetime(mf['end'],utc=True).astype('int64')))
    train=df[df.DL_traj_id.isin(mf.loc[mf.split=='train','DL_traj_id'])].copy()
    train['end_ns']=train.DL_traj_id.map(ends)
    train=train.sort_values(['user_id','traj_id','time_ns','row_id'],kind='stable')
    train['previous_poi']=train.groupby('DL_traj_id').venue_id.shift(1)
    reserve={int(u) for u in split['users']['reserve']}
    pool=train[~train.user_id.isin(reserve)]
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    assert tokenizer.eos_token_id==128001
    meta_names=['tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    tokenizer_hashes={n:sha(a.model/n) for n in meta_names if (a.model/n).exists()}
    totals=Counter();mem_tokens={name:[] for name in cfg['reader_conditions']};empty=Counter()
    selected=split['queries'][a.phase]
    with (a.output/'inputs.jsonl').open('x') as fi,(a.output/'labels.jsonl').open('x') as fl,(a.output/'audit.jsonl').open('x') as fa:
        for n,q in enumerate(selected):
            u,t=int(q['user']),int(q['trajectory'])
            seq=df[(df.user_id==u)&(df.traj_id==t)].sort_values(['time_ns','row_id'],kind='stable')
            assert len(seq)>=4 and seq.DL_traj_id.nunique()==1
            assert seq.iloc[0].datetime==pd.Timestamp(q['session_start'])
            prefix=seq.iloc[:-1].copy();cutoff=int(prefix.iloc[-1].time_ns)
            start=int(pd.Timestamp(q['personal_history_before']).value)
            personal=train[(train.user_id==u)&(train.time_ns<start)]
            population=pool[(pool.user_id!=u)&(pool.end_ns<cutoff)]
            assert not set(population.user_id)&(reserve|{u})
            assert population.time_ns.lt(cutoff).all() and personal.time_ns.lt(start).all()
            item,audit,catalog=build_query(q,prefix,personal,population,tokenizer,cfg)
            # Extraction of the held-out event occurs only after input creation.
            target=seq.iloc[-1]
            label=dict(user=q['user'],trajectory=q['trajectory'],target=target.venue_id,target_time_ns=int(target.time_ns))
            assert label['target_time_ns']>=cutoff
            totals['queries']+=1;totals['catalog_hit']+=target.venue_id in catalog
            for k in (24,48,96):totals[f'hit{k}']+=target.venue_id in audit['candidates96'][:k]
            for name,v in item['memory'].items():
                assert len(v['rows'])<=12 and v['tokens']<=1024
                mem_tokens[name].append(v['tokens']);empty[name]+=not v['rows']
            fi.write(encode(item)+'\n');fl.write(encode(label)+'\n');fa.write(encode(audit)+'\n')
            if n%20==0:status('building',n+1)
    result=dict(status='memory_inputs_built_pending_independent_validation',phase=a.phase,model_calls=0,queries=len(selected),
        coverage={k:dict(hits=v,n=len(selected),fraction=v/len(selected)) for k,v in totals.items() if k!='queries'},
        empty_memory_queries=dict(empty),memory_token_totals={k:sum(v) for k,v in mem_tokens.items()},
        max_memory_tokens={k:max(v) for k,v in mem_tokens.items()},
        config_sha256=sha(a.config),split_sha256=sha(a.split),script_sha256=sha(__file__),
        source_pickle_sha256=sha(a.inputs/'filtered_session_rows.pkl'),session_manifest_sha256=sha(a.inputs/'session_manifest.csv'),
        tokenizer_sha256=tokenizer_hashes,elapsed_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifacts={n:sha(a.output/n) for n in ('inputs.jsonl','labels.jsonl','audit.jsonl')},
        caveats=['Coverage is not prediction accuracy; shared candidates already use both history sources.',
                 'Bulk source checkpoint loaded, but no reserve query selected or reserve records used for population.',
                 'Upper-bound row/token budget, not exact realized cost matching.'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2));status('built_pending_validation',len(selected))
    print(json.dumps(dict(status=result['status'],phase=a.phase,queries=len(selected),elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':
    main()
