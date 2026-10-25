"""Independent, completion-gated scoring of sealed memory reader responses.

No runner/parser import. No interim metrics. Raw identities stay on server.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def strict_parse(raw, allowed):
    try:x=json.loads(raw.strip())
    except (ValueError, AttributeError):return dict(valid=False,predictions=[],reason='invalid_json')
    if type(x) is not list or len(x)!=5 or any(type(v) is not str for v in x):
        return dict(valid=False,predictions=[],reason='wrong_shape')
    if len(set(x))!=5:return dict(valid=False,predictions=[],reason='duplicate_ids')
    if any(v not in allowed for v in x):return dict(valid=False,predictions=[],reason='invalid_ids')
    return dict(valid=True,predictions=x,reason=None)


def scores(pred, target):
    if target is None or target not in pred:return [0.,0.,0.]
    rank=pred.index(target)+1
    return [float(rank==1),float(rank<=5),1/math.log2(rank+1) if rank<=5 else 0.]


def require_completed(root):
    root=Path(root)
    assert (root/'exit.json').exists() and read(root/'exit.json')['exit_code']==0, 'No successful exit'
    s=read(root/'status.json');r=read(root/'summary.json')
    assert s['stage']==r['status']=='completed', 'No interim or preflight performance analysis'
    assert s['model_calls']==r['model_calls']==2400 and s['completed_queries']==r['queries']==600
    assert sha(root/'run_metadata.json')==r['metadata_sha256']
    return r


def paired(a,b):
    import numpy as np
    a=np.asarray(a);b=np.asarray(b);diff=a-b
    rng=np.random.default_rng(20261024)
    boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(axis=1)
    wins=int(((a[:,0]==1)&(b[:,0]==0)).sum());losses=int(((a[:,0]==0)&(b[:,0]==1)).sum())
    m=wins+losses
    p=min(1.,2*sum(math.comb(m,k) for k in range(min(wins,losses)+1))/2**m) if m else 1.
    return dict(difference=diff.mean(axis=0).tolist(),ci95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                a_only_acc1=wins,b_only_acc1=losses,acc1_exact_p=p)


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);adjusted=[None]*len(values);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,values[i]*(len(values)-rank)));adjusted[i]=last
    return adjusted


def summarize(vectors, valid, hits):
    import numpy as np
    result={};keys=['Acc1','Acc5','NDCG5']
    mask=np.asarray(hits,dtype=bool)
    for name,rows in vectors.items():
        a=np.asarray(rows,dtype=float);assert len(a)==len(mask)
        result[name]=dict(n=len(a),metrics=dict(zip(keys,a.mean(axis=0).tolist())),format_valid=sum(valid[name]),
            candidate_hit_n=int(mask.sum()),conditional_metrics=dict(zip(keys,a[mask].mean(axis=0).tolist())) if mask.any() else None)
    comparisons={f'{a}-{b}':paired(vectors[a],vectors[b]) for a,b in [('personal','population'),('fixed_fusion','personal'),('fixed_fusion','population')]}
    adjusted=holm([x['acc1_exact_p'] for x in comparisons.values()])
    for obj,p in zip(comparisons.values(),adjusted):obj['acc1_holm_p']=p
    p=np.asarray(vectors['personal'])[:,0];g=np.asarray(vectors['population'])[:,0]
    oracle=dict(acc1=float(np.maximum(p,g).mean()),personal_only=int(((p==1)&(g==0)).sum()),
                population_only=int(((p==0)&(g==1)).sum()),both=int(((p==1)&(g==1)).sum()),
                neither=int(((p==0)&(g==0)).sum()),interpretation='Post-hoc diagnostic upper bound, not a deployable selector')
    return dict(conditions=result,paired_comparisons=comparisons,oracle_personal_population=oracle,
                metric_order=keys,bootstrap_draws=10000,seed=20261024)


def evaluate(a):
    import numpy as np
    summary=require_completed(a.run)  # Gate before opening any input or label.
    assert not a.output.exists()
    cfg=read(a.config);pre=read(a.preflight_report);accepted=read(a.accepted_inputs);meta=read(a.run/'run_metadata.json')
    assert meta['mode']=='inference' and meta['config_sha256']==pre['config_sha256']==sha(a.config)
    assert meta['preflight_report_sha256']==sha(a.preflight_report)
    assert pre['accepted_report_sha256']==sha(a.accepted_inputs)
    assert accepted['status']=='fit_and_development_inputs_independently_validated'
    totals=Counter();report=dict(status='memory_reader_results_independently_recomputed',model_calls=2400,phases={},
        config_sha256=sha(a.config),run_summary_sha256=sha(a.run/'summary.json'),evaluator_sha256=sha(__file__))
    for phase,n in cfg['phases'].items():
        root=a.input_root/phase;src=accepted['phases'][phase]
        assert read(root/'result.json')==src['build'] and read(root/'validation.json')==src['validation']
        for name in ('inputs.jsonl','labels.jsonl'):
            assert sha(root/name)==src['build']['artifacts'][name]
        path=a.prompts/(phase+'_prompts.jsonl');assert sha(path)==pre['phases'][phase]['prompts_sha256']==meta['prompts'][phase]
        response_path=a.run/(phase+'_responses.jsonl');assert sha(response_path)==summary['responses'][phase]
        inputs=[json.loads(x) for x in (root/'inputs.jsonl').read_text().splitlines()]
        labels=[json.loads(x) for x in (root/'labels.jsonl').read_text().splitlines()]
        prompts=[json.loads(x) for x in path.read_text().splitlines()]
        responses=[json.loads(x) for x in response_path.read_text().splitlines()]
        assert len(inputs)==len(labels)==n and len(prompts)==len(responses)==n*4
        assert len({x['user'] for x in inputs})==n
        vectors={k:[] for k in cfg['conditions']};valid={k:[] for k in cfg['conditions']};hits=[];reasons={k:Counter() for k in cfg['conditions']}
        for i,(x,y) in enumerate(zip(inputs,labels)):
            assert (x['user'],x['trajectory'])==(y['user'],y['trajectory'])
            mapping={c['poi']:c['candidate_id'] for c in x['candidates']};target=mapping.get(y['target']);hits.append(target is not None)
            allowed=[c['candidate_id'] for c in x['candidates']]
            for j,condition in enumerate(cfg['conditions']):
                p,r=prompts[i*4+j],responses[i*4+j]
                for v in (p,r):assert (v['user'],v['trajectory'],v['condition'])==(x['user'],x['trajectory'],condition)
                assert p['valid_ids']==allowed
                assert r['source_shared_sha256']==p['source_shared_sha256']==x['shared_sha256']
                assert r['input_token_ids_sha256']==p['input_token_ids_sha256']
                assert r['input_tokens']==p['input_tokens']
                assert len(r['generated_ids'])==r['output_tokens']<=cfg['max_new_tokens']
                parsed=strict_parse(r['raw'],allowed);assert parsed==r['parsed']
                vectors[condition].append(scores(parsed['predictions'],target));valid[condition].append(parsed['valid'])
                reasons[condition][parsed['reason'] or 'valid']+=1
                totals['input']+=r['input_tokens'];totals['output']+=r['output_tokens'];totals['invalid']+=not parsed['valid']
        assert sum(hits)==src['build']['coverage']['hit48']['hits']
        report['phases'][phase]=summarize(vectors,valid,hits)
        report['phases'][phase].update(format_reasons={k:dict(v) for k,v in reasons.items()},response_sha256=sha(response_path))
    assert totals['input']==summary['total_input_tokens'] and totals['output']==summary['total_output_tokens']
    assert totals['invalid']==summary['format_failures']
    report.update(cost=summary,caveats=['Development diagnostics; not independent confirmatory habit-change evaluation.',
        'Shared retrieval already uses both sources; condition differences concern additional evidence.',
        'Strict formatting failures count as misses; report format success separately.',
        'Holm correction is within each phase across the three planned Acc1 comparisons; no pooled fit/development test.'])
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','input-root','prompts','preflight-report','accepted-inputs','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not __debug__:raise RuntimeError('Do not use -O')
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    evaluate(a)


if __name__=='__main__':main()
