"""Explicit acceptance and zero-model prompt checks for the paper-order cohort."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    'agent.py': 'b03c77b10350213dabf6363613039eba519fefec9b41ad3807d7df7281cf83cf',
    'utils.py': '8eb71891071e804bf69cbdf6aa371c267d6c2e73e030512e8fb91824a70abff4',
    'config.py': '3ce4191e35e46158ee5fd243ab593c3e5a8c488876efec49492ba98178af5b35',
    'models/world_model.py': '7b251f5237b7b87fa70f7061ec26c1a88031798752048210c8dc6f7803d71f5c',
    'models/personal_memory.py': 'aef9e9538ae767f55d981b9814c8929b0bdab24c8e79214f950b57eed0782fd9',
    'models/prompts.py': '14b44b5b216825561eed907dbc8dcd44fea3d53e134b7ce9581f785d3f476ff4',
    'models/llm_api.py': '1e9eabd438a771609591f2fe03e6150b9bc24a841cea4418ccff81cdce5d9d59',
    'evaluate/evaluations.py': '718b07d02fabfb95e3248a9dda5e0b7d965158e1db25767187a4cd9d688084eb',
    'processing/data.py': 'de6bd5b410f3c4e7f402869a16a4e0d3e419a9c52d24d4bf4e754def071a07b8',
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(args):
    assert not args.ablation_source and args.include_official_cke and args.weight_manifest
    accepted=read(args.inputs/'result.json')
    assert accepted['status']=='paper_order_selected_inputs_pass'
    assert sha(args.inputs/'result.json')=='4bedb61d371a670701454b8b23494a986a82acbbf3a8fa78ab6c9d94cbe9e81f'
    for name,meta in accepted['artifacts'].items():
        assert (args.inputs/name).stat().st_size==meta['bytes'] and sha(args.inputs/name)==meta['sha256'], name
    validation=read(args.inputs/'validation.json')
    assert validation['status']=='serialized_paper_order_inputs_reference_pass'
    assert validation['result_sha256']==sha(args.inputs/'result.json')
    for name,fingerprint in SOURCES.items():assert sha(args.upstream/name)==fingerprint,name
    sidecar=read(args.causal_bundle.parent/'result.json')
    assert sidecar['status']=='paper_order_graph_and_causal_bundle_pass'
    assert sidecar['input_result_sha256']==sha(args.inputs/'result.json')
    assert sidecar['bundle_sha256']==sha(args.causal_bundle)=='5eff847b69c2b3f15809af9270183ddd198fb3c0aea2d024bd11ba93bc619c23'
    assert sidecar['query_selection_sha256']==sha(args.inputs/'query_selection.json')
    assert sidecar['full_city_proxy_used'] and sidecar['event_timestamp_violations']==0
    assert sha(args.causal_bundle.parent/'Tokyo_graph.gml')==sidecar['graph_sha256']
    inputs=read(args.inputs/'test_dictionary_Tokyo_paper_order.json')
    truths=read(args.inputs/'true_locations_Tokyo_paper_order.json')
    boundaries=read(args.inputs/'query_boundaries.json')
    rows=read(args.causal_bundle)
    assert len(rows)==len(boundaries)==200
    for row,b in zip(rows,boundaries):
        u,t=row['user'],row['trajectory']
        assert row['inputs']==inputs[u][t] and row['label']==truths[u][t]
        assert (u,t)==(b['user'],b['trajectory']) and row['cutoff_ns']==b['cutoff']<=b['target_time']
    return accepted,sidecar,args.causal_bundle.parent


def prompt_preflight(rows, official_social, token_count, tokenizer, conditions):
    """Capture real Agent call sites without generation or prediction file writes.

WKG inputs are exact. Prediction prompts use a labelled synthetic WKG output;
actual generated WKG and prompt sizes must still be logged at inference time.
"""
    import agent
    import models.world_model as world
    from models.personal_memory import Memory
    from models.prompts import prompt_generator
    from ablation_adapter import retention
    class Captured(Exception):pass
    class Capture:
        prompts=[]
        prediction=False
        def __init__(self,*args,**kwargs):pass
        def get_response(self,prompt_text):
            self.prompts.append(prompt_text)
            if self.prediction:raise Captured()
            return 'PREFLIGHT_WKG_PLACEHOLDER'
    old_world,old_agent=world.LLMWrapper,agent.LLMWrapper
    manifest=[]
    clipped=Counter()
    retained={}
    try:
        world.LLMWrapper=agent.LLMWrapper=Capture
        for row in rows:
            ex=copy.deepcopy(row['inputs']); query=row['user']+':'+row['trajectory']
            memory=Memory(ex['historical_stays_long'],ex['context_stays'],memory_lens=15)
            Capture.prediction=False;Capture.prompts=[]
            spatial=world.SpatialWorld('vllm','preflight','Tokyo',ex,explore_num=5)
            assert len(Capture.prompts)==2
            prompts=[('WKG_0',Capture.prompts[0]),('WKG_1',Capture.prompts[1])]
            Capture.prediction=True
            for condition,prompt_type in conditions.items():
                social=official_social if condition=='agent_move_v6_officialCKE' else SimpleNamespace(get_world_info=lambda *a:row['social_world_info'])
                social_text=social.get_world_info(ex['context_stays'][-1][3],[x[3] for x in ex['context_stays']],'address')
                expected=prompt_generator(ex,prompt_type,spatial.get_world_info(),memory.read_memory(row['user'],ex['target_stay']),social_text,{})
                predictor=agent.Agent('Tokyo','vllm','preflight',spatial,social,memory,prompt_type,'UNUSED',False,'address')
                Capture.prompts=[]
                try:predictor.predict(row['user'],row['trajectory'],ex,ex['target_stay'],{'ground_stay':'UNUSED_TRUTH'}, {})
                except Captured:pass
                else:raise AssertionError('Prediction must stop at capture boundary')
                assert Capture.prompts==[expected] and ex==row['inputs']
                prompts.append((condition,expected))
            for condition,prompt in prompts:
                count=token_count(prompt);effective=prompt[-6000:] if count>2000 else prompt
                rendered=tokenizer.apply_chat_template([dict(role='user',content=effective)],tokenize=False,add_generation_prompt=True)
                length=len(tokenizer(rendered,add_special_tokens=False)['input_ids'])
                assert length+1000<=8192
                clipped[condition]+=int(count>2000)
                item=dict(query=query,condition=condition,original_sha256=digest(prompt),effective_sha256=digest(effective),input_tokens=length,clipped=count>2000)
                if condition.startswith('agent_move'):
                    item['retained_payloads']=retention(prompt,effective)
                    for module,state in item['retained_payloads'].items():
                        retained.setdefault(condition,{}).setdefault(module,Counter())[state]+=1
                manifest.append(item)
    finally:
        world.LLMWrapper,agent.LLMWrapper=old_world,old_agent
    assert len(manifest)==1000
    return manifest,dict(status='paper_order_zero_model_preflight_pass',queries=200,
                         captured_WKG_inputs=400,captured_prediction_templates=600,model_calls=0,
                         prediction_templates_use_placeholder_WKG=True,agent_capture_identity_matches=600,
                         placeholder_clipped_counts=dict(clipped),placeholder_retention_counts=retained,
                         adapter_sha256=sha(__file__),prompt_manifest_sha256=digest(json.dumps(manifest,sort_keys=True)),
                         caveat='Placeholder WKG sizes are not actual runtime cost or clipping statistics; recheck each actual generated prompt.')
