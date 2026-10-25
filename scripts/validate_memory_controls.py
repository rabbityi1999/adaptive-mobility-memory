"""Independently check sealed control text and old-response cache references.

Does not import the control builder, renderer, runner or evaluator. No labels,
model weights, new generations or performance metrics are read/computed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time

START = '\nAdditional memory:\n'
END = '\nReturn the JSON array of five candidate IDs:'
HEADER = 'source: candidate_id | support_count | source_share | days_since_latest_support'


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def split_text(msg):
    assert len(msg) == 2 and msg[0]['role'] == 'system' and msg[1]['role'] == 'user'
    prefix, tail = msg[1]['content'].rsplit(START, 1)
    assert tail.endswith(END)
    body = tail[:-len(END)]
    if body == '(none)': return prefix, []
    header, *lines = body.split('\n')
    assert header == HEADER and lines
    for line in lines:
        assert re.fullmatch(r'(personal|population): C\d{2} \| \d+ \| \d+\.\d{6} \| \d+\.\d{6}', line)
    return prefix, lines


def expected_source_lines(rows):
    return ['{}: {} | {} | {:.6f} | {:.6f}'.format(r['source'], r['candidate_id'], r['support_count'],
             r['source_share'], r['days_since_latest_support']) for r in rows]


def validate_control(item, row, originals, cfg):
    assert (row['user'], row['trajectory']) == (item['user'], item['trajectory'])
    condition = row['condition']; assert condition in cfg['conditions']
    origin = 'fixed_fusion' if condition == 'fusion_population_rebound' else 'personal'
    old = originals[origin]
    assert row['original_condition'] == origin
    assert row['source_shared_sha256'] == item['shared_sha256'] == old['source_shared_sha256']
    allowed = [c['candidate_id'] for c in item['candidates']]
    assert row['valid_ids'] == old['valid_ids'] == allowed
    prefix, lines = split_text(old['messages'])
    assert lines == expected_source_lines(item['memory'][origin]['rows'])
    assert row['messages'][0] == old['messages'][0]
    if condition == 'personal6':
        expected = lines[:6]
        _, fusion = split_text(originals['fixed_fusion']['messages'])
        assert expected == [line for line in fusion if line.startswith('personal: ')]
        assert row['offset'] is None
        count = len(lines)-len(expected)
    else:
        names = sorted(allowed)
        seed = '{}:{}:{}:{}'.format(cfg['protocol'], item['user'], item['trajectory'], condition)
        offset = (int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (len(names)-1)) + 1
        assert row['offset'] == offset
        expected = []; count = 0
        for line in lines:
            source, cid, rest = re.fullmatch(r'(personal|population): (C\d{2})(.*)', line).groups()
            if condition == 'personal_rebound' or source == 'population':
                new_id = names[(names.index(cid)+offset) % len(names)]
                assert new_id != cid
                expected.append(source+': '+new_id+rest); count += 1
            else:
                expected.append(line)
        assert row['input_tokens'] == old['input_tokens']
    assert row['changed_rows'] == count
    expected_msg = copy.deepcopy(old['messages'])
    body = HEADER+'\n'+'\n'.join(expected) if expected else '(none)'
    expected_msg[1]['content'] = prefix+START+body+END
    assert row['messages'] == expected_msg, 'Control changed more than prescribed memory fields'
    matches = [name for name, prior in originals.items() if prior['messages'] == row['messages']
               and prior['input_token_ids_sha256'] == row['input_token_ids_sha256']]
    if matches:
        assert row['action'] == 'reuse' and row['reuse_condition'] == matches[0], 'Wrong cache mapping'
    else:
        assert row['action'] == 'generate' and row['reuse_condition'] is None, 'False cache reuse'
    return count


def validate(a):
    from transformers import AutoTokenizer
    started=time.time()
    assert not a.output.exists()
    cfg=read(a.config);rcfg=read(a.reader_config);report=read(a.prompts/'result.json')
    pre=read(a.original_preflight);accepted=read(a.accepted_inputs);results=read(a.original_results)
    assert report['status']=='controls_cpu_preflight_pass_pending_independent_validation'
    assert cfg['phases']==rcfg['phases']=={'fit':400,'development':200}
    assert report['config_sha256']==digest(a.config)
    assert report['reader_config_sha256']==pre['config_sha256']==results['config_sha256']==digest(a.reader_config)
    assert report['script_sha256']==digest(Path(__file__).with_name('prepare_memory_controls.py'))
    assert report['renderer_sha256']==pre['script_sha256']==digest(Path(__file__).with_name('prepare_memory_reader.py'))
    assert report['accepted_inputs_sha256']==pre['accepted_report_sha256']==digest(a.accepted_inputs)
    assert report['original_preflight_sha256']==digest(a.original_preflight)
    assert report['original_results_sha256']==digest(a.original_results)
    assert pre==read(a.original_prompts/'result.json')
    assert results['status']=='memory_reader_results_independently_recomputed'
    assert read(a.original_run/'exit.json')['exit_code']==read(Path(str(a.original_run)+'.outer_exit.json'))['exit_code']==0
    status=read(a.original_run/'status.json');summary=read(a.original_run/'summary.json')
    assert status['stage']==summary['status']=='completed' and summary['model_calls']==2400
    assert status['model_calls']==2400 and status['completed_queries']==summary['queries']==600
    assert digest(a.original_run/'summary.json')==results['run_summary_sha256']
    assert digest(a.original_run/'run_metadata.json')==summary['metadata_sha256']
    meta=read(a.original_run/'run_metadata.json')
    assert meta['mode']=='inference' and meta['config_sha256']==digest(a.reader_config)
    assert meta['preflight_report_sha256']==digest(a.original_preflight)
    assert pre['model_config_sha256']==digest(a.model/'config.json')
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    all_users=set();new_total=reuse_total=0;phases={}
    for phase,n in cfg['phases'].items():
        prior_path=a.original_prompts/(phase+'_prompts.jsonl');path=a.prompts/(phase+'_prompts.jsonl')
        input_path=a.input_root/phase/'inputs.jsonl';response_path=a.original_run/(phase+'_responses.jsonl')
        seal=report['phases'][phase]
        assert digest(path)==seal['prompts_sha256']
        assert digest(prior_path)==seal['original_prompts_sha256']==pre['phases'][phase]['prompts_sha256']
        assert digest(input_path)==seal['inputs_sha256']==accepted['phases'][phase]['build']['artifacts']['inputs.jsonl']
        response_hash=digest(response_path)
        assert response_hash==results['phases'][phase]['response_sha256']==summary['responses'][phase]
        for name,value in pre['phases'][phase]['tokenizer_sha256'].items():assert digest(a.model/name)==value
        items=[json.loads(x) for x in input_path.read_text().splitlines()]
        old=[json.loads(x) for x in prior_path.read_text().splitlines()]
        new=[json.loads(x) for x in path.read_text().splitlines()]
        responses=[json.loads(x) for x in response_path.read_text().splitlines()]
        assert len(items)==n and len(old)==len(responses)==4*n and len(new)==3*n
        assert seal['queries']==n and seal['prompts']==3*n
        stats={c:dict(prompts=0,generate=0,reuse=0,null_interventions=0,changed_rows=0,total_input_tokens=0,max_input_tokens=0) for c in cfg['conditions']}
        for i,item in enumerate(items):
            assert item['user'] not in all_users; all_users.add(item['user'])
            assert set(item)=={'user','trajectory','cutoff_ns','context','candidates','shared_sha256','memory'}
            originals={}; prior_responses={}
            for j,name in enumerate(rcfg['conditions']):
                o=old[i*4+j];response=responses[i*4+j]
                assert o['condition']==name and (o['user'],o['trajectory'])==(item['user'],item['trajectory'])
                assert (response['user'],response['trajectory'],response['condition'])==(o['user'],o['trajectory'],name)
                assert response['input_token_ids_sha256']==o['input_token_ids_sha256']
                assert response['source_shared_sha256']==o['source_shared_sha256']==item['shared_sha256']
                tokens=tok.apply_chat_template(o['messages'],tokenize=True,add_generation_prompt=True)
                assert len(tokens)==o['input_tokens']==response['input_tokens'] and token_digest(tokens)==o['input_token_ids_sha256']
                assert len(response['generated_ids'])==response['output_tokens']<=rcfg['max_new_tokens']
                assert tok.decode(response['generated_ids'],skip_special_tokens=True)==response['raw']
                originals[name]=o;prior_responses[name]=response
            for j,condition in enumerate(cfg['conditions']):
                r=new[i*3+j];assert r['condition']==condition
                count=validate_control(item,r,originals,cfg)
                tokens=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
                assert len(tokens)==r['input_tokens'] and token_digest(tokens)==r['input_token_ids_sha256']
                assert len(tokens)+rcfg['max_new_tokens']<=rcfg['max_sequence_tokens']
                assert r['original_response_file_sha256']==response_hash
                if r['action']=='reuse':
                    cached=prior_responses[r['reuse_condition']]
                    assert cached['input_token_ids_sha256']==r['input_token_ids_sha256']
                s=stats[condition];s['prompts']+=1;s[r['action']]+=1;s['null_interventions']+=count==0;s['changed_rows']+=count
                s['total_input_tokens']+=len(tokens);s['max_input_tokens']=max(s['max_input_tokens'],len(tokens))
        assert stats==seal['conditions']
        new_count=sum(x['generate'] for x in stats.values());reuse_count=sum(x['reuse'] for x in stats.values())
        new_total+=new_count;reuse_total+=reuse_count
        phases[phase]=dict(queries=n,controls_checked=len(new),old_responses_checked=len(responses),
            generate=new_count,reuse=reuse_count,prompts_sha256=digest(path),response_sha256=response_hash,
            input_tokens_and_cache_mapping_verified=True,all_prescribed_text_changes_verified=True)
    assert len(all_users)==600 and new_total==report['planned_new_calls'] and reuse_total==report['planned_reuse']
    out=dict(status='independent_memory_controls_validation_pass',phases=phases,model_calls=0,
        model_loaded=False,label_files_opened=0,old_responses_checked=2400,reserve_queries=0,
        planned_new_calls=new_total,planned_reuse=reuse_total,preflight_sha256=digest(a.prompts/'result.json'),
        validator_sha256=digest(__file__),config_sha256=digest(a.config),
        original_results_sha256=digest(a.original_results),elapsed_seconds=time.time()-started,
        scope='Independent implementation in same research session, not an external reviewer. Checks full 1800 controls and 2400 sealed old responses; no new prediction or performance claim.')
    a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out))


def main():
    import resource
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input-root','accepted-inputs','original-prompts','original-preflight','original-results','original-run','reader-config','config','model','prompts','output'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();assert __debug__
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    validate(args)


if __name__=='__main__':main()
