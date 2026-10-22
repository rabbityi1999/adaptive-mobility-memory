import os,json,time,hashlib,collections,argparse
if not __debug__: raise RuntimeError("Run without -O: audit assertions must remain enabled")
from pathlib import Path
from types import SimpleNamespace
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
import pandas as pd,numpy as np
parser=argparse.ArgumentParser(description="Audit final L4 batch and full Tokyo POI ID coverage; no model calls.")
for name in ["run", "seed", "requests", "normalization", "output"]:
 parser.add_argument("--"+name, type=Path, required=True)
args=parser.parse_args()
RUN=args.run
assert not args.output.exists(), "Refusing to overwrite an existing audit"
assert hashlib.sha256(args.requests.read_bytes()).hexdigest() == 'df8b3dd2a9813e073bfd91368b9109dfc68f80f88428a9dee137e9aa3380f356', "Frozen request hash changed"

def read_complete(p):
 data=p.read_bytes();parts=data.splitlines(keepends=True);tail=bool(parts and not parts[-1].endswith(b'\n'))
 if tail:parts=parts[:-1]
 return [json.loads(x) for x in parts],tail
raw,raw_tail=read_complete(RUN/'raw_responses.jsonl');parsed,parsed_tail=read_complete(RUN/'parsed_responses.jsonl')
result_file=json.loads((RUN/'run_result.json').read_text())
assert result_file['status']=='chunk_complete'
assert not raw_tail and not parsed_tail
n=len(raw);assert n==len(parsed)==result_file['responses_completed']==3090
raw=raw[:n];parsed=parsed[:n]
selection=json.loads((RUN/'selection.json').read_text());expected=selection['venue_ids'];assert len(expected)==n
assert [x['venue_id'] for x in raw]==[x['venue_id'] for x in parsed]==expected
assert len(set(expected))==n
seed=(args.seed/'raw_responses.jsonl');seed_ids={json.loads(x)['venue_id'] for x in seed.read_text().splitlines()};assert not seed_ids.intersection(expected)
prior=json.loads((RUN/'run_result.json').read_text())['provenance']['completed_chunks']
assert len(seed_ids)==100 and len(seed.read_text().splitlines())==100
prior_ids=set(seed_ids)
for item in prior:
 rows=[json.loads(x) for x in (Path(item['folder'])/'raw_responses.jsonl').read_text().splitlines()]
 ids={x['venue_id'] for x in rows};assert len(ids)==len(rows)==item['responses'] and not prior_ids.intersection(ids)
 prior_ids.update(ids)
assert len(prior_ids)==80100 and not prior_ids.intersection(expected)
all_ids=prior_ids.union(expected)
source_rows=[json.loads(line)['venue_id'] for line in args.requests.open()]
assert len(source_rows)==len(set(source_rows))==len(all_ids)==83190
assert all_ids==set(source_rows)

by_id={x['venue_id']:x for x in raw};matched=set()
with args.requests.open() as f:
 for line in f:
  x=json.loads(line);key=x['venue_id']
  if key in by_id:
   a=by_id[key];assert a['api_request']==x['api_request'] and a['prompt_sha256']==x['prompt_sha256'] and a['input_tokens']==x['llama_chat_token_count'];matched.add(key)
  if len(matched)==n:break
assert len(matched)==n
join_errors=[x['venue_id'] for x in parsed if x['official_join_error'] or x['official_process_error']]
normalization=None
if not join_errors:
 frame=pd.DataFrame([x['official_joined'] for x in parsed]);address_cols=['admin','subdistrict','poi','street'];self=SimpleNamespace(data=frame.copy())
 frozen=json.loads(args.normalization.read_text())
 ns=dict(self=self,address_cols=address_cols,np=np);exec(compile(frozen['statement'],'frozen_official_normalization','exec'),ns)
 bad=[(i,k,type(v).__name__) for i,row in enumerate(self.data[address_cols].to_dict('records')) for k,v in row.items() if not isinstance(v,str)]
 normalization=dict(before_null_cells=int(frame.isna().sum().sum()),after_null_cells=int(self.data.isna().sum().sum()),nonstring_cells=bad[:20],nonstring_count=len(bad))
result=dict(status='completed_chunk_audit_pass' if not join_errors and normalization['nonstring_count']==0 else 'completed_chunk_needs_inspection',checked_at=time.time(),prefix_rows=n,selected_rows=len(selection['venue_ids']),seed_rows=100,partial_final_line_ignored=dict(raw=raw_tail,parsed=parsed_tail),request_matches=n,duplicate_pois=0,seed_overlap=0,prior_completed_rows=len(prior_ids),prior_chunk_overlap=0,join_error_count=len(join_errors),join_error_examples=join_errors[:20],finish_reasons=dict(collections.Counter(x['finish_reason'] for x in raw)),input_tokens=sum(x['input_tokens'] for x in raw),output_tokens=sum(x['output_tokens'] for x in raw),inference_seconds=sum(x['inference_seconds'] for x in raw),diagnostic_counts={k:sum(bool(x['diagnostic'][k]) for x in parsed) for k in ['strict_json','schema_ok','non_string_fields','empty_fields','non_verbatim_fields','needs_review']},normalization=normalization,full_chunk_complete=(RUN/'run_result.json').exists(),no_new_model_calls=True)
assert result['input_tokens']==result_file['input_tokens'] and result['output_tokens']==result_file['output_tokens']
assert abs(result['inference_seconds']-result_file['inference_seconds'])<1e-6
assert result['finish_reasons']=={'stop':n}
assert result['status']=='completed_chunk_audit_pass'
result['full_unique_poi_coverage']=dict(expected=83190,actual=len(all_ids),missing=0,extra=0)
result['archive_manifest']={name:dict(bytes=(RUN/name).stat().st_size,sha256=hashlib.sha256((RUN/name).read_bytes()).hexdigest()) for name in ['run_result.json','selection.json','events.jsonl','raw_responses.jsonl','parsed_responses.jsonl']}
result['archive_manifest']['run.log']=dict(bytes=(RUN.parent/'chunk017.log').stat().st_size,sha256=hashlib.sha256((RUN.parent/'chunk017.log').read_bytes()).hexdigest())
result['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
print(json.dumps(result))
