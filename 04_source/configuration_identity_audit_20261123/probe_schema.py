from pathlib import Path
import sys,json,hashlib,traceback,time,os
R=Path('/mnt/project/perun2601449/dp_llm_cfd');O=R/'results/llm_config_integrity_20261123_v1';sys.path.insert(0,str(R/'test/llm'));sys.path.insert(0,str(R/'test/runtime/ldc_schema_v30/site'))
import torch
from transformers import AutoTokenizer
from ldc_main_schema_v33 import prepare_tokenizer,build_constraint,make_parser
from ldc_main_contract_v33 import messages,parse_reply
from ldc_main_common_v33 import read,save

def run():
 torch.set_num_threads(4);assert not torch.cuda.is_available();A=R/'results/ldc_main_20261104_v33';E=R/'results/llama_extension_20261119_v1';methods=['qwen3_8b','phi4_mini','granite33_8b','Llama31_8B','ministral3_8b'];rows=[]
 for method in methods:
  started=time.monotonic();root=E if method=='Llama31_8B' else A;protocol=read(root/'protocol.json');spec=next(s for s in protocol['models'] if s['label']==method);proof=read(Path(protocol['preparation'])/(method+'.json'));state=read(root/'main/state.json');t=state['tasks'][method+'_r00_c01'];directory=Path(t['outcome']['trial_summary_path']).parents[1];feedback=read(directory/'feedback.json')
  if spec['loader']=='mistral3_dequantize_fp8':
   from transformers import MistralCommonBackend
   tokenizer=MistralCommonBackend.from_pretrained(proof['path'],local_files_only=True)
  else:tokenizer=AutoTokenizer.from_pretrained(proof['path'],local_files_only=True,trust_remote_code=False)
  kwargs={'enable_thinking':False} if spec.get('disable_thinking') else {};batch=tokenizer.apply_chat_template(messages(feedback),add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',**kwargs);prompt=batch['input_ids'][0].tolist();data=prepare_tokenizer(tokenizer);results=[]
  configs=[('fixed_start',protocol['baseline_parameters']),('all_parameters_low',dict(learning_rate=.0001,pressure_weight=.5,physics_weight=.0001,bc_weight=.001,warmup_fraction=0)),('all_parameters_high',dict(learning_rate=.002,pressure_weight=2.,physics_weight=.01,bc_weight=.1,warmup_fraction=.0625))]
  for name,parameters in configs:
   obj=dict(parameters=parameters,reason='Schema audit',evidence_ids=[feedback['latest_required_id']],expected_effect='Technical acceptance check',risk='No model call');target=json.dumps(obj,separators=(',',':'));parsed,errors,_=parse_reply(target,feedback);assert not errors and parsed['parameters']==parameters;parser=make_parser(feedback);character_ok=True
   for character in target:
    if character not in parser.get_allowed_characters():character_ok=False;break
    parser=parser.add_character(character)
   character_ok=character_ok and parser.can_end();assert character_ok,(method,name,'Character grammar rejected a valid alternative')
   generated=tokenizer.encode(target,add_special_tokens=False);prefix=list(prompt);allowed=build_constraint(data,feedback);failure=None
   for j,token in enumerate(generated):
    choices=allowed(0,torch.tensor(prefix,dtype=torch.long))
    if token not in choices:failure=dict(step=j,token=int(token),decoded_token=tokenizer.decode([token],clean_up_tokenization_spaces=False),prefix=tokenizer.decode(generated[:j],clean_up_tokenization_spaces=False));break
    prefix.append(token)
   results.append(dict(probe=name,parameters=parameters,independent_parser_accepts=True,character_schema_accepts=True,canonical_token_path_accepts=failure is None,canonical_token_path_failure=failure))
  row=dict(method=method,model_repo=spec['repo'],tokenizer_model_path=proof['path'],tests=results,elapsed_seconds=time.monotonic()-started,no_model_weights_loaded=True,no_LLM_generation=True);rows.append(row);save(O/'schema_probe_progress.json',rows);print('SCHEMA_PROBE',method,[(v['probe'],v['canonical_token_path_accepts']) for v in results],flush=True)
 summary=dict(status='ALL_FIVE_NATIVE_TOKENIZER_SCHEMA_PROBES_COMPLETE',job=os.environ['SLURM_JOB_ID'],models=rows,scope='CPU tokenizer/grammar validation only. Deliberately valid alternative vectors test whether constraints force fixed defaults; no LLM sampling or retraining.');save(O/'schema_probes.json',summary)
if __name__=='__main__':
 try:run()
 except Exception:save(O/'schema_probe_failure.json',dict(traceback=traceback.format_exc()));raise
