"""Offline four-family local proposer with a PINN-only contract and public reason."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import traceback
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from ldc_main_common_v32 import WORK,read,save,digest,verify,phase_path
from ldc_main_contract_v32 import VERSION,system_digest,messages,canonical,parse_reply,repair_message
from smoke_local_configurator import ResourceSampler,FirstToken
from ldc_main_schema_v32 import prepare_tokenizer,build_constraint,CompleteJSON
from ldc_main_contract_v32 import reply_schema


def run(a):
    started=time.perf_counter();cpu=time.process_time();verify();a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(WORK/'protocol.json');spec=protocol['models'][a.index];generation=protocol['generation']
    packages={n:importlib.metadata.version(n) for n in protocol['llm_runtime']}
    if packages!=protocol['llm_runtime']:raise ValueError('LLM runtime versions changed')
    proof=read(Path(protocol['preparation'])/(spec['label']+'.json'));feedback=read(a.feedback)
    if feedback['phase']!=a.phase:raise ValueError('Feedback phase mismatch')
    if proof['spec']!=spec or proof['status']!='LOCAL_MODEL_VERIFIED':raise ValueError('Pinned model mismatch')
    if not read(phase_path(a.phase)/'models'/(spec['label']+'.json'))['verified']:raise ValueError('Unverified local model')
    if os.environ.get('HF_HUB_OFFLINE')!='1' or not torch.cuda.is_available():raise ValueError('Offline CUDA required')
    if torch.cuda.device_count()!=1:raise ValueError('One independently assigned GPU per proposer')
    sampler=ResourceSampler(.2);sampler.start()
    record=dict(status='RUNNING',contract_version=VERSION,system_prompt_sha256=system_digest(a.phase),
        feedback_sha256=digest(a.feedback),initial_messages_sha256=hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest(),
        model=spec,packages=packages,generation=dict(generation,seed=a.seed),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,attempts=[],test_evaluated=False,
        model_receipt_sha256=digest(Path(protocol['preparation'])/(spec['label']+'.json')),
        weight_origin='Original FP8 dequantized to BF16' if spec['loader']=='mistral3_dequantize_fp8' else 'Published BF16 checkpoint')
    try:
        torch.set_num_threads(4);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        t=time.perf_counter();path=Path(proof['path']);extra={}
        if spec['loader']=='mistral3_dequantize_fp8':
            from transformers import MistralCommonBackend,Mistral3ForConditionalGeneration,FineGrainedFP8Config
            tokenizer=MistralCommonBackend.from_pretrained(path,local_files_only=True)
            cls=Mistral3ForConditionalGeneration;extra={'quantization_config':FineGrainedFP8Config(dequantize=True)}
        else:
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False);cls=AutoModelForCausalLM
        model=cls.from_pretrained(path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',**extra).eval()
        torch.cuda.synchronize();record.update(load_seconds=time.perf_counter()-t,parameter_count=sum(p.numel() for p in model.parameters()),
            chat_template_sha256=hashlib.sha256(str(getattr(tokenizer,'chat_template',None)).encode()).hexdigest())
        t=time.perf_counter();tokenizer_data=prepare_tokenizer(tokenizer)
        record.update(grammar_setup_seconds=time.perf_counter()-t,syntax_constrained=True,
            schema=reply_schema(feedback),schema_sha256=hashlib.sha256(canonical(reply_schema(feedback)).encode()).hexdigest(),
            guidance_note='Shared required fields/types/categories/lengths; latest evidence label is routed, not proof of reasoning. Learning-rate bounds independently validated.')
        conversation=messages(feedback)
        for repair in range(generation['max_repairs']+1):
            kwargs={'enable_thinking':False} if spec.get('disable_thinking') else {}
            batch=tokenizer.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',**kwargs)
            if batch['input_ids'].shape[-1]>generation['max_input_tokens']:raise ValueError('Context cap, no truncation')
            batch={k:v.cuda() for k,v in batch.items() if torch.is_tensor(v)};streamer=FirstToken();t=time.perf_counter()
            prefix=build_constraint(tokenizer_data,feedback);stop=CompleteJSON(tokenizer,batch['input_ids'].shape[-1])
            with torch.inference_mode():
                output=model.generate(**batch,max_new_tokens=generation['max_new_tokens'],max_time=generation['max_time_seconds'],
                    do_sample=True,temperature=generation['temperature'],top_p=generation['top_p'],top_k=generation['top_k'],
                    repetition_penalty=generation['repetition_penalty'],use_cache=True,streamer=streamer,
                    prefix_allowed_tokens_fn=prefix,stopping_criteria=[stop])
            torch.cuda.synchronize();raw=tokenizer.decode(output[0,batch['input_ids'].shape[-1]:],skip_special_tokens=True)
            parsed,errors,transport=parse_reply(raw,feedback)
            attempt=dict(repair=repair,messages=json.loads(json.dumps(conversation)),raw_output=raw,parsed=parsed,errors=errors,transport=transport,
                generation_seconds=time.perf_counter()-t,first_token_seconds=None if streamer.first is None else streamer.first-t,
                input_tokens=int(batch['input_ids'].shape[-1]),output_tokens=int(output.shape[-1]-batch['input_ids'].shape[-1]))
            record['attempts'].append(attempt);save(a.output/f'attempt_{repair}.json',attempt)
            if not errors:record['configuration']=parsed;break
            conversation.extend([{'role':'assistant','content':raw},{'role':'user','content':repair_message(errors)}])
        record['status']='LDC_PROPOSAL_COMPLETE' if 'configuration' in record else 'LDC_PROPOSAL_INVALID'
    except Exception:
        record['status']='LDC_PROPOSAL_FAILED';record['error']=traceback.format_exc();print(record['error'],flush=True)
    finally:
        sampler.stop();rows=sampler.rows
        values=[r['gpu_process_bytes'] for r in rows if r['gpu_process_bytes'] is not None]
        record.update(elapsed_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_sampled_rss_bytes=max((r['rss_with_children_bytes'] for r in rows),default=None),
            peak_sampled_nvml_gpu_bytes=max(values) if values else None,nvml_warning=sampler.nvml_error,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
        save(a.output/'resources.json',rows);save(a.output/'summary.json',record)
    return record['status']=='LDC_PROPOSAL_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--phase',choices=['main','integration'],required=True)
    p.add_argument('--feedback',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)
