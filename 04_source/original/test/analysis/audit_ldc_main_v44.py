"""Independent CPU audit of a live, immutable main experiment.

Only the audit output directory is written. Completed checkpoints are loaded
on CPU. Mutable RUNNING checkpoints are never opened. TEST targets are never
opened, and no optimizer, inference benchmark on GPU, or training is run.
"""
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
for folder in ['analysis', 'llm', 'training']:
    sys.path.insert(0, str(ROOT/'test'/folder))

import numpy as np
import torch
from ldc24_ops_v36 import WORK, POOL, verify, read, save, digest, queue_snapshot
from ldc_main_scheduler_v33 import locked, make_plan, choose_selection
from ldc_main_contract_v33 import training_config, parse_reply, messages, canonical, reply_schema, system_digest, BASELINE
from ldc_main_controls_v33 import propose_control
from ldc_shared_pinn_v26 import SharedSteadyPINN, steady_residual, regular_grid
from ldc_store_v26 import query

OUT = ROOT/'results/ldc_full_audit_20261106_v44/final'
checks, problems, manifests, rows = [], [], [], []


def check(name, condition, detail=None):
    ok = bool(condition)
    checks.append(dict(name=name, pass_check=ok))
    if not ok:
        problems.append(dict(name=name, detail=str(detail)[:2000]))
    return ok


def close(a, b, rtol=1e-8, atol=1e-12):
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def note_file(path, expected=None):
    value = digest(path)
    manifests.append(dict(path=str(path), bytes=path.stat().st_size, sha256=value))
    if expected is not None:
        check(str(path)+':sha256', value == expected)
    return value


def snapshot():
    with locked(WORK/'main'):
        raw = (WORK/'main/state.json').read_bytes()
    (OUT/'state_snapshot.json').write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def initialization(config, norm):
    torch.manual_seed(config['seed'])
    model = SharedSteadyPINN(config, norm)
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_data(protocol):
    prep = read(protocol['prepared_path'])
    split = read(protocol['split_path'])['indices']
    check('data:ready', prep['status']=='LDC_DATA_READY' and prep['physics_ready'])
    check('data:revision', prep['revision']=='937c672604d4c9f39aceedf1fa2f0a603695264b')
    check('data:steady_harmonics_512', prep['family']=='harmonics' and prep['no_time_dimension'] and prep['native_resolution']==[512,512])
    sets = {k:set(v) for k,v in split.items()}
    check('data:split_sizes', {k:len(v) for k,v in split.items()}=={'train':640,'validation':160,'test':200})
    check('data:unique_and_disjoint', all(len(sets[k])==len(split[k]) for k in split)
          and not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
          and set.union(*sets.values())==set(range(1000)))
    check('data:test_targets_not_prepared', prep['test_target_samples_read']==0 and not prep['main_test_opened'])
    cache = Path(prep['cache'])
    check('data:no_test_target_cache', not (cache/'Y_test.npy').exists())
    check('data:normalized_viscosity', prep['coordinates']=='Cell centres; xi=xphysical/2; physicalnu2/Re correspondsnormalizednu1/Re')
    check('data:reference_scale_audit', prep['median_relative_ns_residual']<.35 and .5<prep['median_inferred_viscosity_multiplier']<1.5)
    manifest = read(cache.parent/'cache_manifest.json')
    for path, expected in manifest.items():
        note_file(Path(path), expected['sha256'])
    norm = read(cache/'normalization.json')
    yt = np.load(cache/'Y_train.npy', mmap_mode='r', allow_pickle=False)
    yv = np.load(cache/'Y_validation.npy', mmap_mode='r', allow_pickle=False)
    fluid = np.load(cache/'fluid.npy', mmap_mode='r', allow_pickle=False)
    low = np.load(cache/'geometry_low.npy', mmap_mode='r', allow_pickle=False)
    check('data:target_shapes', yt.shape==(640,3,512,512) and yv.shape==(160,3,512,512))
    total = np.zeros(3); squares = np.zeros(3); count = 0
    max_pressure_mean = 0.
    for slot, index in enumerate(split['train']):
        v = np.asarray(yt[slot, :, fluid[index]], dtype=np.float64)
        # numpy mixed indexing can move the boolean-indexed axis to the front.
        if v.shape[0] != 3:
            v = v.T
        total += v.sum(1); squares += (v*v).sum(1); count += v.shape[1]
        max_pressure_mean = max(max_pressure_mean, abs(float(v[2].mean())))
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 0)).clip(1e-4)
    geometry = np.asarray(low[split['train']], dtype=np.float64)
    check('data:TRAIN_only_target_mean_recomputed', close(mean,norm['target_mean']))
    check('data:TRAIN_only_target_std_recomputed', close(std,norm['target_std']))
    check('data:TRAIN_fluid_point_count', count==norm['target_fluid_points'])
    check('data:TRAIN_geometry_normalization', close(geometry.mean((0,2,3)),norm['geometry_mean'])
          and close(geometry.std((0,2,3)).clip(1e-4),norm['geometry_std']))
    check('data:pressure_gauge', max_pressure_mean<1e-6)
    return prep, split, norm


def audit_completed(identifier, row, state, protocol, split, norm, expected_initial, paired):
    task, outcome = row['task'], row['outcome']
    label = identifier+':'
    if not check(label+'successful_training', outcome['status']=='COMPLETE', outcome['status']):
        return
    path = Path(outcome['trial_summary_path']); folder = path.parent; execution = folder.parent
    s = read(path); request = read(execution/'request.json'); config = read(folder/'config.json')
    env = read(folder/'environment.json'); history = read(folder/'history.json')
    selected = read(folder/'selected_validation/summary.json')
    diag = read(folder/'selected_diagnostics/summary.json')
    note_file(path, outcome['trial_summary_sha256'])
    check(label+'outcome_receipt', read(execution/'outcome.json')==outcome)
    check(label+'request_and_task', request['task']==task and read(execution/'task.json')==task
          and request['protocol_sha256']==digest(WORK/'protocol.json') and request['phase']=='main')
    expected = training_config(protocol['base_model'],request['parameters'],'main',task['seed'])
    check(label+'config', config==expected)
    check(label+'budget', s['phase']=='main' and s['epochs']==400 and s['seed']==task['seed']
          and config['width']==192 and config['depth']==4)
    check(label+'all_TRAIN_cases_400_times', len(s['case_counts'])==640
          and set(map(int,s['case_counts']))==set(split['train']) and set(s['case_counts'].values())=={400})
    check(label+'training_and_validation_coverage', s['train_cases']==640 and s['validation_cases']==160)
    check(label+'physics_updates', s['physics_updates']==64000-config['warmup'] and s['physics_updates']>0)
    check(label+'no_TEST', not s['main_test_opened'] and not selected['main_test_opened'] and not diag['main_test_opened'])
    check(label+'environment', env['torch']=='2.12.0+cu130' and env['numpy']=='2.5.4'
          and env['parameters']==134922 and env['precision']=='float32' and env['device']=='NVIDIA H200'
          and env['seed']==task['seed'] and not env['allow_tf32'] and env['cudnn_allow_tf32']
          and env['source_freeze']==digest(WORK/'freeze.json'))
    pair = (s['initial_weight_sha256'],s['sample_sha256'])
    check(label+'common_initialization_and_samples', paired.setdefault(task['seed'],pair)==pair)
    if task['seed'] not in expected_initial:
        expected_initial[task['seed']] = initialization(config,norm)
    check(label+'initialization_recreated_on_CPU', s['initial_weight_sha256']==expected_initial[task['seed']])
    check(label+'history_updates', [r['update'] for r in history]==list(range(0,64001,1600)))
    check(label+'history_finite', finite_tree(history))
    check(label+'every_validation_full_and_unweighted', all(
        not h['validation']['failure'] and [c['sample'] for c in h['validation']['cases']]==split['validation']
        and close(h['validation']['score'],np.mean([c['mse_all'] for c in h['validation']['cases']]))
        for h in history))
    check(label+'PDE_ramp_history', all(close(h['physics_coefficient'],config['physics_weight']*
          min(1.,max(0.,(h['update']-config['warmup'])/config['ramp']))) for h in history))
    best_history = min(history,key=lambda h:(h['validation']['score'],h['update']))
    check(label+'best_checkpoint_chosen_by_VAL', s['selected_update']==best_history['update'] and s['selected_update']>0)
    check(label+'selected_validation_receipt', selected==s['selected_validation'] and not selected['failure']
          and [c['sample'] for c in selected['cases']]==split['validation'])
    check(label+'selected_metric_means', all(close(selected['means'][key],np.mean([c[key] for c in selected['cases']]))
          for key in ['velocity_relative_l2','pressure_relative_l2','mse_all','relative_l2_all']))
    check(label+'selected_score', close(selected['score'],selected['means']['mse_all'])
          and close(selected['score'],best_history['validation']['score'],rtol=1e-6))
    check(label+'physics_diagnostics', close(selected['mean_momentum_rms'],np.mean([c['physics']['momentum_rms'] for c in selected['cases']]))
          and close(selected['mean_divergence_rms'],np.mean([c['physics']['divergence_rms'] for c in selected['cases']])))
    check(label+'all_boundary_cases', [c['sample'] for c in diag['cases']]==split['validation'])
    sides=['bottom','top','left','right','obstacle']
    side_means={side:float(np.mean([c['boundary'][side]['velocity_error_rms'] for c in diag['cases']])) for side in sides}
    check(label+'boundary_means_recomputed', all(close(side_means[k],diag['boundary_by_side'][k]) for k in sides)
          and close(np.mean(list(side_means.values())),diag['boundary_mean']) and close(diag['boundary_mean'],s['boundary_mean']))
    raw=np.array([c['timing']['raw_ms'] for c in diag['cases']])
    check(label+'raw_timings', raw.shape==(160,200) and np.isfinite(raw).all() and (raw>0).all())
    quantiles=np.percentile(raw,[50,95,99],axis=1)
    check(label+'timing_quantiles_recomputed', all(close(quantiles[j],[c['timing'][f'p{p}'] for c in diag['cases']]) for j,p in enumerate([50,95,99])))
    speed=s['selected_speed']
    check(label+'speed_scope', speed==diag['speed'] and speed['fullfield']==[512,512]
          and speed['warmup']==30 and speed['repeats']==200 and speed['inputs_gpu'])
    check(label+'speed_rows_recomputed', speed['cases']==[dict(sample=c['sample'],**{k:v for k,v in c['timing'].items() if k!='raw_ms'}) for c in diag['cases']])
    check(label+'p99_under_5ms', speed['pass_5ms'] and bool((quantiles[2]<=5).all()))
    note_file(folder/'best.pt',s['checkpoint_sha256'])
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    resume=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True)
    note_file(folder/'last.pt'); note_file(folder/'training_state.pt')
    model=SharedSteadyPINN(config,norm); model.load_state_dict(best['state'],strict=True)
    check(label+'checkpoint_tensor_shapes_and_finiteness', sum(p.numel() for p in model.parameters())==134922
          and finite_tree(best) and finite_tree(last) and finite_tree(resume))
    check(label+'checkpoint_metadata', best['config']==config and best['normalization']==norm
          and best['source_freeze']==digest(WORK/'freeze.json') and best['update']==s['selected_update']
          and close(best['score'],selected['score'],rtol=1e-6))
    check(label+'last_checkpoint_full_budget', last['update']==64000 and last['config']==config
          and close(last['score'],history[-1]['validation']['score']))
    check(label+'resume_state_complete', set(resume['case_counts'].values())=={400}
          and resume['sample_sha256']==s['sample_sha256'] and resume['scheduler']['last_epoch']==64000
          and all(k in resume for k in ['optimizer','numpy_rng','data_rng','pde_rng','bc_rng','torch_rng','cuda_rng']))
    check(label+'resume_matches_last', resume['checkpoint']['update']==64000 and all(
          torch.equal(v,resume['checkpoint']['state'][k]) for k,v in last['state'].items()))
    process=read(execution/'training_process.json')
    check(label+'training_process', process==outcome['processes']['training'] and process['exit_code']==0
          and not process['timed_out'] and 0<s['training_seconds']<=process['wall_seconds']<21600)
    h=outcome['history']
    check(label+'actual_feedback_values', h['parameters']==request['parameters'] and h['training_seed']==task['seed']
          and h['epochs']==400 and h['score']==selected['score'] and h['velocity_error']==selected['means']['velocity_relative_l2']
          and h['pressure_error']==selected['means']['pressure_relative_l2'] and h['boundary_error']==s['boundary_mean']
          and h['momentum_rms']==selected['mean_momentum_rms'] and h['divergence_rms']==selected['mean_divergence_rms'])
    proposal_seconds=0.
    if task['kind']=='search' and task['candidate']>0:
        feedback=read(execution/'feedback.json')
        previous=[state['tasks'][f"{task['method']}_r{task['run']:02d}_c{i:02d}"]['outcome']['history'] for i in range(task['candidate'])]
        check(label+'own_history_only', feedback['history']==previous and feedback['remaining_attempts']==8-task['candidate'])
        if task['method'] in ['random','tpe']:
            control=read(execution/'control_proposal.json')
            expected_parameters,details=propose_control(task['method'],130001+task['run'],previous)
            check(label+'control_replay', expected_parameters==request['parameters']==control['parameters']
                  and details==control['details'] and control['feedback_sha256']==digest(execution/'feedback.json'))
            proposal_seconds=control['decision_seconds']
        else:
            proposal=read(execution/'proposal/summary.json')
            note_file(execution/'proposal/summary.json',outcome['proposal_summary_sha256'])
            spec=next(m for m in protocol['models'] if m['label']==task['method'])
            check(label+'LLM_pin_and_runtime', proposal['model']==spec and proposal['packages']==protocol['llm_runtime']
                  and proposal['model_receipt_sha256']==digest(Path(protocol['preparation'])/(spec['label']+'.json')))
            check(label+'LLM_prompt_and_seed', proposal['system_prompt_sha256']==system_digest('main')
                  and proposal['initial_messages_sha256']==hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest()
                  and proposal['feedback_sha256']==digest(execution/'feedback.json')
                  and proposal['generation']==dict(protocol['generation'],seed=120001+100*task['run']+task['candidate']))
            parsed,errors,_=parse_reply(proposal['attempts'][-1]['raw_output'],feedback)
            check(label+'LLM_raw_JSON_used_without_manual_repair', not errors and parsed==proposal['configuration']
                  and parsed['parameters']==request['parameters'])
            check(label+'LLM_common_schema', proposal['schema']==reply_schema(feedback)
                  and proposal['syntax_constrained'] and not proposal['test_evaluated'])
            attempts=proposal['attempts']; conversation=messages(feedback)
            valid_attempts=1<=len(attempts)<=2
            from ldc_main_contract_v33 import repair_message
            for n,a in enumerate(attempts):
                parsed_a,errors_a,_=parse_reply(a['raw_output'],feedback)
                valid_attempts &= a['repair']==n and a['messages']==conversation and a['parsed']==parsed_a and a['errors']==errors_a
                valid_attempts &= 0<a['input_tokens']<=8192 and 0<=a['output_tokens']<=512 and a['generation_seconds']>=0
                conversation=conversation+[{'role':'assistant','content':a['raw_output']},{'role':'user','content':repair_message(errors_a)}]
            check(label+'LLM_attempts_and_repair_budget',valid_attempts)
            resources=read(execution/'proposal/resources.json')
            check(label+'LLM_resources_recorded', len(resources)>0 and all(k in proposal for k in [
                'load_seconds','grammar_setup_seconds','elapsed_seconds','cpu_seconds','peak_rss_kib',
                'peak_sampled_rss_bytes','peak_sampled_nvml_gpu_bytes','peak_torch_allocated_bytes'])
                and finite_tree(proposal) and proposal['peak_torch_allocated_bytes']>0)
            pp=read(execution/'proposal_process.json')
            check(label+'LLM_process',pp==outcome['processes']['proposal'] and pp['exit_code']==0 and not pp['timed_out'])
            proposal_seconds=pp['wall_seconds']
    elif task['kind']=='retrain':
        selection=choose_selection(state,task)
        check(label+'selection_without_best_seed_picking',selection==read(execution/'selection.json')
              and selection==read(WORK/'main/selections'/f"{task['method']}_r{task['run']:02d}.json")
              and selection['parameters']==request['parameters'])
    else:
        check(label+'fixed_start_or_baseline',request['parameters']==BASELINE)
    rows.append(dict(id=identifier,method=task['method'],kind=task['kind'],seed=task['seed'],
        selected_update=s['selected_update'],val_mse=selected['score'],velocity_l2=selected['means']['velocity_relative_l2'],
        pressure_l2=selected['means']['pressure_relative_l2'],momentum_rms=selected['mean_momentum_rms'],
        boundary_rms=s['boundary_mean'],worst_case_p99_ms=float(quantiles[2].max()),
        training_seconds=s['training_seconds'],proposal_wall_seconds=proposal_seconds,
        checkpoint_sha256=s['checkpoint_sha256'],path=str(folder)))


def cpu_replay(first, prep, split, norm):
    folder=Path(first['outcome']['trial_summary_path']).parent
    c=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    model=SharedSteadyPINN(c['config'],norm); model.load_state_dict(c['state']); model.eval()
    cache=Path(prep['cache']); inputs=read(cache/'inputs.json'); index=split['validation'][0]
    fluid=np.array(np.load(cache/'fluid.npy',mmap_mode='r')[index],copy=True)
    original_x=np.load(cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False)
    sdf=np.asarray(original_x[index,inputs['schema']['sdf_channel']],dtype=np.float32)*inputs['schema']['sdf_orientation']
    known=dict(geometry_native=torch.tensor(np.stack((fluid.astype('float32'),sdf))[None]),
               fluid=torch.tensor(fluid),reynolds=inputs['reynolds'][index])
    with torch.inference_mode():
        prediction=query(model,known,regular_grid(device='cpu')).numpy()
    target=np.moveaxis(np.asarray(np.load(cache/'Y_validation.npy',mmap_mode='r')[0]),0,2)
    d=prediction.astype(np.float64)-target
    metric=dict(mse_all=float(np.mean(d*d)),
        velocity_relative_l2=float(np.sqrt((d[fluid,:2]**2).sum()/(target[fluid,:2].astype(np.float64)**2).sum())),
        pressure_relative_l2=float(np.sqrt((d[fluid,2]**2).sum()/(target[fluid,2].astype(np.float64)**2).sum())))
    original=read(folder/'selected_validation/summary.json')['cases'][0]
    check('CPU_replay:finite_full_native_prediction',prediction.shape==(512,512,3) and np.isfinite(prediction).all())
    check('CPU_replay:metrics_match_saved_GPU_result',all(close(v,original[k],rtol=1e-3,atol=1e-9) for k,v in metric.items()),metric)
    check('CPU_replay:solid_mask_and_pressure_gauge',np.all(prediction[~fluid]==0) and abs(float(prediction[fluid,2].mean()))<1e-6)
    q=torch.tensor([[.2,.3],[.4,.6],[.7,.8]],requires_grad=True)
    re=73.
    poiseuille=torch.stack((q[:,1]*(1-q[:,1]),q[:,0]*0,-2/re*q[:,0]),1)
    residual=steady_residual(q,poiseuille,re).detach().numpy()
    check('physics:manufactured_Poiseuille_NS',float(np.abs(residual).max())<1e-7)
    q=torch.tensor([[.2,.3],[.4,.6],[.7,.8]],requires_grad=True)
    divergent=torch.stack((q[:,0],q[:,1],q[:,0]*0),1)
    r=steady_residual(q,divergent,re)
    check('physics:nonzero_residual_detected',close(r[:,:2].detach(),q.detach(),rtol=1e-6) and close(r[:,2].detach(),[2,2,2]))
    with torch.no_grad():
        geometry=torch.nn.functional.avg_pool2d(known['geometry_native'],8)
        context=model.context(geometry,known['reynolds'])
        grid=regular_grid(64,64,device='cpu')
        full=model.grid_prediction(grid,context,64,64)
        y=torch.tensor([0,0,1,16,31,62,63,63]); x=torch.tensor([0,63,21,40,25,44,0,63])
        points=torch.stack(((x+.5)/64,(y+.5)/64),1)
        sampled=model.sampled_prediction(points,context,y,x,64,64)
    check('physics:sampled_FD_matches_grid_including_edges',close(sampled,full[y,x],rtol=1e-3,atol=1e-3))
    q=torch.tensor([[.15,.22],[.64,.71]],requires_grad=True)
    value=model(q,context)
    du=torch.autograd.grad(value[:,0].sum(),q,create_graph=True,retain_graph=True)[0]
    dv=torch.autograd.grad(value[:,1].sum(),q,create_graph=True,retain_graph=True)[0]
    divergence=du[:,0]+dv[:,1]
    check('physics:streamfunction_AD_divergence',float(divergence.detach().abs().max())<1e-4)
    save(OUT/'cpu_replay.json',dict(sample=index,checkpoint_sha256=digest(folder/'best.pt'),metrics=metric,
         saved_GPU_metrics={k:original[k] for k in metric},poiseuille_max_residual=float(np.abs(residual).max()),
         streamfunction_divergence_max=float(divergence.detach().abs().max()),
         native_resolution=[512,512],CPU_only=True,new_training=False,TEST_opened=False))


def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    check('audit:CPU_only',not torch.cuda.is_available())
    verify()
    source_sha=digest(__file__)
    save(OUT/'started.json',dict(source_sha256=source_sha,at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()))
    protocol=read(WORK/'protocol.json'); state,state_sha=snapshot(); tasks=state['tasks']
    check('plan:exact_395',len(tasks)==395 and [r['task'] for r in tasks.values()]==make_plan('main')==read(WORK/'main/plan.json'))
    check('plan:kind_counts',Counter(r['task']['kind'] for r in tasks.values())=={'search':240,'retrain':150,'fixed':5})
    check('plan:method_counts',Counter(r['task']['method'] for r in tasks.values())==dict.fromkeys(protocol['methods'],65)|{'fixed':5})
    check('plan:seeds',protocol['seeds']['search_training']==list(range(140001,140006)) and protocol['seeds']['final_training']==list(range(150001,150006)))
    check('plan:no_attention',not state['attention_required'] and not (POOL/'hold.json').exists())
    check('plan:dependencies_done_before_claim',all(all(tasks[d]['status']=='DONE' and tasks[d]['finished_unix']<=r['claimed_unix']
          for d in r['task']['dependencies']) for r in tasks.values() if r['status'] in ['RUNNING','DONE']))
    queue=queue_snapshot(); registrations={str(read(p)['job']):read(p) for p in (WORK/'submissions').glob('*.json')}
    check('operations:all_running_owners_active',all(r['job'] in queue and registrations[r['job']]['phase']=='main'
          for r in tasks.values() if r['status']=='RUNNING'))
    check('operations:reserved_nodes',sum(queue.values())<=4)
    active=[j for j in queue if registrations.get(j,{}).get('phase')=='main']
    devices=[]
    for job in active:
        assigned=read(WORK/'main/batches'/job/'device_assignment.json')['rows']
        check('operations:'+job+':distinct_devices',len(assigned)==registrations[job]['gpus']
              and all(r['offline'] and 'H200' in r['device'] for r in assigned))
        devices.extend(r['uuid'] for r in assigned)
        bootstrap_path=POOL/'bootstrap'/(job+'.json')
        if bootstrap_path.exists():
            check('operations:'+job+':bootstrap_gate',read(bootstrap_path)['status']=='ACCOUNTING_READY_BEFORE_TRAINING')
        else:
            # V35 launched 143656 before the V36 bootstrap existed. Validate
            # its original pre-release registration and recorded migration,
            # rather than demand a later-generation bootstrap retroactively.
            reg=registrations[job]
            migration=read(POOL/'migration_completed.json')
            launch=read(POOL/'launch_verification_24.json')
            check('operations:'+job+':legacy_registration_and_migration',
                reg['registered_before_release'] and reg['freeze_sha256']==digest(WORK/'freeze.json')
                and reg['operational_manifest_sha256']==digest(WORK/'operational_manifest_v35.json')
                and str(reg['accounting_job']) in migration['old_cpu_jobs']
                and job in launch['jobs'] and launch['distinct_gpu_count']==24)
    check('operations:all_active_devices_unique',len(devices)==len(set(devices))<=24)
    group=read(POOL/'groups/group_000.json'); collector=group['active_collector']
    status=read(POOL/'collector_status'/(collector+'.json'))
    check('operations:CPU_collector_active',collector in queue and registrations[collector]['phase']=='operational_collector_v36')
    check('operations:collector_tracks_full_group',set(group['jobs'])==set(status['accounted'])|set(status['waiting']))
    check('operations:collector_recent',time.time()-status['unix']<120)
    check('operations:capacity_GPU_released',not any(j in queue for j in ['143805','143807']))
    check('operations:capacity_closed',read(ROOT/'results/ldc_capacity_20261105_v39/closed_for_new_training_20261106_v43.json')['new_capacity_runs']==0)
    for job in status['accounted']:
        account=read(WORK/'scheduler'/job/'summary.json')
        batch_review=read(WORK/'main/batches'/job/'review.json')
        check('accounting:'+job,account['state']=='COMPLETED' and account['exit_code']=='0:0' and all(batch_review['checks'].values()))
    overview=read(POOL/'current_group.json')
    overview_stale=overview.get('collector')!=collector or set(overview.get('jobs',[]))!=set(group['jobs'])
    save(OUT/'operational_overview_finding.json',dict(stale=overview_stale,old=overview,authoritative_group=group,
         running_collector=status,scientific_results_affected=False))
    print('GLOBAL_PLAN_AND_OPERATIONS_CHECKED',flush=True)
    prep,split,norm=audit_data(protocol)
    print('DATA_HASHES_AND_TRAIN_NORMALIZATION_CHECKED',flush=True)
    for spec in protocol['models']:
        receipt=read(WORK/'main/models'/(spec['label']+'.json'))
        check('LLM:'+spec['label']+':verified_pinned_weights',receipt['verified'] and receipt['spec']==spec
              and receipt['manifest_sha256']==digest(Path(protocol['preparation'])/(spec['label']+'.json')))
    expected_initial={}; paired={}; completed=[]
    for identifier,row in tasks.items():
        if row['status']!='DONE':continue
        completed.append(row)
        try:
            audit_completed(identifier,row,state,protocol,split,norm,expected_initial,paired)
        except Exception:
            check(identifier+':AUDITOR_EXCEPTION',False,traceback.format_exc())
        if len(completed)%20==0: print('COMPLETED_MODELS_AUDITED',len(completed),'PROBLEMS',len(problems),flush=True)
    print('ALL_COMPLETED_MODELS_CHECKED',len(completed),flush=True)
    try: cpu_replay(completed[0],prep,split,norm)
    except Exception: check('CPU_replay:AUDITOR_EXCEPTION',False,traceback.format_exc())
    running=[]
    for identifier,row in tasks.items():
        if row['status']!='RUNNING':continue
        e=WORK/'main/executions'/identifier/row['job']
        request=read(e/'request.json') if (e/'request.json').exists() else None
        conf=read(e/'training/config.json') if (e/'training/config.json').exists() else None
        if request is not None:
            check(identifier+':running_request',request['task']==row['task'] and request['protocol_sha256']==digest(WORK/'protocol.json'))
        if conf is not None:
            check(identifier+':running_config',conf==training_config(protocol['base_model'],request['parameters'],'main',row['task']['seed']))
        running.append(dict(id=identifier,job=row['job'],worker=row['worker'],claimed_unix=row['claimed_unix'],
             request_present=request is not None,config_present=conf is not None))
    pending=[]
    for identifier,row in tasks.items():
        if row['status']!='PENDING':continue
        deps=row['task']['dependencies']
        pending.append(dict(id=identifier,kind=row['task']['kind'],method=row['task']['method'],seed=row['task']['seed'],
            dependencies=';'.join(deps),ready=all(tasks[d]['status']=='DONE' for d in deps)))
    # A fresh end snapshot distinguishes progress during the audit from duplicate
    # or orphan executions. Only pre-existing DONE rows must stay byte-identical.
    end=read(WORK/'main/state.json')
    check('live:completed_outcomes_unchanged',all(end['tasks'][k]==v for k,v in tasks.items() if v['status']=='DONE'))
    for identifier,row in end['tasks'].items():
        dirs=list((WORK/'main/executions'/identifier).glob('*'))
        dirs=[p for p in dirs if p.is_dir()]
        check(identifier+':no_hidden_retry',len(dirs)<=1,dirs)
        if row['status']=='PENDING':check(identifier+':pending_not_executed',not dirs)
    verify()
    check('audit:source_unchanged',digest(__file__)==source_sha)
    save(OUT/'checks.json',checks); save(OUT/'problems.json',problems)
    save(OUT/'artifact_hashes.json',manifests); save(OUT/'running_snapshot.json',running)
    for name,table in [('completed_trials.csv',rows),('pending_tasks.csv',pending)]:
        if table:
            with (OUT/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    future=[
        dict(stage='Finish existing training',scope='395 immutable attempts:240search+150selected-retrain+5fixed',status='IN_PROGRESS'),
        dict(stage='Freeze final checkpoint manifest',scope='All30selected configurations x5seeds +5fixed =155final models, include failures',status='PENDING_TRAINING'),
        dict(stage='Freeze final TEST evaluator and analysis',scope='Separate future stage; exact split/gauge/scales/metric/seed nesting/timing/resources before opening targets',status='REQUIRED_BEFORE_TEST'),
        dict(stage='Evaluate sealed TEST',scope='200cases per frozen selected/fixed seed; never tune or pick best final seed using TEST',status='SEALED_NOT_RUN'),
        dict(stage='Archive final artifacts',scope='Weights, numeric results, proposal reasons/resources, closure/hashes/accounting directly Perun->GitHub; no raw arrays on PC',status='FINAL_SNAPSHOT_PENDING'),
        dict(stage='No further larger-network experiments',scope='Existing V39 results preserved; zero new capacity jobs',status='CLOSED')]
    save(OUT/'remaining_stage_checklist.json',future)
    result=dict(status='PASS' if not problems else 'REVIEW_REQUIRED',
        observed_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),elapsed_seconds=time.monotonic()-started,
        checks_total=len(checks),checks_passed=sum(r['pass_check'] for r in checks),problem_count=len(problems),
        completed_audited=len(completed),snapshot_counts=dict(Counter(r['status'] for r in tasks.values())),
        latest_counts=dict(Counter(r['status'] for r in end['tasks'].values())),source_sha256=source_sha,state_sha256=state_sha,
        full_checkpoint_and_resume_checks=True,full_raw_timing_quantiles_recomputed=True,
        TRAIN_normalization_recomputed=True,cache_hashes_rechecked=True,CPU_only=True,
        TEST_targets_opened=False,new_training=False,existing_files_modified=False,
        operational_overview_needs_update=overview_stale,
        limitations=['Steady fields only, not temporal rollout','Flowcase split shares geometries across splits; no unseen-geometry claim',
                     'NS and wall residuals are measured and nonzero; PINN is not an exact PDE solution',
                     '5ms is GPU full-field query timing, not end-to-end rendering or physical timestep',
                     'LLM weight full-hash receipts and current frozen fingerprints checked; 61GB weights not rehashed again',
                     'GitHub model release is a point-in-time snapshot; final models require a new final export'])
    save(OUT/'summary.json',result)
    print('AUDIT_RESULT',json.dumps(result),flush=True)
    return not problems


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
