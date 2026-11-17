"""Read-only V23 result audit; keep selected and equal60k results separate."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v23 import ROOT,WORK,save,digest,verify


def metrics(cases):
    if len(cases) not in [3,6] or any(c['failure'] for c in cases):
        raise ValueError('Incomplete or failed assessment')
    return dict(mean_errors={k:float(np.mean([c['mean_errors'][k] for c in cases]))
                             for k in ['velocity_relative_l2','pressure_relative_l2']},
        mean_fd_momentum_loss=float(np.mean([c['mean_fd_momentum_loss'] for c in cases])),
        mean_reference_fd_momentum_loss=float(np.mean([c['mean_reference_fd_momentum_loss'] for c in cases])),
        accuracy_pass_count=sum(c['accuracy_pass'] for c in cases),mass_pass_count=sum(c['mass_pass'] for c in cases),
        latency_pass_count=sum(c['latency_pass'] for c in cases),
        query_p99_max_ms=max(c['direct_query_gpu_ms']['p99'] for c in cases))


def run():
    verify();base=WORK/'training/142897';s=json.loads((base/'summary.json').read_text())
    if s['status']!='KNOWN_HISTORY_COMPLETE' or not s['matched_initial_state'] or not s['matched_full_sampling']:
        raise ValueError('Incomplete/unmatched run')
    if s['main_test_opened'] or s['independent_final_test'] or s['viable_shared_variants']:
        raise ValueError('Unexpected interpretation')
    if s['decision']!='REVIEW_THEN_AUTHORIZED_CASE_SPECIFIC_FALLBACK':
        raise ValueError('Unexpected follow-up decision')
    out=WORK/'review_20261104';out.mkdir(exist_ok=False);results=[];inputs={}
    inputs[str(base/'summary.json')]=digest(base/'summary.json')
    for r in s['results']:
        name=r['config']['name'];p=base/name
        if digest(p/'best.pt')!=r['checkpoint_sha256']:raise ValueError('Selected weights changed')
        h=json.loads((p/'history.json').read_text());counts=r['added_training_case_updates']
        assert [a['update'] for a in h]==list(range(0,60001,10000))
        assert len(counts)==69 and sum(counts)==60000 and min(counts)==869 and max(counts)==870
        assert r['all220_training_frames_visited'] and r['anchor_frame']==21
        assert r['predicted_frames_per_case']==220
        assert r['physics_last'] is None and r['physics_selected'] is None
        assert r['train_scenarios']==69 and r['validation_scenarios']==6
        inputs[str(p/'history.json')]=digest(p/'history.json')
        for phase in ['selected_checkpoint','assessments/update_060000']:
            for i in range(6):
                f=p/phase/str(i)/'per_frame.json';rows=json.loads(f.read_text())
                assert len(rows)==220 and [a['frame'] for a in rows]==list(range(22,242))
                assert [a['offset'] for a in rows]==list(range(1,221))
                assert np.allclose([a['time'] for a in rows],np.arange(1,221)*.05,rtol=0,atol=1e-12)
                summary=json.loads((f.parent/'summary.json').read_text())
                for k,v in summary['mean_errors'].items():
                    assert np.isclose(v,np.mean([a['model'][k] for a in rows]),rtol=0,atol=1e-12)
                inputs[str(f)]=digest(f)
        results.append(dict(name=name,history_length=r['history_length'],selected_added_updates=r['selected_added_updates'],
            selected=metrics(r['accuracy']),end_60000=metrics(h[-1]['assessment']['cases']),
            selected_cases=r['accuracy'],selected_train_monitors=metrics(r['training_monitor']),
            end_train_monitors=metrics(h[-1]['train_fit_monitor']),
            selected_checkpoint_sha256=r['checkpoint_sha256'],last_checkpoint_sha256=digest(p/'last.pt'),
            source_checkpoint_sha256=r['source_checkpoint_sha256'],initial_state_sha256=r['initial_state_sha256'],
            full_sampling_sha256=r['sample_schedule_sha256'],case_frame_sha256=r['case_frame_schedule_sha256'],
            per_case_updates=dict(min=min(counts),max=max(counts),sum=sum(counts)),
            no_separate_pointwise_physics_audit=True))
    if [r['name'] for r in results]!=['last2_control','history20']:raise ValueError('Unexpected arms')
    for key in ['initial_state_sha256','full_sampling_sha256','case_frame_sha256','source_checkpoint_sha256']:
        if results[0][key]!=results[1][key]:raise ValueError('Mismatch '+key)
    manifest={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=digest(p))
              for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt']}
    save(out/'output_manifest.json',manifest)
    accounting=json.loads((WORK/'scheduler/142897/summary.json').read_text())
    assert accounting['state']=='COMPLETED' and accounting['exit_code']=='0:0'
    assert accounting['allocated_gpu_seconds']==19664 and accounting['allocated_core_seconds']==78656
    changes={}
    for phase in ['selected','end_60000']:
        a,b=[r[phase] for r in results]
        changes[phase]=dict(velocity_error_change_percentage_points=100*(b['mean_errors']['velocity_relative_l2']-a['mean_errors']['velocity_relative_l2']),
            pressure_error_change_percentage_points=100*(b['mean_errors']['pressure_relative_l2']-a['mean_errors']['pressure_relative_l2']),
            fd_momentum_reduction_percent=100*(a['mean_fd_momentum_loss']-b['mean_fd_momentum_loss'])/a['mean_fd_momentum_loss'])
    save(out/'analysis.json',dict(status='V23_COMPLETED_RESULTS_VERIFIED',results=results,changes=changes,
        actual_resources=accounting,verified_frame_range=[22,241],predicted_frames=220,
        pointwise_npz_files_on_perun=sum(k.endswith('.npz') for k in manifest),
        decision=s['decision'],fallback_authorized_by_user=s['fallback_authorized_by_user'],
        limitations=['Development validation reused for checkpoint selection; no independent final significance',
                     'Both arms anchor21 and predict220futureframes; not a causal comparison with old238frameheadlines',
                     'Three TRAIN fit monitors are not a full69case fit audit',
                     'Full220frame FDphysics and referencefloor measured; no separate continuous AD-FD pointwise audit',
                     'Direct absolute-time fullfield GPUquery, not CPUtransfer/rendering/causal-step latency'],
        independent_final_test=False,main_test_opened=False,llm_ready=False))
    save(out/'provenance.json',dict(source_sha256=digest(__file__),input_sha256=inputs,
        frozen_protocol_sha256=digest(WORK/'freeze.json'),analysis_sha256=digest(out/'analysis.json')))
    files=[WORK/n for n in ['protocol.json','freeze.json','prepared.json','submission.json']]
    files+=sorted((WORK/'benchmark/142897').rglob('*.json'))+sorted(base.rglob('*.json'))+sorted((WORK/'smoke/142897').rglob('*.json'))
    files+=sorted((WORK/'scheduler/142897').glob('*.json'))+sorted(out.iterdir())+[Path(__file__).resolve()]
    files += [ROOT/'logs'/('dp-pinn23-history_142897.'+e) for e in ['out','err']]
    assert len(files)==len(set(files))
    assert all(p.is_file() and p.stat().st_size<2000000 and p.suffix in ['.json','.py','.out','.err'] for p in files)
    archive=WORK/'review_142897_small.zip'
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=digest(archive),changes=changes,
        results=[{k:r[k] for k in ['name','selected_added_updates','selected','end_60000']} for r in results])))


if __name__=='__main__':run()
