"""Preserve matched end-budget and selected-checkpoint comparisons separately."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v19 import ROOT,WORK,save,digest,verify


def metrics(cases):
    return dict(mean_errors={k:float(np.mean([c['mean_errors'][k] for c in cases]))
                             for k in ['velocity_relative_l2','pressure_relative_l2']},
        mean_fd_momentum_loss=float(np.mean([c['mean_fd_momentum_loss'] for c in cases])),
        accuracy_pass_count=sum(c['accuracy_pass'] for c in cases),mass_pass_count=sum(c['mass_pass'] for c in cases),
        query_p99_max_ms=max(c['direct_query_gpu_ms']['p99'] for c in cases))


def run():
    verify();base=WORK/'training/142615';s=json.loads((base/'summary.json').read_text())
    if s['status']!='ADDED_GRID_PHYSICS_COMPLETE' or not s['matched_initial_state'] or not s['matched_full_sampling']:raise ValueError('Incomplete/unmatched run')
    out=WORK/'review_20261103';out.mkdir(exist_ok=False);results=[];inputs={}
    inputs[str(base/'summary.json')]=digest(base/'summary.json')
    for r in s['results']:
        name=r['config']['name'];p=base/name
        if digest(p/'best.pt')!=r['checkpoint_sha256']:raise ValueError('Selected weights changed')
        h=json.loads((p/'history.json').read_text())
        assert h[-1]['update']==30000 and len(r['added_training_case_updates'])==69
        assert sum(r['added_training_case_updates'])==30000 and r['all238_training_frames_visited']
        inputs[str(p/'history.json')]=digest(p/'history.json')
        for phase in ['selected_checkpoint','assessments/update_030000']:
            for i in range(6):
                f=p/phase/str(i)/'per_frame.json';rows=json.loads(f.read_text())
                assert len(rows)==238 and [a['frame'] for a in rows]==list(range(4,242))
                inputs[str(f)]=digest(f)
        results.append(dict(name=name,selected_added_updates=r['selected_added_updates'],
            selected=metrics(r['accuracy']),end_30000=metrics(h[-1]['assessment']['cases']),
            selected_train_monitors=metrics(r['training_monitor']),
            end_train_monitors=metrics(h[-1]['train_fit_monitor']),
            selected_physics=r['physics_selected'],end_physics=r['physics_last'],
            selected_checkpoint_sha256=r['checkpoint_sha256'],last_checkpoint_sha256=digest(p/'last.pt'),
            source_checkpoint_sha256=r['source_checkpoint_sha256'],initial_state_sha256=r['initial_state_sha256'],
            full_sampling_sha256=r['sample_schedule_sha256'],case_frame_sha256=r['case_frame_schedule_sha256']))
    manifest={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=digest(p))
              for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt']}
    save(out/'output_manifest.json',manifest)
    control,added=results
    reduction=lambda a,b:100*(a-b)/a
    accounting=json.loads((WORK/'scheduler/142615/summary.json').read_text())
    assert accounting['state']=='COMPLETED' and accounting['exit_code']=='0:0'
    save(out/'analysis.json',dict(results=results,
        selected_fd_momentum_reduction_percent=reduction(control['selected']['mean_fd_momentum_loss'],added['selected']['mean_fd_momentum_loss']),
        equal_budget_30000_fd_momentum_reduction_percent=reduction(control['end_30000']['mean_fd_momentum_loss'],added['end_30000']['mean_fd_momentum_loss']),
        actual_resources=accounting,pointwise_npz_files_on_perun=sum(k.endswith('.npz') for k in manifest),
        pointwise_npz_bytes_on_perun=sum(v['bytes'] for k,v in manifest.items() if k.endswith('.npz')),
        independent_final_test=False,main_test_opened=False,llm_ready=False))
    save(out/'provenance.json',dict(source_sha256=digest(__file__),input_sha256=inputs,
        frozen_protocol_sha256=digest(WORK/'freeze.json'),analysis_sha256=digest(out/'analysis.json')))
    files=[WORK/n for n in ['protocol.json','freeze.json','prepared.json','sampling.json','diagnostic_protocol.json','submission.json']]
    files+=sorted(base.rglob('*.json'))+sorted((WORK/'smoke/142615').rglob('*.json'))
    files+=sorted((WORK/'scheduler/142615').glob('*.json'))+sorted(out.iterdir())+[Path(__file__).resolve()]
    files += [ROOT/'logs'/('dp-pinn19-grid-physics_142615.'+e) for e in ['out','err']]
    assert all(p.is_file() and p.stat().st_size<2000000 and p.suffix in ['.json','.py','.out','.err'] for p in files)
    archive=WORK/'review_142615_small.zip'
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=digest(archive),
        equal_budget_fd_reduction_percent=reduction(control['end_30000']['mean_fd_momentum_loss'],added['end_30000']['mean_fd_momentum_loss']),
        selected_fd_reduction_percent=reduction(control['selected']['mean_fd_momentum_loss'],added['selected']['mean_fd_momentum_loss']),
        pointwise_npz_bytes_on_perun=sum(v['bytes'] for k,v in manifest.items() if k.endswith('.npz')))))


if __name__=='__main__':run()
