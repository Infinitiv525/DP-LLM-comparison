"""Small read-only review and exportable development-learning figure."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v17 import ROOT,WORK,save,digest,verify


def means(cases):
    return {k:float(np.mean([c['mean_errors'][k] for c in cases]))
            for k in ['velocity_relative_l2','pressure_relative_l2']}


def run():
    verify();out=WORK/'review_20261103';out.mkdir(exist_ok=False)
    inputs={};results=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True)
    for job,name,color in [('141545','multicase_primitive','#2369a8'),('141546','multicase_curl','#ce6c19')]:
        base=WORK/'training'/job/name
        paths=[base/'summary.json',base/'history.json',WORK/'scheduler'/job/'summary.json']
        inputs.update({str(p):digest(p) for p in paths})
        s=json.loads(paths[0].read_text());history=json.loads(paths[1].read_text());account=json.loads(paths[2].read_text())
        assert account['state']=='COMPLETED' and account['exit_code']=='0:0'
        assert len(s['training_case_updates'])==69 and sum(s['training_case_updates'])==600000 and s['all238_training_frames_visited']
        times=[]
        for i in range(6):
            p=base/'selected_checkpoint'/str(i)/'per_frame.json';inputs[str(p)]=digest(p)
            rows=json.loads(p.read_text())
            assert len(rows)==238 and [r['frame'] for r in rows]==list(range(4,242))
            times.append(rows)
        timeline=[dict(offset=o,time=o*.05,mean_errors={k:float(np.mean([a[o-1]['model'][k] for a in times]))
                  for k in times[0][0]['model']}) for o in [1,5,10,20,40,80,160,238]]
        result=dict(job=job,name=name,selected_update=s['selected_update'],validation_mean_errors=means(s['accuracy']),
            selected_three_train_monitor_errors=means(s['training_monitor']),
            last_validation_mean_errors=means(history[-1]['assessment']['cases']),
            last_three_train_monitor_errors=means(history[-1]['train_fit_monitor']),
            validation_accuracy_pass_count=sum(c['accuracy_pass'] for c in s['accuracy']),
            validation_mass_pass_count=sum(c['mass_pass'] for c in s['accuracy']),
            query_p99_max_ms=max(c['direct_query_gpu_ms']['p99'] for c in s['accuracy']),
            validation_mean_fd_momentum_loss=float(np.mean([c['mean_fd_momentum_loss'] for c in s['accuracy']])),
            validation_mean_div_rms=float(np.mean([c['interior_div_rms'] for c in s['accuracy']])),
            case_frame_schedule_sha256=s['case_frame_schedule_sha256'],
            selected_checkpoint_sha256=s['checkpoint_sha256'],actual_resources=account,time_slices=timeline)
        results.append(result)
        label='Primitive u/v/p' if 'primitive' in name else 'Streamfunction psi/p'
        for ax,key in zip(axes,['velocity_relative_l2','pressure_relative_l2']):
            ax.plot([r['update']/1000 for r in history],[100*means(r['assessment']['cases'])[key] for r in history],
                    color=color,label=label+' | 6 VAL',linewidth=2)
            monitor=[r for r in history if 'train_fit_monitor' in r]
            ax.plot([r['update']/1000 for r in monitor],[100*means(r['train_fit_monitor'])[key] for r in monitor],
                    color=color,linestyle='--',marker='o',markersize=3,label=label+' | 3 TRAIN monitors')
    assert results[0]['case_frame_schedule_sha256']==results[1]['case_frame_schedule_sha256']
    for ax,title in zip(axes,['Velocity error','Pressure error']):
        ax.set_title(title);ax.set_xlabel('Optimizer updates (thousands)');ax.set_ylabel('Mean relative L2 error (%)')
        ax.axvline(100,color='#555555',linestyle=':',linewidth=1);ax.grid(alpha=.22);ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=7.5)
    fig.suptitle('69 TRAIN / 6 development VAL | one seed | both selected at 100k updates',fontsize=11)
    fig.text(.5,.015,'TRAIN curves cover 3 monitored cases, not all 69. No final TEST results.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.94]);fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    save(out/'analysis.json',dict(date='2026-11-03',results=results,case_frame_schedules_match=True,
        allocated_gpu_seconds=sum(r['actual_resources']['allocated_gpu_seconds'] for r in results),
        allocated_core_seconds=sum(r['actual_resources']['allocated_core_seconds'] for r in results),
        main_test_opened=False,independent_final_test=False))
    save(out/'provenance.json',dict(source=str(Path(__file__).resolve()),source_sha256=digest(__file__),input_sha256=inputs,
        frozen_protocol_sha256=digest(WORK/'freeze.json'),output_sha256={p.name:digest(p) for p in [out/'analysis.json',out/'learning_curves.png']}))
    files=[WORK/n for n in ['protocol.json','freeze.json','prepared.json','submission.json']]
    for job,name in [('141545','primitive'),('141546','curl')]:
        files+=sorted((WORK/'training'/job).rglob('*.json'))
        files+=sorted((WORK/'smoke'/job).rglob('*.json'))
        files+=sorted((WORK/'scheduler'/job).glob('*.json'))
        files += [ROOT/'logs'/f'dp-pinn17-{name}_{job}.{ext}' for ext in ['out','err']]
    files+=sorted(out.iterdir())+[Path(__file__).resolve()]
    assert all(p.is_file() and p.stat().st_size<2000000 and p.suffix in ['.json','.png','.out','.err','.py'] for p in files)
    archive=WORK/'review_141545_141546_small.zip'
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(results=results,archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=digest(archive)),indent=2))


if __name__=='__main__':run()
