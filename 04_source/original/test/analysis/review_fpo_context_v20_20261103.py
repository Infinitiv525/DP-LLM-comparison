"""Small read-only audit review; perturbations are explicitly nonphysical."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v20 import ROOT,WORK,save,digest,verify


def run():
    verify();base=WORK/'audit/142743';s=json.loads((base/'summary.json').read_text())
    if s['status']!='READ_ONLY_CONTEXT_AUDIT_COMPLETE' or s['optimizer_updates']!=0:raise ValueError('Incomplete audit')
    out=WORK/'review_20261103';out.mkdir(exist_ok=False);results=[];inputs={};row_count=0
    for result in s['results']:
        name=result['checkpoint']['name'];zfile=base/name/'latent_neighbours.json'
        z=json.loads(zfile.read_text());inputs[str(zfile)]=digest(zfile)
        assert len(z['records'])==75
        vectors=np.array([v['latent'] for v in z['records']],dtype=np.float64)
        records=z['records'];neighbours={}
        for scope in ['train','validation']:
            selected=[i for i,r in enumerate(records) if r['scope']==scope]
            standardized=[records[i]['nearest_other_geometry_train'][0]['distance'] for i in selected]
            raw=[]
            for i in selected:
                path=records[i]['path'].split('/');geometry=(path[-3],path[-2])
                other=[j for j,r in enumerate(records) if r['scope']=='train' and tuple(r['path'].split('/')[-3:-1])!=geometry]
                raw.append(float(np.sqrt(np.mean((vectors[other]-vectors[i])**2,axis=1)).min()))
            describe=lambda a:dict(min=float(np.min(a)),median=float(np.median(a)),max=float(np.max(a)))
            neighbours[scope]=dict(original_train_standardized=describe(standardized),raw_tanh_latent_rms=describe(raw))
        for p in sorted((base/name).glob('case_*.json')):
            rows=json.loads(p.read_text())['rows'];assert len(rows)==40;row_count+=len(rows);inputs[str(p)]=digest(p)
            assert sorted({r['offset'] for r in rows})==[1,5,10,20,40,80,160,238]
            assert sum(r['physical_input_consistency'] for r in rows)==8
        results.append(dict(checkpoint=result['checkpoint'],by_scope=result['by_scope'],
            original_by_scope_horizon=result['original_by_scope_horizon'],context_stats=result['context_stats'],
            latent_distance_review=neighbours,small_variance_latent_dimensions=int((np.asarray(z['train_only_std'])<.001).sum()),
            latent_distance_note='Standardization of nearconstant dimensions can amplify small differences; rawRMS is a posthoc sensitivity check, neither metric proves physicalOOD or missing information.'))
    assert row_count==1080
    accounting=json.loads((WORK/'scheduler/142743/summary.json').read_text())
    assert accounting['state']=='COMPLETED' and accounting['exit_code']=='0:0'
    save(out/'analysis.json',dict(results=results,rows=row_count,baseline_rows=216,latent_vectors=225,
        actual_resources=accounting,optimizer_updates=0,main_test_opened=False,independent_final_test=False,
        full_238frame_accuracy=False,counterfactuals_are_not_physical_cases=True))
    save(out/'provenance.json',dict(source_sha256=digest(__file__),input_sha256=inputs,
        frozen_protocol_sha256=digest(WORK/'freeze.json'),analysis_sha256=digest(out/'analysis.json')))
    files=[WORK/n for n in ['protocol.json','freeze.json','prepared.json','geometry_notes.json','submission.json']]
    files+=sorted(base.rglob('*.json'))+sorted((WORK/'scheduler/142743').glob('*.json'))+sorted(out.iterdir())
    files += [ROOT/'logs'/('dp-pinn20-context-audit_142743.'+e) for e in ['out','err']]+[Path(__file__).resolve()]
    assert all(p.is_file() and p.stat().st_size<2000000 and p.suffix in ['.json','.out','.err','.py'] for p in files)
    archive=WORK/'review_142743_small.zip'
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=digest(archive),rows=row_count)))
    for r in results:print('LATENT_REVIEW',r['checkpoint']['name'],json.dumps(r['latent_distance_review']))


if __name__=='__main__':run()
