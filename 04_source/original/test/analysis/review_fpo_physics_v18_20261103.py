"""Small numerical review; pointwise arrays stay on Perun."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from fpo_common_v18 import ROOT,WORK,save,digest,verify


def run():
    verify();base=WORK/'audit/142584';out=WORK/'review_20261103';out.mkdir(exist_ok=False)
    source=base/'summary.json';summary=json.loads(source.read_text());results=[]
    assert summary['status']=='READ_ONLY_PHYSICS_AUDIT_COMPLETE' and summary['optimizer_updates']==0
    inputs={str(source):digest(source)}
    for result in summary['results']:
        name=result['checkpoint']['name'];rows=[]
        for p in sorted((base/name).glob('*/summary.json')):
            s=json.loads(p.read_text());assert [r['offset'] for r in s['rows']]==[1,5,10,20,40,80,160,238]
            rows+=s['rows'];inputs[str(p)]=digest(p)
        assert len(rows)==72
        by_scope={}
        for scope in ['train','validation']:
            selected=[r for r in rows if r['scope']==scope]
            by_scope[scope]=dict(aggregate=result['by_scope'][scope],terms={
                term:dict(ad_rms=float(np.mean([r['continuous_centre'][term]['rms'] for r in selected])),
                          fd_rms=float(np.mean([r['fd_0.0025']['terms'][term]['rms'] for r in selected])),
                          difference_rms=float(np.mean([r['fd_0.0025']['versus_continuous'][term]['difference_rms'] for r in selected])))
                for term in ['velocity','temporal','advection','pressure_gradient','diffusion','momentum']})
        results.append(dict(checkpoint=result['checkpoint'],by_scope=by_scope,by_scope_horizon=result['by_scope_horizon']))
    manifest=json.loads((base/'output_manifest.json').read_text())
    for rel,info in manifest.items():
        p=base/rel
        if p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:raise ValueError('Changed output: '+rel)
    save(out/'analysis.json',dict(date='2026-11-03',results=results,optimizer_updates=0,
        case_time_rows=288,matched_centre_samples=147456,main_test_opened=False,
        pointwise_files_on_perun=sum(p.endswith('.npz') for p in manifest),
        pointwise_bytes_on_perun=sum(i['bytes'] for p,i in manifest.items() if p.endswith('.npz')),
        audit_not_full_238frame_accuracy=True))
    save(out/'provenance.json',dict(source_sha256=digest(__file__),input_sha256=inputs,
        frozen_protocol_sha256=digest(WORK/'freeze.json'),analysis_sha256=digest(out/'analysis.json')))
    files=[WORK/n for n in ['protocol.json','freeze.json','prepared.json','sampling.json','submission.json']]
    files+=sorted(base.rglob('*.json'))+sorted((WORK/'scheduler/142584').glob('*.json'))
    files += [ROOT/'logs'/('dp-pinn18-physics-audit_142584.'+e) for e in ['out','err']]
    files += sorted(out.iterdir())+[Path(__file__).resolve()]
    assert all(p.is_file() and p.stat().st_size<2000000 and p.suffix in ['.json','.py','.out','.err'] for p in files)
    archive=WORK/'review_142584_small.zip'
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=digest(archive),
        output_manifest_verified=len(manifest),pointwise_npz_not_downloaded=sum(p.endswith('.npz') for p in manifest))))


if __name__=='__main__':run()
