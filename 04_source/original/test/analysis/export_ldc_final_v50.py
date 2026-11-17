"""Completed research archive parts, built and verified on Perun, not on the PC."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import ROOT,WORK,read,save,digest,verify,fingerprint

FINAL=ROOT/'results/ldc_final_test_20261108_v49'
AUDIT=ROOT/'results/ldc_final_closeout_20261108_v50a'
BASE=ROOT/'results/ldc_final_backup_20261108_v50'
SECRET=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated CPU node required')
    verify();assert read(AUDIT/'summary.json')['status']=='FINAL_OUTPUTS_AUDITED'
    freeze=read(FINAL/'freeze.json')
    assert all(digest(p)==h for p,h in freeze['files'].items())
    assert all(fingerprint(p)==s for p,s in freeze['input_fingerprints'].items())
    state=read(WORK/'main/state.json');assert len(state['tasks'])==395 and all(r['status']=='DONE' for r in state['tasks'].values())
    final_state=read(FINAL/'evaluation/state.json');assert len(final_state['tasks'])==155 and all(r['status']=='DONE' for r in final_state['tasks'].values())
    stamp='final_20261108_'+str(int(time.time()));out=BASE/stamp;out.mkdir(parents=True,exist_ok=False)
    final_weights={Path(m['checkpoint']) for m in read(FINAL/'models.json')['models']}
    primary=set(final_weights)
    for folder in [FINAL,AUDIT,ROOT/'results/ldc_final_closeout_20261108_v50']:
        primary.update(p for p in folder.rglob('*') if p.is_file())
    source_manifests=[WORK/'freeze.json',WORK/'operational_manifest_v36.json',FINAL/'freeze.json']
    for p in source_manifests:primary.add(p);primary.update(Path(x) for x in read(p)['files'])
    for sub in ['analysis','training','scripts']:
        for version in ['v49','v50','v50a']:
            primary.update((ROOT/'test'/sub).glob('*'+version+'.py'))
            primary.update((ROOT/'test'/sub).glob('*'+version+'.sbatch'))
    provenance={p for p in WORK.rglob('*') if p.is_file()}-primary
    for folder in ['ldc_full_audit_20261106_v44','ldc_status_20261106_v45','ldc_status_20261107_v47','ldc_status_20261108_v48']:
        provenance.update(p for p in (ROOT/'results'/folder).rglob('*') if p.is_file())
    records=[]
    for category,files in [('final_models_results',primary),('training_provenance',provenance)]:
        for p in sorted(files):
            relative=p.relative_to(ROOT)
            if p.is_symlink() or p.suffix in ['.pyc','.tmp','.lock']:continue
            assert relative.parts[0] in ['test','results']
            if p.suffix!='.pt' and p.stat().st_size<=32*1024*1024 and SECRET.search(p.read_bytes()):
                raise ValueError('Credential-like content; filename only: '+str(relative))
            records.append(dict(path=str(relative),category=category,bytes=p.stat().st_size,sha256=digest(p)))
    assert len({r['path'] for r in records})==len(records)
    # Bound uncompressed payload to 450 MB per part. No giant local download.
    chunks=[]
    for category in ['final_models_results','training_provenance']:
        bucket=[];size=0
        for row in [r for r in records if r['category']==category]:
            if size+row['bytes']>450000000 and bucket:chunks.append((category,bucket));bucket=[];size=0
            if row['bytes']>450000000:raise ValueError('Unexpectedly large individual research artifact: '+row['path'])
            bucket.append(row);size+=row['bytes']
        if bucket:chunks.append((category,bucket))
    manifest=dict(status='COMPLETE_MAIN_AND_FINAL_TEST_RESEARCH_SNAPSHOT',snapshot=stamp,build_job=os.environ['SLURM_JOB_ID'],
        completed_trainings=395,final_models=155,test_cases=200,case_model_pairs=31000,TEST_opened=True,
        primary_best_method='tpe',main_freeze_sha256=digest(WORK/'freeze.json'),final_freeze_sha256=digest(FINAL/'freeze.json'),
        audit_sha256=digest(AUDIT/'summary.json'),files=records,uncompressed_bytes=sum(r['bytes'] for r in records),
        excludes='Raw FlowBench reference/input arrays, downloaded LLM weights, environments, credentials, 3D article folder and older pilot archives already backed up separately.',
        notes='Own best/last and optimizer/RNG checkpoints, LLM raw proposals/reasons/resources, validation histories, final TEST case metrics and raw timings included. No license grant over user research; preserve upstream license notices. Steady 2D fields and known-geometry split.')
    save(out/'manifest.json',manifest);parts=[]
    for number,(category,rows) in enumerate(chunks,1):
        archive=out/(f'{stamp}_{category}_part{number:02d}.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
            tar.add(out/'manifest.json',arcname='manifest.json',recursive=False)
            for row in rows:
                p=ROOT/row['path'];assert digest(p)==row['sha256']
                tar.add(p,arcname='project/'+row['path'],recursive=False)
        expected={'project/'+r['path']:r['sha256'] for r in rows};expected['manifest.json']=digest(out/'manifest.json');seen=set()
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('Unsafe member')
                h=hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                if member.name in seen or h.hexdigest()!=expected[member.name]:raise ValueError('Archive member hash mismatch')
                seen.add(member.name)
        assert seen==set(expected) and archive.stat().st_size<500000000
        part=dict(category=category,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),member_count=len(rows))
        parts.append(part);print('ARCHIVE_VERIFIED',number,len(chunks),part['archive_bytes'],flush=True)
    index=dict(snapshot=stamp,status='ALL_ARCHIVE_PARTS_HASH_VERIFIED',parts=parts,
        manifest_path=str(out/'manifest.json'),manifest_bytes=(out/'manifest.json').stat().st_size,manifest_sha256=digest(out/'manifest.json'),
        source_files=len(records),uncompressed_bytes=manifest['uncompressed_bytes'],archive_total_bytes=sum(p['archive_bytes'] for p in parts),
        completed_trainings=395,final_models=155,test_cases=200,TEST_opened=True,github_uploaded=False,archive_stored_on_pc=False)
    save(out/'index.json',index);save(BASE/'latest.json',index);verify();print(json.dumps(index,indent=2),flush=True)


if __name__=='__main__':main()
