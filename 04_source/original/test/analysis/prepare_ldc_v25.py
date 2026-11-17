"""Remote-only pinned LDC download/cache/audit; final test targets sealed."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from prepare_flowbench import download,header
from ldc_data_v25 import standard_split,input_fields,canonical_target,reference_physics

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/ldc_pinn_prepare_20261104_v25'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify():
    frozen=json.loads((WORK/'freeze.json').read_text())
    for p,h in frozen['files'].items():
        if digest(p)!=h:raise ValueError('Frozen source changed: '+p)


def extract(archive,output,expected):
    if output.exists():raise ValueError('Raw output already exists')
    partial=output.with_suffix('.npy.part')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['data.npy']:raise ValueError('Unexpected archive members')
        with z.open('data.npy') as f:
            shape,dtype=header(f)
        if tuple(shape)!=tuple(expected) or dtype!=np.dtype('float64'):raise ValueError('Header disagrees with pinned audit')
        with z.open('data.npy') as source,partial.open('xb') as target:shutil.copyfileobj(source,target,8*1024*1024)
    partial.rename(output)
    return np.load(output,mmap_mode='r',allow_pickle=False)


def run():
    verify();config=json.loads((WORK/'protocol.json').read_text());raw=WORK/'raw';raw.mkdir(exist_ok=False)
    archives={role:download(raw,config,role) for role in ['X','Y']}
    x=extract(archives['X'],raw/'X_original.npy',[1000,3,512,512])
    y=extract(archives['Y'],raw/'Y_original.npy',[1000,4,512,512])
    split=standard_split();save(WORK/'split.json',dict(indices=split,seed=20261104,
        policy='Authors standard sample-random80/20test then80/20trainvalidation; own deterministic IDs, not identical unpublished author IDs'))
    cache=WORK/'cache';cache.mkdir(exist_ok=False)
    low=np.lib.format.open_memmap(cache/'geometry_low.npy',mode='w+',dtype='float32',shape=(1000,2,64,64))
    masks=np.lib.format.open_memmap(cache/'fluid.npy',mode='w+',dtype='bool',shape=(1000,512,512))
    re_values=[];groups=[];schema=None
    for i in range(1000):
        re,fluid,sdf,meta=input_fields(np.asarray(x[i]))
        if schema is None:schema=meta
        if schema!=meta:raise ValueError('Mixed inputchannel schemas')
        low[i,0]=fluid.reshape(64,8,64,8).mean((1,3))
        low[i,1]=sdf.reshape(64,8,64,8).mean((1,3))
        masks[i]=fluid;re_values.append(re);groups.append(hashlib.sha256(np.packbits(fluid).tobytes()).hexdigest())
    low.flush();masks.flush();save(cache/'inputs.json',dict(reynolds=re_values,geometry_hashes=groups,schema=schema))
    total=np.zeros(3);squares=np.zeros(3);count=0;edge_rows=[];physics_rows=[]
    physics_ids=set(split['train'][::max(1,len(split['train'])//16)][:16])
    for role in ['train','validation']:
        ids=split[role]
        target=np.lib.format.open_memmap(cache/('Y_'+role+'.npy'),mode='w+',dtype='float32',shape=(len(ids),3,512,512))
        for slot,i in enumerate(ids):
            fluid=np.asarray(masks[i]);value=canonical_target(np.asarray(y[i]),fluid);target[slot]=value
            if role=='train':
                cells=np.asarray(value[:,fluid],dtype=np.float64);total+=cells.sum(1);squares+=(cells*cells).sum(1);count+=cells.shape[1]
                if i in physics_ids:
                    physics_rows.append(dict(sample=i,reynolds=re_values[i],**reference_physics(value,fluid,re_values[i])))
                    edge_rows.append(dict(sample=i,top_u=float(value[0,-1,8:-8].mean()),bottom_u=float(value[0,0,8:-8].mean()),
                        left_u=float(value[0,8:-8,0].mean()),right_u=float(value[0,8:-8,-1].mean())))
            if slot%100==0:print('CACHE',role,slot,len(ids),flush=True)
        target.flush()
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0)).clip(1e-4)
    train_low=np.asarray(low[split['train']],dtype=np.float64)
    save(cache/'normalization.json',dict(target_mean=mean.tolist(),target_std=std.tolist(),target_fluid_points=count,
        geometry_mean=train_low.mean((0,2,3)).tolist(),geometry_std=train_low.std((0,2,3)).clip(1e-4).tolist(),source='TRAIN only; pressuregauge fluid mean per sample; no TESTtargets'))
    rel=float(np.median([r['momentum_relative'] for r in physics_rows]))
    multiplier=float(np.median([r['inferred_viscosity_multiplier'] for r in physics_rows]))
    top=float(np.median([r['top_u'] for r in edge_rows]));bottom=float(np.median([r['bottom_u'] for r in edge_rows]))
    physics_ready=rel<=.35 and .5<=multiplier<=1.5 and top>bottom+.5
    by_role={role:set(groups[i] for i in ids) for role,ids in split.items()}
    prepared=dict(status='LDC_DATA_READY' if physics_ready else 'LDC_REQUIRES_PHYSICS_CONVENTION_REVIEW',
        revision=config['revision'],family='harmonics',cases=1000,split_counts={k:len(v) for k,v in split.items()},
        native_resolution=[512,512],input_channels=schema,output_fields=['u','v','p'],unused_output_channel='PackedCd/Cl, excludedfromfieldmodel',
        geometry_count=len(set(groups)),geometry_overlaps={a+'_'+b:len(by_role[a]&by_role[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
        split_meaning='Standard random flowcase split; not geometry-disjoint generalization',
        physical_domain=[[0,2],[0,2]],model_coordinates=[[0,1],[0,1]],coordinates='Cell centres; xi=xphysical/2; physicalnu2/Re correspondsnormalizednu1/Re',
        median_relative_ns_residual=rel,median_inferred_viscosity_multiplier=multiplier,physics_audit=physics_rows,boundary_audit=edge_rows,
        top_u_median=top,bottom_u_median=bottom,physics_ready=physics_ready,
        test_target_samples_read=0,train_label_samples_read=640,validation_label_samples_read=160,
        no_time_dimension=True,shared_model_required=True,new_CFD=False,main_test_opened=False,
        training_started=False,cache=str(cache),rawtest_targets_sealed_in=str(raw/'Y_original.npy'))
    save(WORK/'prepared.json',prepared)
    files=[cache/n for n in ['geometry_low.npy','fluid.npy','inputs.json','Y_train.npy','Y_validation.npy','normalization.json']]
    save(WORK/'cache_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files})
    verify();print('DATA_PREPARATION_COMPLETE',json.dumps({k:prepared[k] for k in ['status','split_counts','geometry_count','median_relative_ns_residual','median_inferred_viscosity_multiplier','top_u_median','bottom_u_median','physics_ready']}),flush=True)


if __name__=='__main__':run()
