"""Read-only region and conditioning diagnostics. Perturbations are not CFD cases."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from fpo_common_v20 import WORK,save,digest,verify
from fpo_direct_curl_v15 import DirectPINN
from fpo_step_v5a import grid,project_known_constraints,fingerprint
from train_fpo_direct_curl_v15 import query
from audit_fpo_physics_v18 import known_case
from fpo_context_audit_v20 import contexts,regional_errors,prediction_change,ZONE_NAMES,zones_from_distance


def prefix_only(record):
    if record['proposed_split'] not in ['train','validation']:raise ValueError('Sealed role')
    if digest(record['state_cache'])!=record['state_cache_sha256']:raise ValueError('Changed prefix')
    p=str(Path(record['cache'])/'geometry.npz')
    if digest(p)!=record['hashes'][p]:raise ValueError('Changed geometry')
    with np.load(p,allow_pickle=False) as z:geometry=z['encoded'].copy()
    with np.load(record['state_cache'],allow_pickle=False) as z:raw=z['features'].copy()
    return dict(raw=torch.tensor(raw,device='cuda'),geometry=torch.tensor(geometry[None],device='cuda'),record={'reynolds':record['reynolds']})


def predict(model,known,qbase,t,context):
    q=qbase.clone();q[:,2]=t
    value=project_known_constraints(model.grid_prediction(q,context,known['raw']),known['fluid'])
    if not torch.isfinite(value).all() or value.abs().max()>1000:raise RuntimeError('Invalid prediction, no clipping')
    return value


def latent_neighbours(model,records):
    vectors=[]
    with torch.inference_mode():
        for r in records:
            known=prefix_only(r);raw=known['raw'];ny,nx=raw.shape[:2]
            low=raw.reshape(ny//8,8,nx//8,8,6).mean((1,3)).permute(2,0,1)[None]
            vectors.append(model.context(known['geometry'],low,r['reynolds'])[0,:-1].cpu().numpy())
    z=np.asarray(vectors,dtype=np.float64);train=np.array([r['proposed_split']=='train' for r in records])
    mean=z[train].mean(0);std=z[train].std(0);safe=np.maximum(std,1e-6);normalized=(z-mean)/safe
    result=[]
    for i,r in enumerate(records):
        candidates=[j for j,v in enumerate(records) if v['proposed_split']=='train' and (v['family'],v['geometry_id'])!=(r['family'],r['geometry_id'])]
        distances=np.sqrt(np.mean((normalized[candidates]-normalized[i])**2,axis=1))
        order=np.argsort(distances)[:3]
        result.append(dict(path=r['path'],scope=r['proposed_split'],latent=z[i].tolist(),
            nearest_other_geometry_train=[dict(path=records[candidates[int(k)]]['path'],distance=float(distances[k])) for k in order]))
    return dict(records=result,train_only_mean=mean.tolist(),train_only_std=std.tolist(),std_floor=1e-6,
        constant_dimensions=int((std<1e-6).sum()),includes_explicit_Re_channel=False,
        interpretation='Inputembeddingdistance diagnostic, not information-theoretic proof or physical-neighbour equivalence')


def aggregate(rows,mode):
    chosen=[r for r in rows if r['mode']==mode];result={}
    if not chosen:raise ValueError('Empty aggregation')
    for region in ['all_fluid']+ZONE_NAMES:
        a=[r['regions'][region] for r in chosen if not r['regions'][region]['empty']]
        if not a:result[region]=dict(empty=True);continue
        keys=['velocity_relative_l2','pressure_relative_l2','velocity_rmse','pressure_rmse','reference_velocity_rms','reference_pressure_rms','area_fraction','velocity_error_fraction','pressure_error_fraction']
        record={k:float(np.mean([r[k] for r in a if r[k] is not None])) if any(r[k] is not None for r in a) else None for k in keys}
        den_u=sum(r['regions']['all_fluid']['velocity_squared_error'] for r in chosen)
        den_p=sum(r['regions']['all_fluid']['pressure_squared_error'] for r in chosen)
        record['pooled_velocity_error_fraction']=sum(r['velocity_squared_error'] for r in a)/den_u if den_u>1e-24 else 0.
        record['pooled_pressure_error_fraction']=sum(r['pressure_squared_error'] for r in a)/den_p if den_p>1e-24 else 0.
        result[region]=record
    return dict(case_time_count=len(chosen),regions=result,
        mean_prediction_change={k:float(np.mean([r['prediction_change'][k] for r in chosen])) for k in chosen[0]['prediction_change']})


def run():
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    protocol=json.loads((WORK/'protocol.json').read_text());receipt=json.loads((WORK/'prepared.json').read_text());records=receipt['records']
    if len(records)!=75 or any(r['proposed_split']=='test' for r in records):raise ValueError('Existing development75only')
    selected=[r for r in records if r['path'] in protocol['audit_paths']]
    if (sum(r['proposed_split']=='train' for r in selected),sum(r['proposed_split']=='validation' for r in selected))!=(3,6):raise ValueError('Expected3TRAIN6VAL')
    cases=[known_case(r) for r in selected]
    normal=json.loads(Path(receipt['normalization']).read_text());qbase=grid()
    out=WORK/'audit'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False);results=[]
    for checkpoint in protocol['checkpoints']:
        chosen=torch.load(checkpoint['path'],weights_only=True,map_location='cuda')
        model=DirectPINN(chosen['config'],normal['mean'],normal['std']).cuda();model.load_state_dict(chosen['state']);model.eval();model.requires_grad_(False)
        target=out/checkpoint['name'];target.mkdir()
        save(target/'latent_neighbours.json',latent_neighbours(model,records));rows=[];context_stats=[]
        for i,(known,scoring) in enumerate(cases):
            path=scoring['record']['path'];donor=protocol['geometry_donors'][path]
            with np.load(donor['geometry_cache'],allow_pickle=False) as z:donor_geometry=torch.tensor(z['encoded'][None],device='cuda')
            with np.load(protocol['zone_files'][path],allow_pickle=False) as z:zones=z['zones'].copy();distance=z['distance'].copy()
            np.testing.assert_array_equal(zones,zones_from_distance(scoring['fluid_np'],distance))
            expected={r['offset']:r for r in json.loads(Path(checkpoint['expected_per_frame'][path]).read_text())}
            case_rows=[]
            with torch.inference_mode():
                ctx=contexts(model,known,donor_geometry)
                original_context=ctx['original'][:,:-1]
                context_stats.append(dict(path=path,scope=scoring['record']['proposed_split'],geometry_donor_path=donor['path'],
                    latent_norm=float(original_context.norm()),latent_change={m:float((v[:,:-1]-original_context).square().mean().sqrt()) for m,v in ctx.items()}))
                for mode,c in ctx.items():
                    torch.testing.assert_close(c[:,-1],ctx['original'][:,-1],rtol=0,atol=0)
                    torch.testing.assert_close(predict(model,known,qbase,0.,c),known['raw'][:,:,:3],rtol=1e-5,atol=1e-6)
                for offset in protocol['offsets']:
                    t=offset*.05;original=predict(model,known,qbase,t,ctx['original'])
                    if offset==1:torch.testing.assert_close(original,query(model,known,qbase,t),rtol=1e-5,atol=1e-6)
                    original_np=original.cpu().numpy();reference=np.asarray(scoring['reference'][offset+3])
                    for mode,c in ctx.items():
                        value=original_np if mode=='original' else predict(model,known,qbase,t,c).cpu().numpy()
                        regions=regional_errors(value,reference,scoring['fluid_np'],zones)
                        if mode=='original':
                            for k in ['velocity_relative_l2','pressure_relative_l2']:
                                if not np.isclose(regions['all_fluid'][k],expected[offset]['model'][k],rtol=1e-5,atol=1e-7):raise ValueError('Baseline disagrees with frozen prior scoring')
                        row=dict(path=path,scope=scoring['record']['proposed_split'],offset=offset,frame=offset+3,time=t,
                            mode=mode,physical_input_consistency=(mode=='original'),regions=regions,
                            prediction_change=prediction_change(value,original_np,scoring['fluid_np']))
                        rows.append(row);case_rows.append(row)
                    print('CONTEXT_AUDIT',checkpoint['name'],i,offset,flush=True)
            if fingerprint(scoring['record']['field'])!=scoring['record']['field_fingerprint']:raise ValueError('Reference changed')
            save(target/f'case_{i}.json',dict(path=path,rows=case_rows))
        modes=protocol['modes']
        result=dict(checkpoint=checkpoint,by_scope={role:{mode:aggregate([r for r in rows if r['scope']==role],mode) for mode in modes} for role in ['train','validation']},
            original_by_scope_horizon={role:{label:aggregate([r for r in rows if r['scope']==role and r['offset'] in offsets],'original')
                for label,offsets in [('early',[1,5,10]),('middle',[20,40]),('late',[80,160,238])]} for role in ['train','validation']},
            context_stats=context_stats,optimizer_updates=0,physical_input_consistency_only_for_original=True)
        save(target/'summary.json',result);results.append(result);del model,chosen;torch.cuda.empty_cache()
    verify()
    save(out/'summary.json',dict(status='READ_ONLY_CONTEXT_AUDIT_COMPLETE',results=results,optimizer_updates=0,
        full_238frame_accuracy=False,main_test_opened=False,independent_final_test=False,llm_ready=False))


if __name__=='__main__':run()
