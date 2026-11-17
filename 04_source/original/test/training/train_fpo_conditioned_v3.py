"""Controlled pilot accuracy ablations, then geometry/Re development training."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from fpo_conditioned_v3 import (WORK, ConditionalPINN, residual, load_case, errors,
                                save, digest, verify)


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float32, device='cuda')


def grouped_metrics(pred, ref, frames):
    rows = [errors(pred[frames == k], ref[frames == k]) for k in np.unique(frames)]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def context(model, case):
    return model.context(case['gpu_geometry'], case['record']['reynolds'])


def quick_validation(model, cases, prefix):
    rows = []
    model.eval()
    with torch.inference_mode():
        for case in cases:
            c = context(model, case)
            q, y, frames = (case['samples'][prefix + s] for s in ('_q', '_y', '_frame'))
            pieces = [model(tensor(q[j:j+8192]), c).cpu().numpy() for j in range(0, len(q), 8192)]
            metrics = grouped_metrics(np.concatenate(pieces), y, frames)
            rows.append(dict(path=case['record']['path'], **metrics))
    model.train()
    return float(np.mean([(r['velocity_relative_l2'] + r['pressure_relative_l2']) / 2 for r in rows])), rows


def latency(model, case):
    model.eval()
    gx, gy = np.meshgrid(4 + (np.arange(1024) + .5) / 64, 6 + (np.arange(256) + .5) / 64)
    host = torch.tensor(np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, 6.))), dtype=torch.float32)
    q = host.to('cuda')
    timings, transfer, cold = [], [], []
    with torch.inference_mode():
        c = context(model, case)
        for _ in range(30):
            model(q, c)
        torch.cuda.synchronize()
        for repeat in range(3):
            values = []
            for _ in range(200):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); result = model(q, c); end.record(); end.synchronize()
                values.append(start.elapsed_time(end))
            timings.append({name: float(np.percentile(values, p)) for name, p in [('p50', 50), ('p95', 95), ('p99', 99)]})
        for _ in range(30):
            torch.cuda.synchronize(); start = time.perf_counter()
            c2 = context(model, case)
            torch.cuda.synchronize(); cold.append(1000 * (time.perf_counter() - start))
            torch.cuda.synchronize(); start = time.perf_counter()
            result = model(host.to('cuda'), c2).cpu()
            torch.cuda.synchronize(); transfer.append(1000 * (time.perf_counter() - start))
    return dict(grid=[1024, 256], gpu_resident_rounds_ms=timings, samples_per_round=200,
                context_gpu_input_ms={k: float(np.percentile(cold, p)) for k, p in [('p50',50),('p99',99)]},
                host_to_host_ms={k: float(np.percentile(transfer, p)) for k, p in [('p50',50),('p99',99)]},
                context_samples=30, includes_rendering=False, includes_geometry_preprocessing=False,
                warmed_gpu=True, batch_scenarios=1, timing_clock='CUDA events for GPU; perf_counter+synchronize for transfers')


def full_evaluation(model, cases, out, temporal_roles):
    model.eval()
    gx, gy = np.meshgrid(4 + (np.arange(1024) + .5) / 64, 6 + (np.arange(256) + .5) / 64)
    q = tensor(np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size))))
    rows = []; summaries = []
    with torch.inference_mode():
        for case in cases:
            c = context(model, case); local = []
            for k in range(2, 242):
                q[:, 2] = (k - 2) * .05
                # Chunk only accuracy evaluation to bound peak memory; latency
                # above measures a single full-grid forward call.
                pred = torch.cat([model(chunk, c) for chunk in q.split(65536)]).cpu().numpy().reshape(256, 1024, 3)
                mask = case['fluid']; reference = np.asarray(case['field'][k])[mask]
                metrics = errors(pred[mask], reference)
                if temporal_roles:
                    role = 'train' if (k-2) % 6 == 0 else ('validation' if (k-5) % 6 == 0 else 'heldout_time')
                else:
                    role = case['record']['proposed_split']
                local.append(dict(path=case['record']['path'], frame=k, relative_time=(k-2)*.05, role=role, **metrics))
            for role in sorted({r['role'] for r in local}):
                subset = [r for r in local if r['role'] == role]
                summaries.append(dict(path=case['record']['path'], role=role, frames=len(subset),
                                      **{key: float(np.mean([r[key] for r in subset])) for key in metrics}))
            rows.extend(local)
            print('FULL_GRID_EVALUATED', case['record']['path'], flush=True)
    save(out / 'per_frame_accuracy.json', rows)
    return summaries


def train(config, train_cases, validation_cases, out, temporal=False):
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', config)
    seed = config['seed']; torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    prefix = 'pilot_train' if temporal else 'all'
    val_prefix = 'pilot_val' if temporal else 'validation'
    targets = np.concatenate([c['samples'][prefix + '_y'] for c in train_cases])
    mean = targets.mean(0); std = np.maximum(targets.std(0), 1e-4)
    model = ConditionalPINN(config, mean, std).cuda()
    save(out / 'normalization.json', dict(mean=mean.tolist(), std=std.tolist(), source='train cases and train frames only'))
    for case in train_cases + validation_cases:
        case['gpu_geometry'] = tensor(case['geometry'][None])
    pools = []
    for case in train_cases:
        pools.append((tensor(case['samples'][prefix + '_q']), tensor(case['samples'][prefix + '_y']),
                      tensor(case['samples']['initial_q']), tensor(case['samples']['initial_y'])))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['steps'], eta_min=config['lr']*.05)
    history = []; best = float('inf'); start = time.monotonic()
    data_ms = 0.; validation_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, config['steps'] + 1):
        ci = int(rng.integers(len(train_cases))); case = train_cases[ci]
        qpool, ypool, iqpool, iypool = pools[ci]
        ids = torch.randint(len(qpool), (config['batch'],), device='cuda')
        q, target = qpool[ids], ypool[ids]
        optimizer.zero_grad(set_to_none=True)
        c = context(model, case)
        data_loss = ((model(q, c) - target) / model.std).square().mean()
        pde_loss = torch.zeros((), device='cuda'); bc_loss = pde_loss; ic_loss = pde_loss
        ramp = min(1., max(0., (step - config['warmup']) / config['ramp']))
        if config['pde_weight'] and ramp:
            yx = case['interior_yx'][rng.integers(len(case['interior_yx']), size=config['collocation'])]
            qp = tensor(np.column_stack((4 + (yx[:,1]+.5)/64, 6 + (yx[:,0]+.5)/64,
                                         rng.uniform(0, 11.95, config['collocation'])))).requires_grad_(True)
            pde_loss = residual(qp, model(qp, c), case['record']['reynolds']).square().sum(1).mean()
        if config['bc_weight']:
            boundary = case['boundary'][rng.integers(len(case['boundary']), size=256)]
            qb = tensor(np.column_stack((boundary, rng.uniform(0,11.95,len(boundary)))))
            bc_loss = (model(qb, c)[:, :2] / model.std[:2]).square().mean()
        if config['ic_weight']:
            ids = torch.randint(len(iqpool), (256,), device='cuda')
            ic_loss = ((model(iqpool[ids], c) - iypool[ids]) / model.std).square().mean()
        loss = data_loss + ramp * config['pde_weight'] * pde_loss + config['bc_weight']*bc_loss + config['ic_weight']*ic_loss
        if not torch.isfinite(loss):
            save(out / 'failure.json', dict(step=step, reason='nonfinite loss'))
            raise RuntimeError('Nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); schedule.step()
        if step == 1 or step % config['validate_every'] == 0 or step == config['steps']:
            vstart = time.monotonic()
            score, vrows = quick_validation(model, validation_cases, val_prefix)
            validation_seconds += time.monotonic() - vstart
            row = dict(step=step, data=float(data_loss.detach()), pde=float(pde_loss.detach()),
                       boundary=float(bc_loss.detach()), initial=float(ic_loss.detach()),
                       validation_score=score, cases=vrows, elapsed_seconds=time.monotonic()-start,
                       lr=optimizer.param_groups[0]['lr'], ramp=ramp)
            history.append(row); save(out / 'history.json', history)
            state = dict(config=config, state=model.state_dict(), step=step, validation_score=score,
                         source_freeze=digest(WORK/'freeze.json'))
            torch.save(state, out/'last.pt')
            if score < best:
                best = score; torch.save(state, out/'best.pt')
            print(config['name'], step, 'VAL', score, 'DATA', row['data'], 'PDE', row['pde'], flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    state = torch.load(out/'best.pt', weights_only=True)
    model.load_state_dict(state['state']); model.eval()
    accuracy = full_evaluation(model, train_cases if temporal else validation_cases, out, temporal)
    speed = latency(model, validation_cases[0])
    validation_accuracy = [r for r in accuracy if r['role'] == 'validation']
    accurate = bool(validation_accuracy) and all(r['velocity_relative_l2'] <= .10 and r['pressure_relative_l2'] <= .20 for r in validation_accuracy)
    fast = all(r['p99'] <= 5 for r in speed['gpu_resident_rounds_ms'])
    result = dict(status='DEVELOPMENT_TRAINING_EVALUATED', config=config,
                  selected_step=state['step'], best_sampled_validation_score=best,
                  training_process_seconds=elapsed, validation_seconds=validation_seconds,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  train_scenarios=len(train_cases), validation_scenarios=len(validation_cases) if not temporal else 0,
                  accuracy=accuracy, latency=speed, checkpoint_sha256=digest(out/'best.pt'),
                  provisional_engineering_gate=dict(accuracy_pass=accurate, gpu_latency_pass=fast,
                      eligible_for_multi_seed_confirmation=accurate and fast and not temporal,
                      thresholds=dict(velocity=.10, pressure=.20, gpu_p99_ms=5)),
                  independent_main_test_opened=False, temporal_interpolation_only=temporal,
                  llm_comparison_ready=False,
                  limitations=['Data-assisted coordinate reconstruction, not causal rollout',
                               'Time interval is solver-relative, not SI seconds or physical startup',
                               'Only selected development geometries; not the full 775 training scenarios',
                               'Readiness needs additional seeds, residual audit and protocol freeze'])
    save(out/'summary.json', result)
    return result


def run():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['ablations','conditional'], required=True)
    args = parser.parse_args(); verify()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    records = json.loads((WORK/'prepared.json').read_text())['records']
    # Verify all small caches; full reference file hashes were verified by CPU preparation.
    for r in records:
        if r['proposed_split'] == 'test': raise ValueError('Sealed main test')
        for name in ('geometry.npz','samples.npz'):
            path = str(Path(r['cache'])/name)
            if digest(path) != r['hashes'][path]: raise ValueError('Prepared cache changed')
    protocol = json.loads((WORK/'protocol.json').read_text())
    out = WORK/'training'/os.environ['SLURM_JOB_ID']/args.mode
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == 'ablations':
        record = next(r for r in records if r['family']=='harmonics' and r['geometry_id']=='1' and r['reynolds']==382)
        case = load_case(record)
        results=[]
        for config in protocol['ablations']:
            results.append(train(config, [case], [case], out/config['name'], temporal=True))
            torch.cuda.empty_cache()
    else:
        train_cases = [load_case(r, mmap=False) for r in records if r['proposed_split']=='train']
        val_cases = [load_case(r) for r in records if r['proposed_split']=='validation']
        results=[]
        for config in protocol['conditional_candidates']:
            results.append(train(config, train_cases, val_cases, out/config['name']))
            torch.cuda.empty_cache()
    save(out/'summary.json', dict(mode=args.mode, status='COMPLETE', results=results))


if __name__ == '__main__':
    run()
