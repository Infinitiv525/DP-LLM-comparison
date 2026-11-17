"""Full-VAL boundary, AD/FD and native timing audit of a fixed checkpoint."""
import os
import time
import traceback
import numpy as np
import torch
from ldc_quality_common_v31 import ROOT, WORK, PARENT, read, save, digest, verify
from ldc_quality_math_v31 import differential, describe, rms, vector_error
from ldc_shared_pinn_v26 import SharedSteadyPINN, steady_residual
from ldc_store_v26 import Store, query, conditioning, interior_mask, obstacle_faces
from ldc_data_v25 import reference_physics
from fpo_state_v4 import errors


def continuous(model, context, points, residual=False):
    q = torch.tensor(points, dtype=torch.float32, device='cuda', requires_grad=True)
    values = model(q, context)
    result = values.detach().cpu().numpy()
    if not residual:
        return result
    return result, steady_residual(q, values, context['reynolds_for_audit']).detach().cpu().numpy()


def boundary_audit(model, context, fluid, prediction, reference, protocol):
    n = protocol['boundary']['outer_points_per_side']
    eps = protocol['boundary']['corner_exclusion']
    s = np.linspace(eps, 1-eps, n)
    points = [np.column_stack((s, s*0)), np.column_stack((s, s*0+1)),
              np.column_stack((s*0, s)), np.column_stack((s*0+1, s))]
    labels = ['bottom', 'top', 'left', 'right']
    result = {}
    for label, q in zip(labels, points):
        value = continuous(model, context, q)[:, :2]
        target = np.zeros_like(value)
        if label == 'top':
            target[:, 0] = 1.
        result[label] = vector_error(value, target)
    body = obstacle_faces(fluid)
    value = continuous(model, context, body)[:, :2]
    result['obstacle'] = dict(points=len(body), **vector_error(value, np.zeros_like(value)))
    # These are cell centres, not exact wall values. Compare to the reference.
    indices = [(0, slice(1, -1)), (-1, slice(1, -1)),
               (slice(1, -1), 0), (slice(1, -1), -1)]
    result['outer_cell_centres_prediction_vs_reference'] = {
        label: vector_error(prediction[index][:, :2], reference[index][:, :2])
        for label, index in zip(labels, indices)}
    return result


def compare_differentiation(model, context, known, fluid, prediction, protocol):
    cells = np.column_stack(np.where(interior_mask(fluid)))
    rng = np.random.default_rng(protocol['differential']['seed'] + known['index'])
    cells = cells[rng.choice(len(cells), protocol['differential']['points_per_case'], replace=False)]
    y, x = cells.T
    q = np.column_stack(((x+.5)/512, (y+.5)/512)).astype('float32')
    context['reynolds_for_audit'] = known['reynolds']
    ad_value, ad_ns = continuous(model, context, q, residual=True)
    with torch.no_grad():
        sampled = model.sampled_prediction(torch.tensor(q, device='cuda'), context,
            torch.tensor(y.astype('int64'), device='cuda'), torch.tensor(x.astype('int64'), device='cuda')).cpu().numpy()
    native = prediction[y, x, :2]
    delta = sampled[:, :2].astype('float64') - native
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(native), 1e-24))
    absolute = float(np.max(np.abs(delta)))
    fd_ns, _, _, _ = differential(prediction, known['reynolds'])
    fd_sample = fd_ns[y, x]
    return dict(points=len(q), seed=protocol['differential']['seed'] + known['index'],
        sampled_stencil_vs_native=dict(relative_velocity_l2=relative, maximum_absolute=absolute,
            pass_tolerance=relative <= protocol['differential']['stencil_relative_tolerance']
                           and absolute <= protocol['differential']['stencil_max_absolute_tolerance']),
        continuous_velocity_vs_native=dict(relative_l2=float(np.linalg.norm(ad_value[:, :2] - native)
                 / max(np.linalg.norm(native), 1e-24)), component_rms=rms(ad_value[:, :2] - native)),
        continuous_momentum_rms=rms(ad_ns[:, :2]), grid_momentum_rms_at_same_points=rms(fd_sample[:, :2]),
        ad_minus_fd_momentum_rms=rms(ad_ns[:, :2] - fd_sample[:, :2]),
        ad_divergence_rms=rms(ad_ns[:, 2]), fd_divergence_rms=rms(fd_sample[:, 2]),
        note='AD uses continuous first/second velocity derivatives; FD uses native curl and grid derivatives. Pressure gauge cancels in both.')


def run():
    started = time.monotonic()
    verify()
    protocol = read(WORK / 'protocol.json')
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name():
        raise ValueError('One H200 required')
    original_env = read(PARENT / 'training/143115/environment.json')
    torch.backends.cuda.matmul.allow_tf32 = original_env['allow_tf32']
    torch.backends.cudnn.allow_tf32 = original_env['cudnn_allow_tf32']
    checkpoint_path = protocol['checkpoint']
    if digest(checkpoint_path) != protocol['checkpoint_sha256']:
        raise ValueError('Wrong selected checkpoint hash')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['source_freeze'] != digest(PARENT / 'freeze.json') or checkpoint['update'] != protocol['selected_update']:
        raise ValueError('Wrong selected checkpoint provenance')
    prepared = read(protocol['prepared_path'])
    split = read(protocol['split_path'])['indices']
    store = Store(prepared, split, False)
    model = SharedSteadyPINN(checkpoint['config'], store.normalization).cuda()
    model.load_state_dict(checkpoint['state'])
    model.eval().requires_grad_(False)
    old = {r['sample']: r for r in read(PARENT / 'training/143115/selected_validation/summary.json')['cases']}
    save(WORK / 'environment.json', dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), job=os.environ['SLURM_JOB_ID'],
        precision='float32', allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        checkpoint_sha256=protocol['checkpoint_sha256'], source_freeze_sha256=digest(WORK / 'freeze.json')))
    (WORK / 'examples').mkdir(exist_ok=False)
    rows = []
    for number, index in enumerate(store.validation_ids, 1):
        known = store.known(index)
        fluid = np.asarray(store.fluid_np[index])
        reference = np.moveaxis(store.validation_reference(index), 0, 2)
        with torch.inference_mode():
            for _ in range(protocol['timing']['warmup']):
                query(model, known, store.qgrid)
            torch.cuda.synchronize()
            timings = []
            for _ in range(protocol['timing']['repeats']):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                value = query(model, known, store.qgrid)
                b.record()
                b.synchronize()
                timings.append(a.elapsed_time(b))
            prediction = value.cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite model output')
        metrics = errors(prediction[fluid], reference[fluid])
        metrics['mse_all'] = float(np.mean((prediction.astype('float64') - reference)**2))
        # Detached conditioning features are valid here: no weight optimization,
        # while their interpolation remains differentiable with respect to x,y.
        with torch.no_grad():
            context = conditioning(model, known)
        bc = boundary_audit(model, context, fluid, prediction, reference, protocol)
        differentiation = compare_differentiation(model, context, dict(known, index=index), fluid, prediction, protocol)
        row = dict(sample=index, reynolds=known['reynolds'], metrics=metrics,
            timing=dict(p50=float(np.percentile(timings, 50)), p95=float(np.percentile(timings, 95)),
                        p99=float(np.percentile(timings, 99)), raw_ms=timings),
            boundary=bc, differentiation=differentiation,
            physics=reference_physics(np.moveaxis(prediction, 2, 0), fluid, known['reynolds']),
            reference_physics=reference_physics(np.moveaxis(reference, 2, 0), fluid, known['reynolds']),
            reproduction_max_metric_difference=max(abs(metrics[k] - old[index][k])
                for k in ['velocity_relative_l2', 'pressure_relative_l2', 'mse_all']))
        rows.append(row)
        if str(index) in protocol['examples']:
            np.savez_compressed(WORK / 'examples' / f'case_{index}.npz', prediction=prediction,
                reference=reference, fluid=fluid, reynolds=known['reynolds'])
        if number % 10 == 0 or number == len(store.validation_ids):
            save(WORK / 'progress.json', dict(completed=number, total=160, elapsed_seconds=time.monotonic()-started,
                                             last_case=index, main_test_opened=False))
            print('AUDITED', number, 'of160', 'p99', row['timing']['p99'], flush=True)
    checks = dict(exact_160_validation_cases=len(rows) == 160 and {r['sample'] for r in rows} == set(split['validation']),
        checkpoint_unchanged=digest(checkpoint_path) == protocol['checkpoint_sha256'],
        native_grid_stencil_consistent=all(r['differentiation']['sampled_stencil_vs_native']['pass_tolerance'] for r in rows),
        previous_validation_reproduced=max(r['reproduction_max_metric_difference'] for r in rows) <= 1e-5,
        p99_all_cases_within_5ms=all(r['timing']['p99'] <= protocol['timing']['p99_limit_ms'] for r in rows))
    result = dict(status='LDC_BASELINE_QUALITY_AUDIT_COMPLETE' if all(checks.values()) else 'LDC_BASELINE_AUDIT_REQUIRES_REVIEW',
        checks=checks, cases=rows, elapsed_seconds=time.monotonic()-started,
        metrics={k: describe([r['metrics'][k] for r in rows]) for k in ['velocity_relative_l2', 'pressure_relative_l2', 'mse_all']},
        speed={k: describe([r['timing'][k] for r in rows]) for k in ['p50', 'p95', 'p99']},
        boundary={side: describe([r['boundary'][side]['velocity_error_rms'] for r in rows])
                  for side in ['bottom', 'top', 'left', 'right', 'obstacle']},
        differential={k: describe([r['differentiation'][k] for r in rows]) for k in
            ['continuous_momentum_rms', 'grid_momentum_rms_at_same_points', 'ad_minus_fd_momentum_rms', 'ad_divergence_rms']},
        main_test_opened=False, new_training_performed=False, accuracy_threshold_invented=False,
        main_experiment_started=False, human_quality_review_pending=True)
    save(WORK / 'summary.json', result)
    verify()
    print(result['status'], checks, flush=True)
    return all(checks.values())


if __name__ == '__main__':
    try:
        success = run()
    except Exception:
        save(WORK / 'failure.json', dict(status='AUDIT_FAILED', error=traceback.format_exc()))
        raise
    raise SystemExit(0 if success else 1)
