"""Post-hoc development diagnostic from existing scores; no fields or weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK = ROOT / 'results/fpo_pinn_spatial_context_20261103_v21'


def main():
    results, inputs = {}, []
    for name in ['zero_spatial_control', 'learned_spatial_context']:
        paths = sorted((WORK / 'training/142777' / name / 'selected_checkpoint').glob('*/per_frame.json'))
        if len(paths) != 6:
            raise ValueError('Six completed selected-checkpoint case scores required')
        cases = []
        for p in paths:
            content = p.read_bytes()
            rows = json.loads(content)
            if len(rows) != 238 or [r['frame'] for r in rows] != list(range(4, 242)):
                raise ValueError('Unexpected frame coverage')
            cases.append(rows)
            inputs.append(dict(path=str(p), bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        comparisons = []
        for horizon in [1, 5, 10, 20, 40, 80, 238]:
            velocity = [sum(r['model']['velocity_relative_l2'] for r in a[:horizon]) / horizon for a in cases]
            pressure = [sum(r['model']['pressure_relative_l2'] for r in a[:horizon]) / horizon for a in cases]
            comparisons.append(dict(first_predicted_frames=horizon, solver_time_horizon=horizon * .05,
                mean_velocity=sum(velocity) / 6, mean_pressure=sum(pressure) / 6,
                max_case_velocity=max(velocity), max_case_pressure=max(pressure),
                field_accuracy_pass_cases=sum(u <= .1 and p <= .2 for u, p in zip(velocity, pressure))))
        results[name] = comparisons
    report = dict(status='POSTHOC_DEVELOPMENT_HORIZON_DIAGNOSTIC', job=142777,
        checkpoint_selection='Previously selected using full238-frame development validation; not reselected here',
        scope='Means over first N predictions from original prefix2,3; six equally weighted development cases',
        limitations=['Not a causal rollout with model feedback', 'Not a longer-history model',
                    'Not an independent final test or proof of an optimal horizon',
                    'Short horizon success does not establish full trajectory accuracy',
                    'V21 main job may still be completing diagnostics; this is not its final results archive'],
        source_files=inputs, results=results)
    out = WORK / 'horizon_review_20261103'
    out.mkdir(exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(out / 'summary.json'), results=results)))


if __name__ == '__main__':
    main()
