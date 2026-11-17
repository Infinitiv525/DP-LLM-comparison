"""Read-only receipt audit; never load arrays, model weights or TEST targets."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
sys.path.insert(0, str(ROOT / 'test/llm'))
sys.path.insert(0, str(ROOT / 'test/training'))
import ldc_llm_common_v30 as llm
import ldc_common_v26 as baseline
from ldc_contract_v30 import canonical, messages, parse_reply, reply_schema, training_config

OUT = ROOT / 'results/ldc_integration_audit_20261104'


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(values, .95), maximum=max(values))


def run():
    # verify() hashes only the frozen small import closure and checks large-file
    # fingerprints. Full weights/data hashes already ran on compute nodes.
    llm.verify()
    baseline.verify()
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = llm.read(llm.WORK / 'protocol.json')
    split = llm.read(protocol['split_path'])['indices']
    train_ids, val_ids, test_ids = (set(split[k]) for k in ['train', 'validation', 'test'])
    checks = dict(frozen_sources_and_fingerprints=True,
                  disjoint_case_ids=len(train_ids | val_ids | test_ids) == 1000
                  and not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids))
    trial_rows = []
    proposal_rows = []
    common = llm.read(llm.WORK / 'common/summary.json')

    def check_trial(directory, parameters, identifier):
        config = llm.read(directory / 'config.json')
        summary = llm.read(directory / 'summary.json')
        validation = llm.read(directory / 'selected_validation/summary.json')
        process_path = directory.parent / (directory.name + '_process.json')
        checks[identifier + ':actual_config_matches_proposal'] = config == training_config(
            protocol['base_model'], parameters, 2, protocol['training_seed'])
        checks[identifier + ':exact_train_coverage'] = (
            {int(i) for i in summary['case_counts']} == train_ids
            and set(summary['case_counts'].values()) == {2})
        checks[identifier + ':exact_validation_coverage'] = (
            len(validation['cases']) == 160
            and {r['sample'] for r in validation['cases']} == val_ids)
        checks[identifier + ':same_initialization_and_sampling'] = (
            summary['initial_weight_sha256'] == common['initial_weight_sha256']
            and summary['sample_sha256'] == common['sample_sha256'])
        checks[identifier + ':real_physics_and_native_speed'] = (
            summary['physics_updates'] > 0 and config['physics_weight'] > 0
            and summary['selected_speed']['fullfield'] == [512, 512]
            and summary['selected_speed']['pass_5ms'])
        checks[identifier + ':closed_test_and_real_score'] = (
            not summary['main_test_opened'] and not validation['main_test_opened']
            and summary['selected_validation']['score'] == validation['score']
            and math.isclose(validation['score'], statistics.mean(
                r['mse_all'] for r in validation['cases']), rel_tol=1e-12))
        if process_path.exists():
            process = llm.read(process_path)
            checks[identifier + ':successful_timed_process'] = (
                process['exit_code'] == 0 and not process['timed_out']
                and process['wall_seconds'] >= summary['training_seconds'])
        trial_rows.append(dict(id=identifier, config_sha256=llm.digest(directory / 'config.json'),
                               summary_sha256=llm.digest(directory / 'summary.json'),
                               physics_updates=summary['physics_updates'],
                               score=validation['score'], training_seconds=summary['training_seconds']))
        return summary

    check_trial(llm.WORK / 'common', protocol['common_start'], 'common_0')
    for spec in protocol['models']:
        label = spec['label']
        directory = llm.WORK / 'workers' / label
        worker = llm.read(directory / 'summary.json')
        checks[label + ':two_rounds_completed'] = (
            worker['status'] == 'LDC_LLM_TWO_ROUND_PREFLIGHT_COMPLETE' and len(worker['rounds']) == 2)
        for row in worker['rounds']:
            number = row['round']
            identifier = f'{label}_{number}'
            feedback_path = directory / f'feedback_{number}.json'
            feedback = llm.read(feedback_path)
            proposal = llm.read(directory / f'proposal_{number}/summary.json')
            parsed, errors, _ = parse_reply(proposal['attempts'][-1]['raw_output'], feedback)
            checks[identifier + ':unchanged_valid_model_output'] = (
                not errors and parsed == proposal['configuration'] == row['configuration'])
            checks[identifier + ':original_prompt_and_real_feedback'] = (
                proposal['attempts'][0]['messages'] == messages(feedback)
                and proposal['feedback_sha256'] == llm.digest(feedback_path)
                and proposal['initial_messages_sha256'] == hashlib.sha256(
                    canonical(messages(feedback)).encode()).hexdigest())
            checks[identifier + ':identical_schema_and_runtime_policy'] = (
                proposal['syntax_constrained'] and proposal['schema'] == reply_schema(feedback)
                and proposal['schema_sha256'] == hashlib.sha256(
                    canonical(reply_schema(feedback)).encode()).hexdigest()
                and proposal['packages'] == protocol['llm_runtime'] and proposal['model'] == spec)
            checks[identifier + ':successful_timed_proposal'] = (
                row['proposal_process']['exit_code'] == 0
                and not row['proposal_process']['timed_out']
                and row['proposal_process']['wall_seconds'] >= proposal['elapsed_seconds'])
            summary = check_trial(directory / f'trial_{number}', parsed['parameters'], identifier)
            checks[identifier + ':measured_feedback_not_invented'] = (
                row['real_feedback']['score'] == summary['selected_validation']['score']
                and row['trial_summary_sha256'] == llm.digest(directory / f'trial_{number}/summary.json'))
            if number == 2:
                checks[label + ':round_two_receives_round_one_measurement'] = (
                    feedback['history'][-1] == worker['rounds'][0]['real_feedback'])
            proposal_rows.append(dict(model=label, round=number, parameters=parsed['parameters'],
                duplicates_history=any(parsed['parameters'] == h['parameters'] for h in feedback['history']),
                attempts=len(proposal['attempts']), reason=parsed['reason'],
                expected_effect=parsed['expected_effect'], risk=parsed['risk'],
                generation_seconds=sum(a['generation_seconds'] for a in proposal['attempts']),
                load_seconds=proposal['load_seconds'], elapsed_seconds=proposal['elapsed_seconds'],
                process_wall_seconds=row['proposal_process']['wall_seconds'],
                cpu_seconds=proposal['cpu_seconds'], peak_gpu_bytes=proposal['peak_sampled_nvml_gpu_bytes'],
                peak_rss_bytes=proposal['peak_sampled_rss_bytes'],
                p99_max_ms=max(c['p99'] for c in summary['selected_speed']['cases'])))
        for control in worker['controls']:
            summary = check_trial(directory / 'control_trial', control['feedback']['parameters'], control['method'])
            checks[control['method'] + ':real_control_feedback'] = (
                control['feedback']['score'] == summary['selected_validation']['score'])
    checks['eleven_fresh_technical_trials'] = len(trial_rows) == 11

    main_dir = baseline.WORK / 'training/143115'
    main = llm.read(main_dir / 'summary.json')
    config = llm.read(main_dir / 'config.json')
    validation = llm.read(main_dir / 'selected_validation/summary.json')
    history = llm.read(main_dir / 'history.json')
    checks['baseline:400_exact_train_epochs'] = (
        config['epochs'] == 400 and len(main['case_counts']) == 640
        and {int(i) for i in main['case_counts']} == train_ids
        and set(main['case_counts'].values()) == {400} and history[-1]['update'] == 64000)
    checks['baseline:selection_matches_frozen_metric'] = (
        main['selected_update'] == min(history, key=lambda h: h['validation']['score'])['update']
        and main['selected_validation']['score'] == min(h['validation']['score'] for h in history))
    checks['baseline:exact_validation_and_closed_test'] = (
        len(validation['cases']) == 160 and {r['sample'] for r in validation['cases']} == val_ids
        and not main['main_test_opened'] and main['shared_weights'])
    rows = validation['cases']
    metrics = {k: describe([r[k] for r in rows]) for k in
               ['velocity_relative_l2', 'pressure_relative_l2', 'mse_all', 'relative_l2_all']}
    checks['baseline:reported_metrics_recomputed'] = all(math.isclose(
        metrics[k]['mean'], validation['means'][k], rel_tol=1e-12) for k in metrics)
    checks['baseline:native_speed_passes'] = main['selected_speed']['pass_5ms']
    physics = {k: dict(prediction=describe([r['physics'][k] for r in rows]),
                       reference=describe([r['reference_physics'][k] for r in rows]))
               for k in ['momentum_rms', 'momentum_relative', 'divergence_rms']}
    near = {k: describe([r['near_obstacle_errors'][k] for r in rows])
            for k in ['velocity_relative_l2', 'pressure_relative_l2']}
    accounting = {}
    for work, job in [(llm.WORK, '143245'), (baseline.WORK, '143115')]:
        accounting[job] = llm.read(work / 'scheduler' / job / 'summary.json')
        checks[job + ':completed_and_accounted'] = accounting[job]['state'] == 'COMPLETED'
    report = dict(status='INTEGRATION_RECEIPTS_PASS' if all(checks.values()) else 'REVIEW_REQUIRED',
        checks=checks, source_sha256=llm.digest(__file__),
        freeze_sha256=dict(v26=llm.digest(baseline.WORK / 'freeze.json'), v30=llm.digest(llm.WORK / 'freeze.json')),
        technical_trials=trial_rows, proposals=proposal_rows,
        baseline=dict(selected_update=main['selected_update'], selected_epoch=main['selected_update'] / 160,
                      metrics=metrics, physics=physics, near_obstacle=near, speed=main['selected_speed']),
        accounting=accounting, main_test_opened=False, main_experiment_started=False,
        limitations=['Two epochs test integration, not LLM efficacy or converged model ranking.',
                     'Six of eight LLM proposals repeat baseline; duplicate attempts are charged.',
                     'Schema fixes format and latest evidence ID, not semantic reasoning quality.',
                     'Baseline errors are on VAL used for checkpoint selection, not independent TEST.',
                     'Shared geometry across partitions; no unseen-geometry performance claim.',
                     'Physics momentum residual exceeds reference floor; divergence is structurally small.',
                     'Repeated speed measurements cover two inputs on H200; transfers/rendering excluded.',
                     'No new acceptance thresholds or full experiment protocol introduced by this audit.'])
    llm.save(OUT / 'integration_audit.json', report)
    # Small numeric baseline receipt bundle only. No weights/arrays are eligible.
    archive = OUT / 'baseline_numeric_receipts.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        paths = [baseline.WORK / p for p in ['protocol.json', 'freeze.json', 'submission.json',
                 'scheduler/143115/query.json', 'scheduler/143115/summary.json']]
        paths += [p for p in main_dir.rglob('*.json') if p.stat().st_size < 8 * 1024 * 1024]
        for path in sorted(paths):
            z.write(path, path.relative_to(baseline.WORK))
    llm.save(OUT / 'baseline_archive_receipt.json', dict(bytes=archive.stat().st_size, sha256=llm.digest(archive)))
    print(report['status'], sum(checks.values()), len(checks), flush=True)
    print('BASELINE', json.dumps(metrics), flush=True)
    return all(checks.values())


if __name__ == '__main__':
    raise SystemExit(0 if run() else 1)
