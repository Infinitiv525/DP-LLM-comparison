"""Read-only progress and completed capacity audit. Never open TEST or train."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
sys.path.insert(0, str(ROOT / 'test/analysis'))
from ldc24_ops_v36 import WORK, POOL, verify, read, digest

CAP = ROOT / 'results/ldc_capacity_20261105_v39'
OUT = ROOT / 'results/ldc_status_20261106_v42'


def main():
    if OUT.exists():
        raise RuntimeError('Existing audit must not be overwritten')
    verify()
    for path, expected in read(CAP / 'freeze.json')['files'].items():
        assert digest(path) == expected, path
    state_path = WORK / 'main/state.json'
    # Hash the same bytes used for this snapshot; the live queue can advance.
    state_bytes = state_path.read_bytes()
    import hashlib
    state = json.loads(state_bytes)
    tasks = state['tasks']
    counts = dict(Counter(r['status'] for r in tasks.values()))
    done_outcomes = dict(Counter(r['outcome']['status'] for r in tasks.values() if r['status'] == 'DONE'))
    queue = subprocess.run(['squeue', '-u', 'lutoke823', '-h', '-o', '%i|%j|%T|%M|%N'],
                           capture_output=True, text=True, check=True, timeout=30).stdout
    account = read(WORK / 'scheduler/143342/summary.json')
    review = read(WORK / 'main/batches/143342/review.json')
    assert account['state'] == 'COMPLETED' and account['exit_code'] == '0:0'
    assert all(review['checks'].values())
    result = read(CAP / 'result.json')
    assert result['all_three_complete'] and len(result['outcomes']) == 3
    rows = []
    for seed in [150001, 150002, 150003]:
        folder = CAP / 'runs' / str(seed)
        path = folder / 'summary.json'
        summary = read(path)
        outcome = next(r for r in result['outcomes'] if r['seed'] == seed)
        receipt = read(folder / 'capacity_receipt.json')
        assert outcome['exit_code'] == 0 and outcome['complete'] and not outcome['timed_out']
        assert digest(path) == outcome['summary_sha256']
        assert summary['status'] == 'LDC_MAIN_TRIAL_COMPLETE'
        assert summary['seed'] == seed and summary['epochs'] == 400
        assert summary['physics_updates'] > 0 and not summary['main_test_opened']
        assert summary['train_cases'] == 640 and summary['validation_cases'] == 160
        assert len(summary['case_counts']) == 640 and set(summary['case_counts'].values()) == {400}
        validation, speed = summary['selected_validation'], summary['selected_speed']
        assert not validation['failure'] and len(validation['cases']) == 160
        assert len(speed['cases']) == 160 and speed['repeats'] == 200 and speed['warmup'] == 30
        assert digest(folder / 'best.pt') == summary['checkpoint_sha256']
        assert receipt['protocol_sha256'] == digest(CAP / 'protocol.json')
        assert receipt['source_freeze_sha256'] == digest(CAP / 'freeze.json')
        assert receipt['final_speed_pass'] == speed['pass_5ms']
        rows.append(dict(seed=seed, width=384, epochs=summary['epochs'],
                         physics_updates=summary['physics_updates'],
                         selected_update=summary['selected_update'],
                         train_cases=summary['train_cases'], validation_cases=summary['validation_cases'],
                         velocity_relative_l2=validation['means']['velocity_relative_l2'],
                         pressure_relative_l2=validation['means']['pressure_relative_l2'],
                         validation_mse=validation['score'],
                         p99_ms_min=min(c['p99'] for c in speed['cases']),
                         p99_ms_max=max(c['p99'] for c in speed['cases']),
                         pass_5ms=speed['pass_5ms'],
                         training_seconds=summary['training_seconds'],
                         summary_sha256=digest(path), checkpoint_sha256=summary['checkpoint_sha256'],
                         initial_weight_sha256=summary['initial_weight_sha256']))
    assert len({r['initial_weight_sha256'] for r in rows}) == 3
    fixed = {r['task']['seed']: {'id': k, 'status': r['status']}
             for k, r in tasks.items() if r['task']['method'] == 'fixed'}
    paired_ready = all(fixed.get(seed, {}).get('status') == 'DONE' for seed in [150001, 150002, 150003])
    stamp = datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()
    report = dict(observed_at=stamp, audit='PASS', frozen_main_and_capacity_sources='PASS',
                  state_sha256=hashlib.sha256(state_bytes).hexdigest(), main_counts=counts,
                  main_done_outcomes=done_outcomes, planned_attempts=len(tasks),
                  completed_attempt_percent=100*counts.get('DONE', 0)/len(tasks),
                  by_method={m: dict(Counter(r['status'] for r in tasks.values() if r['task']['method'] == m))
                             for m in sorted({r['task']['method'] for r in tasks.values()})},
                  queue=queue.splitlines(), attention_required=state['attention_required'],
                  hold_present=(POOL / 'hold.json').exists(), completed_main_job=account,
                  completed_batch_checks={'passed': sum(review['checks'].values()),
                                          'total': len(review['checks']), 'status': review['status']},
                  collector=read(POOL / 'collector_status/143807.json'), capacity_rows=rows,
                  capacity_all_three_complete=True, fixed_final_status=fixed,
                  paired_capacity_comparison_ready=paired_ready,
                  capacity_superiority_claim=False,
                  main_test_opened=False, jobs_submitted=False, existing_training_modified=False)
    assert not report['attention_required'] and not report['hold_present']
    OUT.mkdir(parents=True)
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = ['STAV 2D PINN EXPERIMENTU — ' + stamp, '',
             f"Hlavný experiment: {counts.get('DONE', 0)} / {len(tasks)} hotových pokusov "
             f"({report['completed_attempt_percent']:.1f} % počtu pokusov; nie celej dizertácie).",
             f"Rozpracované: {counts.get('RUNNING', 0)}; čakajúce: {counts.get('PENDING', 0)}.",
             'Úloha 143342 úspešne COMPLETED, exit 0:0; všetky kontroly dávky prešli.',
             'Žiadny attention ani hold. Existujúce úlohy a automatické pokračovanie ponechané.', '',
             'Väčšia sieť: všetky tri nezávislé tréningy po 400 epochách dokončené.',
             'Nižšie sú priemerné relatívne L2 chyby na 160 VAL prípadoch; latencia je',
             'maximum per-case p99 z 200 natívnych GPU meraní pre každý VAL prípad.',
             'seed | rýchlosť L2 [%] | tlak L2 [%] | najvyššie p99 [ms] | limit 5 ms']
    for r in rows:
        lines.append(f"{r['seed']} | {100*r['velocity_relative_l2']:.4f} | "
                     f"{100*r['pressure_relative_l2']:.4f} | {r['p99_ms_max']:.6f} | "
                     f"{'SPLNENÝ' if r['pass_5ms'] else 'NESPLNENÝ'}")
    lines += ['', 'Párové porovnanie proti trom finálnym seedom malej pevnej baseline ešte nie je hotové.',
              'Nesmie sa tvrdiť, že väčšia sieť je lepšia, len podľa porovnania s jedným starším pilotom.',
              'Rozšírená sieť zostáva oddelenou kapacitnou štúdiou; pôvodný hlavný experiment sa nemení.',
              'Úloha 143807 má kapacitné tréningy hotové; ďalej zbiera účtovné doklady ostatných dávok.',
              'Po dobehnutí aktuálnej skupiny je nastavená ďalšia dávka z pôvodného plánu.',
              'TEST ostal uzavretý, nové tréningy ani úlohy tento audit nespustil.',
              'Váhy a veľké polia ostávajú na Perune. Lokálne sa prenáša iba tento TXT/JSON.']
    (OUT / 'STAV.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['observed_at', 'audit', 'main_counts',
          'completed_attempt_percent', 'capacity_all_three_complete', 'paired_capacity_comparison_ready']}))
    print('\n'.join(lines[9:17]))


if __name__ == '__main__':
    main()
