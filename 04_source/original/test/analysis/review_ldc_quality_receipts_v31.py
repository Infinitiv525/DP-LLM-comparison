"""Local independent replay of timing percentiles and small audit receipts."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE / 'dizertacia_2d_pinn/kontrola_kvality_20261104_v31'


def percentile(values, fraction):
    values = sorted(values)
    p = (len(values)-1)*fraction
    return values[math.floor(p)] + (values[math.ceil(p)]-values[math.floor(p)])*(p-math.floor(p))


def run():
    records = FOLDER / 'zaznamy'
    receipt = json.loads((records / 'archive_receipt.json').read_text(encoding='utf-8'))
    archive = records / 'small_quality_audit.zip'
    assert archive.stat().st_size == receipt['bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            parts = Path(name).parts
            assert not Path(name).is_absolute() and '..' not in parts
            assert Path(name).suffix in {'.json', '.txt', '.log', '.png'}
            target = records / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    result = json.loads((records / 'summary.json').read_text(encoding='utf-8'))
    protocol = json.loads((records / 'protocol.json').read_text(encoding='utf-8'))
    checks = dict(archive_size_sha_and_crc=True, exactly_160_unique_cases=len(result['cases']) == 160
                  and len({r['sample'] for r in result['cases']}) == 160,
                  original_checkpoint_unchanged=result['checks']['checkpoint_unchanged'],
                  test_closed=not result['main_test_opened'] and not result['new_training_performed'])
    all_times = []
    for row in result['cases']:
        times = row['timing']['raw_ms']
        assert len(times) == protocol['timing']['repeats']
        assert all(math.isfinite(t) and t > 0 for t in times)
        all_times.extend(times)
        for key, fraction in [('p50', .5), ('p95', .95), ('p99', .99)]:
            checks[f'case_{row["sample"]}_{key}_replayed'] = math.isclose(
                row['timing'][key], percentile(times, fraction), abs_tol=1e-10)
    checks['exactly_32000_measured_queries'] = len(all_times) == 32000
    checks['all_p99_within_declared_limit'] = all(
        r['timing']['p99'] <= protocol['timing']['p99_limit_ms'] for r in result['cases'])
    checks['all_prior_compute_checks_passed'] = all(result['checks'].values())
    notes = dict(
        maximum_p99_ms=max(r['timing']['p99'] for r in result['cases']),
        pooled_p99_ms=percentile(all_times, .99), maximum_single_query_ms=max(all_times),
        queries_over_5ms=sum(t > 5 for t in all_times),
        mean_ad_vs_grid_velocity_relative_l2=statistics.mean(
            r['differentiation']['continuous_velocity_vs_native']['relative_l2'] for r in result['cases']),
        maximum_ad_vs_grid_velocity_relative_l2=max(
            r['differentiation']['continuous_velocity_vs_native']['relative_l2'] for r in result['cases']),
        maximum_stencil_relative_l2=max(
            r['differentiation']['sampled_stencil_vs_native']['relative_velocity_l2'] for r in result['cases']),
        boundary=result['boundary'], differential=result['differential'],
        image_roles=protocol['examples'],
        image_review_pending=True,
        note='Technical consistency and speed do not imply exact satisfaction of BC or NS equations. No accuracy threshold inferred from this audit.')
    report = dict(status='LOCAL_RECEIPTS_PASS' if all(checks.values()) else 'REVIEW_REQUIRED', checks=checks,
                  notes=notes, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (records / 'local_receipt_review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['status'], sum(checks.values()), len(checks))
    print(json.dumps(notes, indent=2))


if __name__ == '__main__':
    run()
