"""Small Git payload; checkpoint binaries remain in the streamed release asset."""
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm'))
from ldc_main_common_v33 import ROOT, WORK, read, save, digest


def run():
    export = ROOT / 'results/ldc_github_export_20261105_v38'
    index = read(export / 'latest.json')
    out = Path(index['manifest_path']).parent
    manifest = read(out / 'manifest.json')
    state = read(out / 'main_state_snapshot.json')
    files = {}
    for row in manifest['files']:
        path = Path(row['path'])
        if path.parts[0] == 'test' and path.suffix in ['.py', '.sbatch'] and 'runtime' not in path.parts:
            files['source/' + row['path']] = ROOT / path
    cap = ROOT / 'results/ldc_capacity_20261105_v39'
    for path in read(cap / 'freeze.json')['files']:
        p = Path(path)
        if str(p).startswith(str(ROOT / 'test')):
            files['source/' + str(p.relative_to(ROOT))] = p
    for p in [ROOT/'test/analysis/export_ldc_github_v38.py', Path(__file__)]:
        files['source/' + str(p.relative_to(ROOT))] = p
    files.update({
        'provenance/main_source_freeze.json': WORK / 'freeze.json',
        'provenance/snapshot_manifest.json': out / 'manifest.json',
        'provenance/snapshot_index.json': out / 'index.json',
        'protocols/main_v33.json': WORK / 'protocol.json',
        'protocols/capacity_v39.json': cap / 'protocol.json',
        'protocols/capacity_source_freeze_v39.json': cap / 'freeze.json',
        'protocols/data_split_v25.json': ROOT / 'results/ldc_pinn_prepare_20261104_v25/split.json',
        'protocols/data_preparation_v25.json': ROOT / 'results/ldc_pinn_prepare_20261104_v25/prepared.json',
        'protocols/input_source_v25.json': ROOT / 'results/ldc_pinn_prepare_20261104_v25/source_manifest.json',
        'protocols/train_normalization_v25.json': ROOT / 'results/ldc_pinn_prepare_20261104_v25/cache/normalization.json',
        'results/pilot_quality_v31.json': ROOT / 'results/ldc_baseline_quality_20261104_v31/summary.json',
        'results/pilot_scientific_review_v31.json': ROOT / 'results/ldc_baseline_quality_20261104_v31/scientific_review.json',
        'results/capacity_v37_pretraining_failure.json': ROOT / 'results/ldc_capacity_20261105_v37/result.json',
    })
    quality = ROOT / 'results/ldc_baseline_quality_20261104_v31'
    for p in (quality / 'images').glob('*.png'):
        files['images/' + p.name] = p
    metrics = [dict(task=k, **r['outcome']['history']) for k, r in state['tasks'].items() if r['status']=='DONE' and r.get('outcome',{}).get('history')]
    model_index = [r for r in manifest['files'] if r['path'].endswith('.pt')]
    readme = f'''RESEARCH ARTIFACTS — SHARED STEADY 2D FLOWBENCH PINN
Snapshot: {index['snapshot']} (interim; main experiment continues)

One shared geometry/Re-conditioned, data- and physics-informed PINN predicts
steady u/v/p on native512x512. 640 TRAIN /160 VAL /200 sealed TEST.
Own V26 architecture:134922 parameters. Main V33 compares four pinned local
LLM configurators, random search, TPE and fixed baseline;395 planned attempts.
The final TEST has NOT been evaluated. These are not final superiority claims.
Current snapshot covers {index['done_count']} completed main attempts, the
development pilot and technical integration. Active checkpoints are excluded.
Technical two-epoch integration models must not be counted as main results.

WHERE THE ACTUAL NETWORKS AND COMPLETE RESULTS ARE
The matching GitHub prerelease includes {Path(index['archive_path']).name}.
It contains own best.pt, last.pt and optimizer/RNG training_state.pt files,
full per-validation histories, proposal logs and pinned source/metadata.
Its SHA256 is {index['archive_sha256']}.
models/INDEX.json maps every checkpoint to its exact archive path and hash.
The archive stays on Perun until streamed to GitHub; it was not saved on PC.
Git stores small source, protocols, figures, consolidated metrics and indexes.
Raw FlowBench arrays and downloaded LLM weights are obtained separately from
the pinned original sources. This repository does not redistribute them.

PILOT, NOT FINAL TEST
VAL mean relative velocity L2:2.465%; pressure:8.875%.
Worst per-case p99 across160 VAL:2.909860ms on H200, inputs already on GPU.
Physical residual and wall errors remain nonzero; see scientific review.
5ms refers to full steady-field inference, not a physical time step.

SEPARATE CAPACITY STUDY
V39 changes width192->384 only (481482 parameters), fixed baseline settings,
three paired seeds150001..150003 and400epochs each. These are compared with
matching main fixed-baseline seeds, without feeding capacity results to LLMs.
The original V37 launcher failed before model initialization; its record is
retained. V39 passes audited data arguments and requires an actual smoke run.
The larger model is measured even if slower than5ms; that fails speed
feasibility and is not relabelled as success. Capacity training is separate
from this completed-only checkpoint snapshot and needs a later snapshot.

REUSE AND PROVENANCE
Verify the archive SHA256, extract using Python tarfile with filter="data",
then verify each file against manifest.json SHA256. Original ROOT paths are
/mnt/project/perun2601449/dp_llm_cfd; scripts preserve those original paths.
Torch2.12.0+cu130/FP32/H200 was used; exact per-run environment/config files
are in the archive. Restore audited TRAIN/VAL data from pinned FlowBench
sources before using the original training scripts. Keep TEST sealed.
Source dependencies are preserved in the full archive, including original
third-party license metadata. No new license over unpublished research is
granted by this snapshot. Original FlowBench/LLM licenses still apply.
'''
    bundle = out / 'repository_payload.zip'
    prefix = 'experiments/ldc_pinn/'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, prefix + name)
        archive.writestr(prefix + 'README.txt', readme)
        archive.writestr(prefix + 'results/completed_main_metrics.json', json.dumps(metrics,indent=2)+'\n')
        archive.writestr(prefix + 'models/INDEX.json', json.dumps(model_index,indent=2)+'\n')
    receipt = dict(path=str(bundle), bytes=bundle.stat().st_size, sha256=digest(bundle), files=len(files)+3,
                   checkpoint_binaries_in_payload=False, raw_target_arrays_in_payload=False)
    save(out/'repository_payload_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    run()
