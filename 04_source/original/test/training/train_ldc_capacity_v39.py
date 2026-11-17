"""Separate width-only capacity study; reuse the immutable main training loop."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm'))
import numpy as np
import torch
import train_ldc_main_v33 as trainer
from ldc_main_common_v33 import ROOT, WORK, read, save, digest, verify
from ldc_main_contract_v33 import training_config
from ldc_store_v26 import Store, query

CAP = ROOT / 'results/ldc_capacity_20261105_v39'


def readiness(model, store, out):
    """Identical timing scope; retain slower models to measure capacity tradeoff."""
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    model.eval()
    with torch.inference_mode():
        for index in [store.train_ids[0], store.validation_ids[0]]:
            known = store.known(index)
            for _ in range(30):
                query(model, known, store.qgrid)
            torch.cuda.synchronize()
            times = []
            for _ in range(200):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                value = query(model, known, store.qgrid)
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b))
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite capacity readiness prediction')
            rows.append(dict(sample=index, **{f'p{p}': float(np.percentile(times, p)) for p in [50, 95, 99]}))
    result = dict(cases=rows, pass_5ms=all(r['p99'] <= 5 for r in rows),
                  fullfield=[512, 512], inputs_gpu=True,
                  includes='Same native input pooling, encoder, MLP, interpolation, curl, gauge, mask as V33',
                  excludes='Host transfer, rendering, static coordinate setup',
                  speed_is_reported_without_training_gate=True)
    save(out / 'summary.json', result)
    model.train()
    return result


def run(seed, smoke_only=False):
    verify()
    protocol = read(CAP / 'protocol.json')
    if seed not in protocol['seeds'] and not (smoke_only and seed == 320101):
        raise ValueError('Undeclared seed')
    for path, expected in read(CAP / 'freeze.json')['files'].items():
        if digest(path) != expected:
            raise ValueError('Capacity source changed: ' + path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name():
        raise ValueError('One assigned H200 required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    main = read(WORK / 'protocol.json')
    config = training_config(main['base_model'], protocol['parameters'], 'integration' if smoke_only else 'main', seed)
    config['width'] = protocol['larger_width']
    out = CAP / 'technical_smoke' if smoke_only else CAP / 'runs' / str(seed)
    if out.exists():
        raise ValueError('Capacity run exists; no silent rerun or overwrite')
    # These adapters affect this new process only; frozen source files stay intact.
    trainer.benchmark = readiness
    original_diagnostics = trainer.selected_diagnostics
    trainer.selected_diagnostics = lambda model, store, path, phase: original_diagnostics(model, store, path, 'main')
    prepared = read(main['prepared_path'])
    split = read(main['split_path'])['indices']
    if prepared['status'] != 'LDC_DATA_READY' or not prepared['physics_ready']:
        raise ValueError('Audited physical scale required')
    store = Store(prepared, split, False)
    summary = trainer.train(config, store, out, phase='capacity_smoke' if smoke_only else 'capacity')
    save(out / 'capacity_receipt.json', dict(seed=seed, width=config['width'], epochs=config['epochs'],
         protocol_sha256=digest(CAP / 'protocol.json'), source_freeze_sha256=digest(CAP / 'freeze.json'),
         main_test_opened=False, final_speed_pass=summary['selected_speed']['pass_5ms'],
         note='Independent capacity study, never a main-search candidate or LLM feedback.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    run(args.seed, args.smoke_only)
