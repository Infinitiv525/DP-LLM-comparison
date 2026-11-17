"""Separate immutable audit of the existing shared baseline; no retraining."""
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
for folder in ['training', 'analysis']:
    sys.path.insert(0, str(ROOT / 'test' / folder))
from ldc_common_v26 import digest, fingerprint, save

WORK = ROOT / 'results/ldc_baseline_quality_20261104_v31'
PARENT = ROOT / 'results/ldc_shared_pinn_20261104_v26'


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    from ldc_common_v26 import verify as verify_parent
    verify_parent()
    freeze = read(WORK / 'freeze.json')
    for path, expected in freeze['files'].items():
        if digest(path) != expected:
            raise ValueError('Audit source changed: ' + path)
    if fingerprint(freeze['checkpoint']) != freeze['checkpoint_fingerprint']:
        raise ValueError('Selected baseline checkpoint changed')
