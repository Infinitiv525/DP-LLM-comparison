"""Immutable main protocol and phase-local atomic receipts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK = ROOT / 'results/llama_extension_20261119_v1'
for name in ['training', 'analysis', 'llm']:
    sys.path.insert(0, str(ROOT / 'test' / name))
sys.path.insert(0, str(ROOT / 'test/runtime/ldc_schema_v30/site'))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def phase_path(phase):
    if phase not in ['integration', 'main']:
        raise ValueError('Unknown phase')
    return WORK / phase


def verify():
    freeze = read(WORK / 'freeze.json')
    for path, expected in freeze['files'].items():
        if digest(path) != expected:
            raise ValueError('Frozen source/metadata changed: ' + path)
    for key in ['cache_fingerprints', 'model_fingerprints']:
        for path, expected in freeze[key].items():
            if fingerprint(path) != expected:
                raise ValueError('Audited input changed: ' + path)
