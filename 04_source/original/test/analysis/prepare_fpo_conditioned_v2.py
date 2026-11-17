"""Pinned, hashed development data only. Never downloads main-test flow fields."""
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import numpy as np
from scipy import ndimage
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from fpo_common_v2 import ROOT, WORK, PILOT, REV, save, digest, verify, coords


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(entry, target):
    """Resume only complete verified files; retain failed partials for diagnosis."""
    if target.exists():
        if target.stat().st_size == entry['size'] and digest(target) == entry['lfs']['oid']:
            return
        raise ValueError('Existing file is not the pinned file: ' + str(target))
    for attempt in range(1, 4):
        part = target.with_suffix(target.suffix + f'.attempt{attempt}.partial')
        if part.exists():
            continue
        try:
            url = f'https://huggingface.co/datasets/BGLab/FlowBench/resolve/{REV}/' + entry['path']
            with urllib.request.urlopen(url, timeout=120) as source, part.open('xb') as dest:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
            if part.stat().st_size != entry['size'] or digest(part) != entry['lfs']['oid']:
                raise ValueError('Download hash/length mismatch')
            part.replace(target)
            return
        except Exception as exc:
            save(target.with_suffix(target.suffix + f'.attempt{attempt}.failure.json'),
                 dict(error=repr(exc), attempt=attempt, unix=time.time()))
    raise RuntimeError('Download failed after three attempts: ' + str(target))


def geometry_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        mask, sdf = z['mask'], z['data']
    if mask.shape != (256, 1024) or sdf.shape != mask.shape or not np.isfinite(sdf).all():
        raise ValueError('Invalid geometry arrays')
    if not set(np.unique(mask)).issubset({0., 1.}):
        raise ValueError('Unknown mask convention')
    fluid = mask > .5
    if not .5 < fluid.mean() < 1:
        raise ValueError('Expected 1=fluid, 0=solid; no silent convention switching')
    # Do not assume that an undocumented array is a signed distance: derive it
    # from the audited binary mask. This is geometry processing, not new CFD.
    signed = (ndimage.distance_transform_edt(fluid) - ndimage.distance_transform_edt(~fluid)) / 64.
    encoded = np.stack((fluid.reshape(32, 8, 128, 8).mean((1, 3)),
                        np.clip(signed, -1, 1).reshape(32, 8, 128, 8).mean((1, 3)))).astype('float32')
    # Halfway between adjacent fluid/solid cell centres; approximate digitized
    # obstacle surface, never the four crop edges. Surface error is documented.
    points = []
    for axis in (0, 1):
        left = fluid[:-1, :] if axis == 0 else fluid[:, :-1]
        right = fluid[1:, :] if axis == 0 else fluid[:, 1:]
        y, x = np.where(left != right)
        points.append(np.column_stack((4 + (x + (.5 if axis == 0 else 1)) / 64,
                                       6 + (y + (1 if axis == 0 else .5)) / 64)))
    boundary = np.concatenate(points).astype('float32')
    interior = ndimage.binary_erosion(fluid, iterations=3, border_value=0)
    interior_yx = np.column_stack(np.where(interior)).astype('int32')
    if len(boundary) < 8 or len(interior_yx) < 1000:
        raise ValueError('Geometry has insufficient interior/obstacle samples')
    return dict(encoded=encoded, fluid=fluid, boundary=boundary, interior_yx=interior_yx)


def pool(a, fluid, rng, frames, n):
    yx = np.column_stack(np.where(fluid))
    # Equal coverage in frame indices; uniform fluid points avoid data-dependent
    # sampling that could leak reference fields into validation input.
    k = np.resize(np.asarray(frames), n).copy()
    rng.shuffle(k)
    y, x = yx[rng.integers(len(yx), size=n)].T
    values = np.asarray(a[k, y, x], dtype='float32').copy()
    for frame in np.unique(k):
        values[k == frame, 2] -= np.asarray(a[frame, :, :, 2])[fluid].mean()
    return coords(k, y, x), values, k


def run():
    verify()
    selected = json.loads((WORK / 'development_manifest.json').read_text())
    if any(r['proposed_split'] == 'test' for r in selected['scenarios']):
        raise ValueError('Main test is sealed')
    out = WORK / 'preparation' / os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    records = []
    for i, record in enumerate(selected['scenarios']):
        family, gid, re = record['family'], record['geometry_id'], record['reynolds']
        cache = WORK / 'data' / family / gid / f'Re_{re}'
        cache.mkdir(parents=True, exist_ok=True)
        if (cache / 'ready.json').exists():
            ready = json.loads((cache / 'ready.json').read_text())
            for name, checksum in ready['hashes'].items():
                if digest(name) != checksum:
                    raise ValueError('Prepared data changed: ' + name)
            records.append(ready)
            continue
        pilot = family == 'harmonics' and gid == '1' and re == 382
        if pilot:
            field = PILOT / 'data/data.npy'
            geometry_file = PILOT / 'data/input_geometry.npz'
            audit = json.loads((PILOT / 'deep_audit/audit.json').read_text())
            if digest(field) != audit['data_sha256']:
                raise ValueError('Pilot reference changed')
        else:
            parent = str(Path(record['path']).parent).replace('\\', '/')
            listing = fetch(f'https://huggingface.co/api/datasets/BGLab/FlowBench/tree/{REV}/{parent}?limit=100')
            entries = {e['path']: e for e in listing if e['type'] == 'file'}
            entry = entries[record['path']]
            if entry['lfs']['oid'] != record['lfs']['oid'] or entry['size'] != record['bytes']:
                raise ValueError('Inventory mismatch')
            packed = cache / 'reference.npz'
            download(entry, packed)
            geometry_file = cache.parent / 'input_geometry.npz'
            download(entries[parent + '/input_geometry.npz'], geometry_file)
            field = cache / 'data.npy'
            if not field.exists():
                with zipfile.ZipFile(packed) as archive:
                    if 'data.npy' not in archive.namelist():
                        raise ValueError('Missing data array')
                    partial = cache / 'data.npy.partial'
                    with archive.open('data.npy') as src, partial.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    partial.replace(field)
        arrays = geometry_arrays(geometry_file)
        np.savez_compressed(cache / 'geometry.npz', **arrays)
        a = np.load(field, mmap_mode='r', allow_pickle=False)
        if a.shape != (242, 256, 1024, 3) or a.dtype != np.float64:
            raise ValueError('Unexpected reference format')
        if not all(np.isfinite(a[k]).all() for k in range(242)):
            raise ValueError('Nonfinite reference')
        rng = np.random.default_rng(20261030 + i)
        samples = {}
        for prefix, frames, n in [('all', range(2, 242), 262144),
                                  ('pilot_train', range(2, 242, 6), 262144),
                                  ('pilot_val', range(5, 242, 6), 8192),
                                  ('initial', [2], 8192),
                                  ('validation', range(2, 242), 24576)]:
            q, y, k = pool(a, arrays['fluid'], rng, frames, n)
            samples[prefix + '_q'], samples[prefix + '_y'], samples[prefix + '_frame'] = q, y, k
        np.savez_compressed(cache / 'samples.npz', **samples)
        ready = dict(**record, cache=str(cache), field=str(field),
                     geometry=str(geometry_file), shape=list(a.shape),
                     hashes={str(p): digest(p) for p in [field, geometry_file, cache / 'geometry.npz', cache / 'samples.npz']},
                     all_finite=True, excluded_frames=[0, 1], pressure_policy='per-frame fluid mean removed from targets',
                     geometry_policy='1=fluid; derived mask distance; no crop-wall BC; digitized obstacle surface')
        save(cache / 'ready.json', ready)
        records.append(ready)
        save(out / 'progress.json', dict(completed=len(records), total=len(selected['scenarios']), records=records))
        print('PREPARED', family, gid, re, record['proposed_split'], flush=True)
    result = dict(status='DEVELOPMENT_DATA_PREPARED', records=records, elapsed_seconds=time.monotonic() - start,
                  main_test_fields_downloaded=False, revision=REV, source_freeze=digest(WORK / 'freeze.json'))
    save(out / 'summary.json', result)
    save(WORK / 'prepared.json', result)


if __name__ == '__main__':
    run()
