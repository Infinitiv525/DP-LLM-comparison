"""Additional pressure colour scale; original audit figures remain unchanged."""
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from ldc_quality_common_v31 import WORK, read, save, digest, verify


def run():
    verify()
    out = WORK / 'pressure_detail'
    out.mkdir(exist_ok=False)
    protocol = read(WORK / 'protocol.json')
    receipt = dict(source_sha256=digest(__file__), audit_freeze_sha256=digest(WORK / 'freeze.json'),
                   reason='Corner pressure extremes hide bulk variations in a linear colour scale.',
                   fields_changed=False, clipping=False, cases=[])
    for sample in protocol['examples']:
        with np.load(WORK / 'examples' / f'case_{sample}.npz', allow_pickle=False) as data:
            fluid = data['fluid']
            reference = data['reference'][..., 2]
            prediction = data['prediction'][..., 2]
            re = float(data['reynolds'])
        difference = prediction-reference
        combined = np.concatenate((reference[fluid], prediction[fluid]))
        limit = max(float(np.abs(combined).max()), 1e-7)
        transition = max(float(np.percentile(np.abs(combined), 99))*.05, 1e-7)
        error_limit = max(float(np.abs(difference[fluid]).max()), 1e-7)
        error_transition = max(float(np.percentile(np.abs(difference[fluid]), 99))*.05, 1e-7)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
        for ax, value, title, vmax, lin in zip(axes,
            [reference, prediction, difference], ['Reference p', 'Shared PINN p', 'PINN - reference p'],
            [limit, limit, error_limit], [transition, transition, error_transition]):
            ax.set_facecolor('#8b8b8b')
            im = ax.imshow(np.ma.masked_where(~fluid, value), origin='lower', extent=[0, 1, 0, 1],
                interpolation='nearest', cmap='RdBu_r', norm=SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax))
            ax.set(title=title, xlabel='x / L', ylabel='y / L')
            fig.colorbar(im, ax=ax, shrink=.85, ticks=[-vmax, -lin, 0, lin, vmax], format='%.3g')
        fig.suptitle(f'Steady LDC | case {sample} | Re={re:g} | pressure with symmetric log colour scale\n'
            'Zero fluid mean; same scale for reference/PINN, own error scale; no clipping or field changes.')
        fig.savefig(out / f'pressure_{sample}.png', dpi=120)
        plt.close(fig)
        receipt['cases'].append(dict(sample=int(sample), limit=limit, linear_transition=transition,
            error_limit=error_limit, error_linear_transition=error_transition))
    save(out / 'render_receipt.json', receipt)
    archive = WORK / 'pressure_detail.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, p.relative_to(WORK))
    save(WORK / 'pressure_detail_archive_receipt.json', dict(bytes=archive.stat().st_size, sha256=digest(archive)))
    verify()
    print('PRESSURE_DETAIL_DONE', archive.stat().st_size, flush=True)


if __name__ == '__main__':
    run()
