"""Scientific comparison plots from audit examples; arrays stay on Perun."""
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ldc_quality_common_v31 import WORK, read, save, digest, verify


def run():
    verify()
    protocol = read(WORK / 'protocol.json')
    summary = read(WORK / 'summary.json')
    images = WORK / 'images'
    images.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})
    for sample, roles in protocol['examples'].items():
        with np.load(WORK / 'examples' / f'case_{sample}.npz', allow_pickle=False) as data:
            prediction = data['prediction']
            reference = data['reference']
            fluid = data['fluid']
            re = float(data['reynolds'])
        row = next(r for r in summary['cases'] if r['sample'] == int(sample))
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
        labels = ['u', 'v', 'p (zero fluid mean)']
        for channel, label in enumerate(labels):
            limit = max(float(np.max(np.abs(reference[..., channel][fluid]))),
                        float(np.max(np.abs(prediction[..., channel][fluid]))), 1e-8)
            difference = prediction[..., channel] - reference[..., channel]
            error_limit = max(float(np.max(np.abs(difference[fluid]))), 1e-8)
            for column, (value, title, scale) in enumerate([
                (reference[..., channel], 'Reference', limit),
                (prediction[..., channel], 'Shared PINN', limit),
                (difference, 'PINN - reference', error_limit)]):
                ax = axes[channel, column]
                ax.set_facecolor('#8b8b8b')
                masked = np.ma.masked_where(~fluid, value)
                im = ax.imshow(masked, origin='lower', extent=[0, 1, 0, 1],
                               interpolation='nearest', cmap='RdBu_r', vmin=-scale, vmax=scale)
                ax.set_title(f'{label}: {title}')
                ax.set_xlabel('x / L')
                ax.set_ylabel('y / L')
                fig.colorbar(im, ax=ax, shrink=.88, format='%.3g')
        fig.suptitle(f'FlowBench steady LDC | case {sample} | Re={re:g}\n'
                     f'{", ".join(roles)} | velocity L2 {row["metrics"]["velocity_relative_l2"]*100:.2f}% | '
                     f'pressure L2 {row["metrics"]["pressure_relative_l2"]*100:.2f}%\n'
                     'Grey: known solid. Reference and prediction share row scale; error has its own scale.')
        fig.savefig(images / f'case_{sample}.png')
        plt.close(fig)
    # Across-case diagnostics, not artificial time evolution.
    rows = summary['cases']
    re = np.array([r['reynolds'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for key, label in [('velocity_relative_l2', 'Velocity'), ('pressure_relative_l2', 'Pressure')]:
        axes[0].scatter(re, [100*r['metrics'][key] for r in rows], s=13, label=label, alpha=.7)
    axes[0].set(xlabel='Re', ylabel='Relative L2 error (%)', title='160 validation cases')
    axes[0].legend()
    for side in ['bottom', 'top', 'left', 'right', 'obstacle']:
        axes[1].scatter(re, [r['boundary'][side]['velocity_error_rms'] for r in rows], s=10, label=side, alpha=.65)
    axes[1].set(xlabel='Re', ylabel='Velocity error RMS / lid speed', title='Exact/approximated surface BC')
    axes[1].legend(fontsize=7)
    axes[2].scatter(re, [r['timing']['p99'] for r in rows], s=12)
    axes[2].axhline(5, color='red', linestyle='--', label='5 ms limit')
    axes[2].set(xlabel='Re', ylabel='Full-field GPU p99 (ms)', title='200 queries per validation case')
    axes[2].legend()
    fig.savefig(images / 'overview.png')
    plt.close(fig)
    lines = ['AUDIT ULOŽENÉHO SPOLOČNÉHO PINN — 04.11.2026', summary['status'], '',
             'Bez ďalšieho učenia, všetkých 160 VAL prípadov. TEST zostal uzavretý.',
             'Každý prípad: 30 zahrievacích + 200 natívnych meraní celého poľa.',
             f'Najvyššie p99 medzi prípadmi: {summary["speed"]["p99"]["maximum"]:.4f} ms.', '',
             'Chyba rýchlosti na stenách, v jednotkách rýchlosti horného veka:']
    for side, metric in summary['boundary'].items():
        lines.append(f'{side}: priemer RMS {metric["mean"]:.6g}, najvyššie RMS {metric["maximum"]:.6g}')
    lines += ['', 'Kontroly:'] + [f'{k}: {v}' for k, v in summary['checks'].items()]
    lines += ['', 'Číselné údaje: summary.json; raw_ms obsahuje všetky merania.',
              'Obrázky: images/overview.png a case_*.png. Zobrazené u/v/p, referencia, model a rozdiel.',
              'Obrázky ukazujú ustálené polia, nie časový vývoj.',
              'Nízka divergencia neznamená automaticky správnu hybnosť. Žiadny nový prah presnosti nebol zavedený.',
              'AD a FD sa porovnávajú na rovnakých 512 vnútorných bodoch každého prípadu.',
              'Povrch prekážky je aproximovaný plochami masky ako pri tréningu.',
              'Obrazové posúdenie a záver o vhodnosti pre hlavné porovnanie sa zaznamenajú samostatne.']
    (WORK / 'VYSLEDOK.txt').write_text('\n'.join(lines) + '\n')
    archive = WORK / 'small_quality_audit.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(WORK.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.txt', '.png', '.log'} and path.stat().st_size < 12*1024*1024:
                z.write(path, path.relative_to(WORK))
    save(WORK / 'archive_receipt.json', dict(bytes=archive.stat().st_size, sha256=digest(archive)))
    save(WORK / 'render_receipt.json', dict(images=[p.name for p in sorted(images.glob('*.png'))],
                                          main_test_opened=False, arrays_exported=False))
    verify()
    print('RENDERED_AND_PACKED', archive.stat().st_size, flush=True)


if __name__ == '__main__':
    run()
