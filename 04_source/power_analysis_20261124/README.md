# Reprodukcia analýzy sily LLM
CPU analýza z archivovaných výsledkov. Žiadne tréningy ani TEST inferencia.
Aktívna analýza: power_analysis_v2.py; nezávislá kontrola: verify_power_v3.py; čitateľný výsledok: build_report.py.
Pôvodné dáta sú na cestách zaznamenaných v input_manifest.json. Cesty v zdroji patria pôvodnému Perunu; pri inom umiestnení treba nastaviť ROOT a výstupné cesty. Výstupný adresár analýzy musí byť nový, aby sa staré výsledky neprepísali.
Prostredie: Python3.12.15, NumPy2.5.4, SciPy1.18.1. CPU Slurm postup je v run_v2.sbatch a verify_v3.sbatch.
Kompaktný analysis_inputs.npz obsahuje všetkých175 finálnych modelov vyhľadávacích metód, 5hľadaní×5finálnychsemien×200TEST×3metriky. Neobsahuje váhy ani nové polia. Manifest a aritmetický audit overujú zhodu s pôvodnými modelovými súhrnmi.
Prvé CPU pokusy zastavili dve technické kontroly: interpretácia hashu účtovného záznamu a NaN v alternatívnej SciPy CDF vetve nezávislého overovača. Obe opravy sú vysvetlené v 05_history/power_analysis_20261124/technical_diagnostic.json; žiadne pôvodné vedecké výsledky neboli zmenené. Aktívna numerika má nezávislú integračnú aj Monte Carlo kontrolu.
