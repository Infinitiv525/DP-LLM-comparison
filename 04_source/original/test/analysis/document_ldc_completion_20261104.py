"""Verify small transferred receipts and update local human-readable records."""
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]
DOC = BASE / 'dizertacia_2d_pinn'
LLM = DOC / 'llm_konfigurator_20261104_v30'
PINN = DOC / 'spolocny_ustaleny_pinn_20261104_v26'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_archive(directory, name, receipt):
    path = directory / name
    record = read(directory / receipt)
    assert path.stat().st_size == record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    archive = zipfile.ZipFile(path)
    assert archive.testzip() is None
    assert all(Path(n).suffix in {'.json', '.txt', '.log'} for n in archive.namelist())
    return archive


def write(path, content):
    path.write_text(content.rstrip() + '\n', encoding='utf-8')


def prepend_once(path, content, marker):
    previous = path.read_text(encoding='utf-8')
    if marker not in previous:
        write(path, content.rstrip() + '\n\nHISTORICKÉ ZÁZNAMY — stav v čase ich zápisu\n\n' + previous)


def run():
    audit = read(LLM / 'zaznamy/integration_audit.json')
    assert audit['status'] == 'INTEGRATION_RECEIPTS_PASS' and all(audit['checks'].values())
    source = BASE / 'analysis/audit_ldc_integration_20261104.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit['source_sha256']
    z = verified_archive(LLM / 'zaznamy', 'small_preflight.zip', 'archive_receipt.json')
    b = verified_archive(PINN / 'zaznamy', 'baseline_numeric_receipts.zip', 'baseline_archive_receipt.json')
    review = json.loads(z.read('review.json'))
    assert review['status'] == 'FOUR_LDC_LLM_PREFLIGHTS_COMPLETE' and all(review['checks'].values())
    for name in ['review.json', 'device_assignment.json']:
        (LLM / 'zaznamy' / name).write_bytes(z.read(name))
    (LLM / 'zaznamy/remote_VYSLEDOK.txt').write_bytes(z.read('VYSLEDOK.txt'))
    for relative, filename in [
        ('training/143115/summary.json', 'vysledok_143115.json'),
        ('training/143115/selected_validation/summary.json', 'validacia_160_pripadov_143115.json'),
        ('training/143115/selected_speed/summary.json', 'rychlost_143115.json'),
        ('scheduler/143115/summary.json', 'accounting_143115.json')]:
        (PINN / 'zaznamy' / filename).write_bytes(b.read(relative))

    p = audit['proposals']
    gen = [r['elapsed_seconds'] for r in p]
    process = [r['process_wall_seconds'] for r in p]
    gpu = [r['peak_gpu_bytes'] / 2**30 for r in p]
    speed = [r['p99_max_ms'] for r in p]
    trial = [r['training_seconds'] for r in audit['technical_trials']]
    duplicate = sum(r['duplicates_history'] for r in p)
    llm_result = f'''DOKONČENIE TECHNICKEJ SKÚŠKY — 04.11.2026

Body 2 a 3 sú technicky overené. Úloha 143245 skončila COMPLETED 0:0,
účtovanie 143246 tiež dobehlo. Na Perune teraz z tohto chatu nič nebeží.
Qwen3-8B, Ministral-3-8B, Phi-4-mini a Granite-3.3-8B každý urobil dva
návrhy, každý návrh sa skutočne použil v novom krátkom tréningu PINN.
Druhý návrh dostal skutočné validačné výsledky prvého. Všetkých osem
odpovedí bolo platných na prvý pokus, nebola potrebná opravná odpoveď.

Pripravené pravidlá
Konfigurátor vyberá rýchlosť učenia, váhu tlaku, váhu fyziky, váhu okrajov
a odklad zapnutia fyziky. Architektúra a dátový i výpočtový rozpočet sú
rovnaké. Automatická schéma stráži zápis, čísla vyberá LLM; žiadna ručná
náhrada konfigurácie. Schéma nevynucuje kvalitné uvažovanie. Stručné
deklarované dôvody, očakávanie a riziko sa uchovávajú so surovou odpoveďou.
Niektoré vety končia na limite 120 znakov; neprepisovali sa za model.

Čo reálne zbehlo
Jeden spoločný základ, osem tréningov podľa LLM a dva kontrolné tréningy
pre random a TPE: spolu 11 nových tréningov. Každý má dve celé epochy,
640 TRAIN / 160 VAL, rovnaké počiatočné váhy a body. Je to skúška zapojenia,
nie nezávislé štatistické opakovanie ani hotové hlavné porovnanie kvality.
Hlavný TEST (200 prípadov) zostal uzavretý.

Vstavaná kontrola prešla 70/70 bodmi. Dodatočný audit skutočných súborov
prešiel 136/136 bodmi: porovnal aj navrhnuté parametre s config.json
tréningu, presné indexy dát, nadväznosť spätnej väzby, zdroje, aktívnu
fyziku, vzorkovanie a časové záznamy. Tento audit nenačítal váhy ani polia.

Pozorovanie z návrhov
{duplicate} z 8 návrhov zopakovalo spoločné základné nastavenie. Qwen, Phi a
Granite ho ponechali v oboch kolách; Ministral zvolil dve odlišné
konfigurácie. Aj opakované návrhy absolvovali nový tréning a spotrebovali
plný pokus. Rovnaké výsledky opakovaných konfigurácií sú očakávané pri
rovnakom seede. Z týchto dvoch epoch nemožno určiť najlepší LLM.
Text zdôvodnenia nie vždy presne opisuje zmenu čísel; rozhodujú uložené
parametre a skutočné merania. Odkaz na spätnú väzbu je riadený schémou.

Namerané časy a pamäť
Návrh vrátane načítania a schémy: {min(gen):.1f}–{max(gen):.1f} s;
celý samostatný proces návrhu: {min(process):.1f}–{max(process):.1f} s.
Jeden dvojepochový tréning s validáciami a záverečným meraním:
{min(trial):.1f}–{max(trial):.1f} s, mimo štartu procesu a prípravy dát.
Vzorkované maximum pamäte procesu LLM na GPU: {min(gpu):.2f}–{max(gpu):.2f} GiB.
P99 celej predikcie 512 × 512 v ôsmich LLM tréningoch:
{min(speed):.3f}–{max(speed):.3f} ms. Ide o H200, vstupy na GPU, dve merané
vstupné geometrie, 200 opakovaní každej; prenos do PC a vykreslenie mimo.

Skutočná alokácia úspešnej úlohy: 648 s na 4 GPU = 0,7200 GPU h,
16 CPU = 2,8800 core h. Zahŕňa aj rezervované nevyužité GPU pri spoločnom
základe a dobiehaní workerov. Predošlé V27–V29 zostávajú zachované:
spolu s V30 ide o 2,4167 GPU h a 9,6667 core h hlavných výpočtových úloh.
Toto nie je spotreba celého projektu; drobné CPU účtovacie úlohy sú mimo.

Kde sú údaje
zaznamy/integration_audit.json — čísla, konfigurácie, dôvody, kontroly.
zaznamy/review.json — výsledok vstavanej kontroly.
zaznamy/small_preflight.zip — všetky malé JSON/TXT/log záznamy vrátane
promptov, surových odpovedí, prostriedkov a validačných tabuliek.
SHA a CRC archívu boli po prenose overené. Archív má približne 2,46 MB.
Veľké polia a váhy zostali na Perune:
/mnt/project/perun2601449/dp_llm_cfd/results/ldc_llm_preflight_20261104_v30

Čo nasleduje
Hlavný pilot 143115 už tiež skončil; jeho číselný audit je v susednom
spolocny_ustaleny_pinn_20261104_v26/VYSLEDOK.txt. Pred hromadným
porovnaním ostáva podrobná kontrola jeho okrajov, kontinuálnej a mriežkovej
fyziky a obrazových rozdielov. Potom uzamknúť hlavný rozsah hľadaní,
tréningový rozpočet a seedy a pripraviť produkčný spúšťač. Súčasný spúšťač
zámerne povoľuje iba dvojepochovú technickú skúšku; stovky tréningov sa
ním nespustili. Hlavné LLM porovnávanie zatiaľ nezačalo.
'''
    write(LLM / 'VYSLEDOK.txt', llm_result)
    metrics = audit['baseline']['metrics']
    physics = audit['baseline']['physics']
    pinn_result = f'''SPOLOČNÝ USTÁLENÝ PINN — VÝSLEDOK PILOTA — 04.11.2026

Úloha 143115 skončila COMPLETED 0:0. Jedna spoločná sieť absolvovala
400 epoch na všetkých 640 tréningových prípadoch, 64 000 aktualizácií.
Na všetkých 160 validačných prípadoch sa podľa vopred určenej MSE vybral
checkpoint z aktualizácie 62 400, teda z epochy 390. Posledná epocha sa
automaticky nevydáva za najlepší model. TEST (200 prípadov) je uzavretý.

Presnosť vybraného modelu na VAL
Priemerná relatívna L2 chyba rýchlosti: {metrics['velocity_relative_l2']['mean']*100:.3f} %.
Priemerná relatívna L2 chyba tlaku: {metrics['pressure_relative_l2']['mean']*100:.3f} %.
95. percentil chýb medzi prípadmi: rýchlosť {metrics['velocity_relative_l2']['p95']*100:.3f} %,
tlak {metrics['pressure_relative_l2']['p95']*100:.3f} %.
Najhorší prípad: rýchlosť {metrics['velocity_relative_l2']['maximum']*100:.3f} %,
tlak {metrics['pressure_relative_l2']['maximum']*100:.3f} %.
Validačné skóre MSE u/v/p na celom poli: {metrics['mse_all']['mean']:.12g}.
Chyby pri prekážke sú vyššie: priemer rýchlosti
{audit['baseline']['near_obstacle']['velocity_relative_l2']['mean']*100:.3f} % a tlaku
{audit['baseline']['near_obstacle']['pressure_relative_l2']['mean']*100:.3f} %.
Nejde o finálny nezávislý test: VAL slúžila na výber checkpointu.

Rýchlosť
P99 celého poľa na H200: 2,895 a 2,893 ms, limit 5 ms splnený na dvoch
meraných vstupoch. Každý 30 zahrievacích a 200 meraných opakovaní.
Zahrnuté spracovanie geometrie aj výpočet rýchlosti, tlaku, masky a
tlakovej kalibrácie. Vstupy sú už na GPU; prenos a vykresľovanie mimo.
Nejde o dôkaz p99 na všetkých prípadoch a ľubovoľnom hardvéri.

Kontrola fyziky
Priemerné mriežkové momentum RMS modelu:
{physics['momentum_rms']['prediction']['mean']:.6g}; referencia {physics['momentum_rms']['reference']['mean']:.6g}.
Priemerné normalizované momentum rezíduum modelu:
{physics['momentum_relative']['prediction']['mean']:.6g}; referencia {physics['momentum_relative']['reference']['mean']:.6g}.
Divergencia modelu je približne {physics['divergence_rms']['prediction']['mean']:.3g}; nízka
hodnota je podporená konštrukciou rýchlosti z prúdovej funkcie a sama
nedokazuje správnosť rovníc hybnosti. Tie model ešte nespĺňa na úrovni
referenčných dát. Presnosť a rýchlosť vyzerajú použiteľne ako východisko
porovnávania, ale nevyhlasujeme automaticky úplnú fyzikálnu zhodu.
Ďalší audit má preveriť okraje, kontinuálny výpočet proti mriežkovému
a obrázky predikcie/referencie. Model ani prahy sa spätne neupravovali.

Spotreba a uchovanie
Skutočný pridelený čas 13 175 s na 1 GPU: 3,6597 GPU h;
4 CPU: 14,6389 core h. Obsahuje krátku úvodnú skúšku a režijný čas.
Lokálne iba číselné JSON a overený baseline_numeric_receipts.zip;
veľké váhy, optimalizátor a dátové polia zostávajú na Perune:
/mnt/project/perun2601449/dp_llm_cfd/results/ldc_shared_pinn_20261104_v26
Základné súbory v zaznamy/: vysledok_143115.json,
validacia_160_pripadov_143115.json, rychlost_143115.json,
accounting_143115.json. Archív navyše obsahuje históriu a všetky VAL tabuľky.
'''
    write(PINN / 'VYSLEDOK.txt', pinn_result)
    marker = 'DOKONČENÉ A OVERENÉ — 04.11.2026'
    prepend_once(LLM / 'PRECITAJ_MA.txt', marker + '\n\n'
        'Všetky 4 lokálne LLM dokončili po 2 nadväzujúce návrhy a reálne\n'
        'krátke tréningy. Zbehlo aj random/TPE. Kontroly 70/70 a 136/136.\n'
        'Podrobné výsledky, obmedzenia a časy: VYSLEDOK.txt. TEST uzavretý.\n'
        'Nie je to ešte hlavné opakované porovnanie kvality.', marker)
    prepend_once(PINN / 'PRECITAJ_MA.txt', marker + '\n\n'
        '400 epoch je hotových. Vybraný model: VAL rýchlosť 2,465 %,\n'
        'tlak 8,875 %, opakované p99 najviac 2,895 ms na H200.\n'
        'Fyzikálne rezíduum hybnosti je vyššie než referencia.\n'
        'Výsledky a obmedzenia: VYSLEDOK.txt. TEST zostal uzavretý.', marker)
    prepend_once(DOC / 'PRECITAJ_MA.txt', marker + '\n\n'
        'Spoločný ustálený PINN 143115 aj skúška štyroch LLM 143245 skončili.\n'
        'Momentálne žiadna naša úloha nebeží. LLM konfigurátor technicky\n'
        'overený, 11 krátkych tréningov, kontrola návrh/skutočný config.\n'
        'Pilot: VAL rýchlosť 2,465 %, tlak 8,875 %, p99 2,895 ms; fyzikálna\n'
        'odchýlka ešte vyššia než referencia. TEST uzavretý, hlavná séria LLM\n'
        'nezačala. Čítaj llm_konfigurator_20261104_v30/VYSLEDOK.txt a\n'
        'spolocny_ustaleny_pinn_20261104_v26/VYSLEDOK.txt.', marker)
    log = BASE.parent / 'dokumentacia.txt'
    section = '108. Dokončenie technických skúšok LLM a číselný audit spoločného pilota'
    previous = log.read_text(encoding='utf-8')
    if section not in previous:
        with log.open('a', encoding='utf-8') as f:
            f.write('\n\n' + section + '\n04.11.2026\n\n'
                'Po prerušení výpadkom prúdu overené SSH, front a ukončenia úloh.\n'
                '143245 COMPLETED 0:0, 648 s, 4 H200 / 16 CPU; 143246 zachytil\n'
                '2592 GPU s a 10368 core s. Všetky štyri rodiny dokončili dve\n'
                'sekvenčné kolá, všetkých 8 návrhov platných bez opravy. Skutočné\n'
                'nové tréningy plus spoločný základ/random/TPE: 11, každý 2 epochy\n'
                '640 TRAIN / 160 VAL. Vstavaný review 70/70. Šesť návrhov opakuje\n'
                'základ; duplicity plne vykonané a účtované. Žiadny záver o prevahe\n'
                'LLM z tejto skúšky. Dôvody niekedy nepresné alebo na limite dĺžky;\n'
                'originály zachované. Schéma riadi formát/evidence-ID, nie úsudok.\n\n'
                'Pridaný read-only audit_ldc_integration_20261104.py, AST úspešné.\n'
                'Na prihlasovacom uzle iba malé JSON a odtlačky, bez načítania\n'
                'váh/polí alebo TEST. 136/136 kontrol vrátane presnej zhody configu\n'
                's návrhom, indexov TRAIN/VAL, vzoriek, štartov, aktívnej fyziky,\n'
                'promptov, schém, reálnej následnosti a procesných časov. Zdroj SHA\n'
                'je v results/ldc_integration_audit_20261104/integration_audit.json.\n'
                'Žiadny zmrazený zdroj alebo protokol V25/V26/V30 sa nemenil.\n\n'
                '143115 COMPLETED 0:0, 13175 s, 1 H200 / 4 CPU. Všetkých 640\n'
                'prípadov navštívených 400-krát, 64000 aktualizácií. Vybraná epocha\n'
                '390 (62400) podľa zmrazenej VAL MSE 0,000416598904397. Priemer\n'
                'relatívnej L2 chyby rýchlosti 2,465000 %, tlaku 8,875105 %.\n'
                'Záverečné p99 2,895330/2,892970 ms na dvoch vstupoch. Mean momentum\n'
                'RMS 0,800666 oproti 0,100106 referencie; normalizované rezíduum\n'
                '0,085599 oproti 0,015985. Divergencia 7,654e-7 podporená curl\n'
                'konštrukciou, nie dôkaz úplnej fyzikálnej zhody. Geometrie sa medzi\n'
                'splitmi opakujú; nejde o nezávislý TEST ani unseen-geometry dôkaz.\n'
                'Pred masovým porovnaním ostáva podrobný audit okrajov, AD/FD\n'
                'a obrazových rozdielov, následne hlavný zmrazený protokol/spúšťač.\n\n'
                'Malé archívy prenesené byte-SCP; po prenose overené SHA256 a ZIP CRC.\n'
                'V30 small_preflight.zip: 2461763 B, SHA256\n'
                '26a4db96add28830af61b73e00e1123d4977b34ec9fe777faf02dff63d757ac2.\n'
                'Baseline SHA256: f305f51e482f08e0f4ed322bdcbbf44d9c534ac537c4e5b23d42f7caab1e235b.\n'
                'Lokálne iba čísla, logy, protokoly, kód; žiadne .pt/.npy. Nové\n'
                'súbory pod test/, hlavný dokumentačný súbor doplnený týmto záznamom.\n'
                'V30 spotreba 0,72 GPU h / 2,88 core h. V27–V30 spolu 2,4167 GPU h\n'
                '/ 9,6667 core h hlavných výpočtových úloh, vrátane neúspechov.\n'
                'V26 3,6597 GPU h / 14,6389 core h. Nejde o celkovú projektovú\n'
                'spotrebu; účtovacie CPU úlohy mimo týchto súčtov. Žiadna nová\n'
                'GPU úloha nespustená, hlavná opakovaná séria LLM ešte nezačala.\n'
                'Nové ľudské súhrny VYSLEDOK.txt pri V26 a V30, aktualizované indexy.\n')
    print('DOCUMENTED; archives verified; no weights or arrays copied locally')


if __name__ == '__main__':
    run()
