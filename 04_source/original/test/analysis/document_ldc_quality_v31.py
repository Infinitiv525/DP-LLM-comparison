"""Close the reviewed audit without modifying its frozen sources or outputs."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DOC = BASE / 'dizertacia_2d_pinn'
FOLDER = DOC / 'kontrola_kvality_20261104_v31'
RECORDS = FOLDER / 'zaznamy'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run():
    s = read(RECORDS / 'summary.json')
    local = read(RECORDS / 'local_receipt_review.json')
    full = read(DOC / 'llm_konfigurator_20261104_v30/zaznamy/integration_audit.json')
    account = read(RECORDS / 'accounting_143299.json')
    extra = read(RECORDS / 'pressure_detail_accounting.json')
    assert all(s['checks'].values()) and all(local['checks'].values())
    assert account['state'] == extra['state'] == 'COMPLETED'
    # Visual review actually performed in the conversation, all eleven PNGs.
    images = sorted(p for p in RECORDS.rglob('*.png'))
    assert len(images) == 11
    scientific = dict(status='RECOMMEND_CONTROLLED_LLM_COMPARISON',
        technical_checks_passed=True, visual_review_completed=True,
        ready_for_controlled_configuration_comparison=True,
        exact_physics_or_boundary_satisfaction_established=False,
        main_test_opened=False, main_experiment_started=False,
        evidence=dict(quality_summary_sha256=hashlib.sha256((RECORDS/'summary.json').read_bytes()).hexdigest(),
                      local_receipt_review_sha256=hashlib.sha256((RECORDS/'local_receipt_review.json').read_bytes()).hexdigest(),
                      images={p.relative_to(RECORDS).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in images}),
        observations=['All 160 VAL cases reproduce previous errors; all native p99 values below 5ms.',
            'Overall flow features agree; local velocity ripples and pressure contour differences remain visible.',
            'Exact-wall velocities show finite slip; obstacle surfaces use the same half-cell approximation as training.',
            'Full-grid momentum residual remains above reference; same-point AD/FD comparisons are consistent in scale, not identical.',
            'Linear pressure plots conceal bulk variation due to corner extremes; additional symmetric-log views inspected.',
            'No new acceptance threshold fitted after seeing results; fixed checkpoint unchanged.'],
        next='Freeze the main search/retraining protocol and implement the production runner with matched budgets; two-epoch technical launcher remains separate.')
    (RECORDS/'scientific_review.json').write_text(json.dumps(scientific,indent=2)+'\n',encoding='utf-8')
    b=s['boundary']; d=s['differential']; n=local['notes']
    text=f'''VÝSLEDOK AUDITU PRED VEĽKÝM LLM TESTOM — 04.11.2026

Záver
Odporúčam prejsť ku kontrolovanému porovnaniu konfigurácií LLM.
Základný spoločný PINN je funkčný a rýchlosť je potvrdená na celej VAL.
Má merateľné nedostatky v tlaku, fyzike a na okrajoch, ktoré sa budú
ďalej uvádzať a porovnávať. Presné splnenie všetkých rovníc a stien
nebolo preukázané. Audit neurčil spätne nový prah úspešnosti presnosti.

Čo prebehlo
Uložený model z epochy 390 sa nemenil ani ďalej neučil. Overený SHA váh,
zmrazené zdroje a dátové odtlačky. Všetkých 160 VAL prípadov, pri každom
30 zahrievacích a 200 meraných predikcií celého natívneho poľa 512 × 512.
Pri každom aj steny, prekážka a porovnanie derivácií. TEST (200) uzavretý.
Výpočtové kontroly 5/5, nezávislé kontroly malých záznamov 487/487,
dve analytické kontroly derivácií prešli. Kontrola archívov: SHA a CRC.

Rýchlosť
Najvyššie p99 zo všetkých 160 prípadov: {n['maximum_p99_ms']:.6f} ms.
P99 zo všetkých 32 000 meraní dohromady: {n['pooled_p99_ms']:.6f} ms.
Najdlhšia jednotlivá predikcia: {n['maximum_single_query_ms']:.6f} ms.
Počet meraní nad 5 ms: {n['queries_over_5ms']}.
Merané na H200, FP32, vstupy na GPU. Zahrnuté spracovanie geometrie,
sieť, interpolácia, rýchlosť, tlak, tlaková kalibrácia a maska. Prenos
medzi PC/GPU a vykresľovanie mimo. Nie je to univerzálna hardvérová záruka.

Presnosť sa reprodukovala
Priemer relatívnej L2 chyby rýchlosti {s['metrics']['velocity_relative_l2']['mean']*100:.3f} %,
tlaku {s['metrics']['pressure_relative_l2']['mean']*100:.3f} %. Najhorší prípad rýchlosti
{s['metrics']['velocity_relative_l2']['maximum']*100:.3f} %, tlaku {s['metrics']['pressure_relative_l2']['maximum']*100:.3f} %.
Ide o VAL používanú na výber modelu, nie záverečný nezávislý TEST.

Steny a prekážka
RMS veľkosti chyby rýchlosti na matematickej stene, normalizované
rýchlosťou horného veka (preto percentá nie sú relatívnou chybou voči
nulovej rýchlosti stojacej steny):
horná: priemer {b['top']['mean']*100:.3f} %, najvyšší prípad {b['top']['maximum']*100:.3f} %;
dolná: priemer {b['bottom']['mean']*100:.3f} %;
ľavá: priemer {b['left']['mean']*100:.3f} %;
pravá: priemer {b['right']['mean']*100:.3f} %;
prekážka: priemer {b['obstacle']['mean']*100:.3f} %, najvyšší prípad {b['obstacle']['maximum']*100:.3f} %.
Rohy s nespojitou rýchlosťou vynechané. Povrch prekážky aproximovaný
plochami natívnej masky, rovnako ako v tréningu. Nejde o presnú hladkú
CAD geometriu. Samostatne uložené chyby voči referencii v krajných
bunkách — ich stredy neležia priamo na matematickej stene.

Tréningový výpočet a fyzika
Najvyšší relatívny rozdiel rýchlosti medzi vzorkovaným tréningovým
stencilom a natívnym výstupom: {n['maximum_stencil_relative_l2']*100:.6f} %.
Vopred daná tolerancia prešla pri všetkých prípadoch. Porovnanie
neodhalilo zásadný nesúlad medzi týmito dvoma výpočtovými cestami.
Rozdiel kontinuálnej rýchlosti (autograd) a mriežkovej rýchlosti má
priemerne {n['mean_ad_vs_grid_velocity_relative_l2']*100:.4f} % relatívne L2,
v najhoršom prípade {n['maximum_ad_vs_grid_velocity_relative_l2']*100:.4f} %.

Na rovnakých 512 náhodných vnútorných bodoch každého prípadu je
priemer momentum RMS cez autograd {d['continuous_momentum_rms']['mean']:.6g},
cez mriežkové derivácie {d['grid_momentum_rms_at_same_points']['mean']:.6g}.
RMS ich rozdielu má priemer {d['ad_minus_fd_momentum_rms']['mean']:.6g}.
Hodnoty sa nesmú zamieňať s výpočtom na celej vnútornej mriežke:
tam zostáva priemer modelu {full['baseline']['physics']['momentum_rms']['prediction']['mean']:.6g}
oproti referencii {full['baseline']['physics']['momentum_rms']['reference']['mean']:.6g}.
Riedke vzorkovanie nezaručuje zachytenie úzkych oblastí veľkého rezídua.
Nízka divergencia je podporená konštrukciou siete a sama nie je dôkaz
správnej hybnosti. Fyzikálne rezíduá musia zostať v hlavnom hodnotení.

Obrazová kontrola
Prezretých všetkých 11 obrázkov: prehľad, 5 polí u/v/p a 5 doplňujúcich
tlakových obrázkov. Prípady 341 (prvý VAL), 283 (najnižšie Re a najhorší
tlak), 424 (najvyššie Re), 396 (medián chyby rýchlosti), 956 (najhoršia
rýchlosť). Voľba podľa už známej VAL tabuľky bola zapísaná pred auditom.
Model zachytáva hlavný tvar polí. Viditeľné zostávajú lokálne zvlnenia
rýchlosti a rozdiely tlakových kontúr, najmä pri vyšších Re a prekážke.
Nejde o chýbajúci prípad či prázdny výstup. Tlakové extrémy v rohoch
prekrývajú jemnejšiu štruktúru v lineárnej škále; doplnená označená
symetrická logaritmická škála, bez orezania hodnôt alebo zmeny polí.
Referencia a predikcia zdieľajú rozsah farieb, rozdiel má vlastný rozsah.

Kde výsledok pozrieť
zaznamy/images/overview.png — prehľad presnosti, okrajov a rýchlosti.
zaznamy/images/case_956.png — najhorší prípad rýchlosti, referencia/model/rozdiel.
zaznamy/pressure_detail/pressure_283.png — najhorší tlak s čitateľnejšou škálou.
zaznamy/summary.json — všetkých 160 tabuliek a všetky jednotlivé časy.
zaznamy/scientific_review.json — záver a jeho obmedzenia.
zaznamy/local_receipt_review.json — nezávislá kontrola percentilov a údajov.
Veľké dátové polia a váhy zostali iba na Perune. Žiadne .pt/.npy/.npz
sa lokálne nekopírovali. Malé archívy spolu približne 3,19 MB.

Výpočtové úlohy a ďalší krok
143299 COMPLETED 0:0, 170 s na 1 GPU / 4 CPU = 0,04722 GPU h,
0,18889 core h; následné účtovanie 143300 skončilo. Doplňujúca tlaková
grafika 143307 COMPLETED 0:0, 31 s na 1 CPU, bez GPU.
Momentálne nič naše nebeží. Hlavný veľký LLM test ešte nezačal.
Nasleduje uzamknutie hlavného rozsahu opakovaní a pravidiel a produkčný
spúšťač: 4 LLM proti random/TPE a pevnému základu, rovnaký rozpočet,
nové tréningy od začiatku a viac seedov. V26 je vývojový pilot;
finálny pevný základ sa znovu učí cez rovnakú cestu ako ostatné metódy.
Technický dvojepochový spúšťač sa nevydáva za produkčný experiment.
'''
    (FOLDER/'VYSLEDOK.txt').write_text(text,encoding='utf-8')
    readme=FOLDER/'PRECITAJ_MA.txt'
    old=readme.read_text(encoding='utf-8')
    marker='AUDIT DOKONČENÝ A POSÚDENÝ — 04.11.2026'
    if marker not in old:
        readme.write_text(marker+'\n\n'
            'Všetkých 160 VAL prešlo. Najvyššie p99 2,910 ms; 32 000 meraní,\n'
            'žiadne nad 5 ms. Pôvodné chyby sa reprodukovali. Steny a fyzika\n'
            'majú vyčíslené nedostatky. Záver: vhodné východisko na kontrolované\n'
            'LLM porovnanie, bez tvrdenia presného splnenia fyziky.\n'
            'Podrobnosti a odkazy na obrázky: VYSLEDOK.txt. TEST ostal uzavretý.\n\n'
            'PÔVODNÝ PLÁN AUDITU\n\n'+old,encoding='utf-8')
    index=DOC/'PRECITAJ_MA.txt'
    old=index.read_text(encoding='utf-8')
    if marker not in old:
        index.write_text(marker+'\n\n'
            'Podrobný audit uloženého V26 je hotový: rýchlosť, steny, AD/FD\n'
            'a obrazové porovnanie. Najvyššie p99 zo 160 VAL 2,910 ms.\n'
            'Záver a 11 obrázkov: kontrola_kvality_20261104_v31/VYSLEDOK.txt.\n'
            'Odporúčané prejsť k veľkému kontrolovanému porovnaniu; rezíduá\n'
            'fyziky zostávajú priznané. TEST uzavretý, hlavná séria ešte nezačala.\n\n'
            'HISTORICKÉ ZÁZNAMY\n\n'+old,encoding='utf-8')
    log=BASE.parent/'dokumentacia.txt'
    marker='110. Ukončenie a obrazové posúdenie auditu V31'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n04.11.2026\n\n'
                '143299 a afterany 143300 COMPLETED. 160 VAL, checkpoint nezmenený,\n'
                '5/5 výpočtových kontrol, reprodukcia predchádzajúcich chýb.\n'
                '32 000 surových meraní; maximum p99 medzi prípadmi 2,909860 ms,\n'
                'pooled p99 2,899616 ms, najdlhšie jedno meranie 3,829088 ms,\n'
                'žiadne nad 5 ms. Nezávislé prehratie percentilov lokálne 487/487.\n'
                'SHA/CRC malého archívu 1903532 B overené:\n'
                '4531e4610b39d9da0109c9b9c2febdf9fa04d6f3aebdd5d9f55d70203e82e158.\n'
                'BC priemer RMS / rýchlosť veka: top 2,168 %, bottom 0,373 %,\n'
                'left 2,016 %, right 1,929 %, obstacle 1,543 %. Steny nie sú presne\n'
                'splnené; maska sa nevydáva za dôkaz no-slip. Povrch half-cell.\n'
                'Maximum sampled-stencil/native relatívne L2 1,13928e-5; AD/native\n'
                'velocity priemer 0,050689 %, max 0,196185 %. AD momentum RMS\n'
                'priemer na 512 bodoch 0,248518, FD na rovnakých 0,252570, RMS ich\n'
                'rozdielu priemer 0,034774. Nezamieňať so všetkými vnútornými bodmi\n'
                'FD 0,800666 oproti referencii 0,100106; riedke vzorky môžu minúť\n'
                'úzke oblasti veľkého rezídua. Žiadny nový prah presnosti.\n\n'
                'Obrazovo prezretý overview a všetkých 5 u/v/p prípadov. Hlavná\n'
                'štruktúra sedí, viditeľné lokálne zvlnenia a tlakové rozdiely.\n'
                'Lineárnu tlakovú škálu dominujú rohové extrémy; dodatočný samostatný\n'
                'render_ldc_pressure_detail_v31.py a skript 248, CPU 143307 held,\n'
                'registrácia zdrojových SHA pred release. Pôvodný zmrazený renderer\n'
                'a obrázky zachované. Nových 5 tlakových PNG so symetrickou log\n'
                'škálou, bez orezania alebo zmien údajov; všetky vizuálne prezreté.\n'
                '1285450 B, SHA/CRC overené:\n'
                'e3d6a3771d79f1adb71d224c4325e2f6e187d1ee682f3cc88477116d37270ce6.\n'
                'Všetkých 11 PNG a malé JSON/TXT lokálne, žiadne váhy/polia.\n\n'
                'Záver scientific_review.json: odporučené kontrolované porovnanie\n'
                'konfigurácií, bez tvrdenia presného splnenia fyziky alebo okrajov.\n'
                'Vývojový V26 nezmenený, hlavný baseline sa musí znovu trénovať\n'
                'tou istou cestou ako ostatné metódy. Hlavná séria zatiaľ nezačala,\n'
                'produkčný spúšťač a presný hlavný rozsah ešte neuzamknuté.\n'
                'Skutočná alokácia 143299: 170 GPU s / 680 core s, doplnkový\n'
                'CPU render 143307: 31 core s. Účtovanie z afterany a živého scontrol;\n'
                'malé CPU účtovanie mimo týchto súčtov. Aktuálne nič naše nebeží.\n'
                'Nový čitateľný súhrn kontrola_kvality_20261104_v31/VYSLEDOK.txt,\n'
                'aktualizované indexy, všetko nové pod test/.\n')
    print('QUALITY_REVIEW_DOCUMENTED', len(images), 'images')


if __name__=='__main__':
    run()
