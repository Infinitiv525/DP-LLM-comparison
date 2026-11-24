"""Readable report built only after arithmetic/provenance audit."""
import json,math,hashlib
from pathlib import Path
R=Path('/mnt/project/perun2601449/dp_llm_cfd/results/llm_power_20261124_v2')
S=Path('/mnt/project/perun2601449/dp_llm_cfd/test/llm_power_20261124_v1')
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(R/'independent_verification.json');assert audit['status']=='INDEPENDENT_PROVENANCE_VARIANCE_POWER_AND_COST_AUDIT_PASS'
summary=read(R/'summary.json');ds=read(R/'sample_size_designs.json');costs=read(R/'costs_by_search_method.json')
def choose(effect,power=.9,mode='fresh_independent_blocks_point_proxy',design='five_LLM_vs_TPE_MSE'):
 return next(x for x in ds if x['design']==design and x['target_relative_difference_vs_TPE']==effect and x['target_power']==power and x['sd_scenario']==mode)
def number(x):return f'{x:,.0f}'.replace(',',' ')
lines=[
'ANALÝZA SILY LLM EXPERIMENTU – VÝSLEDKY (24. 11. 2026)','',
'ČO VLASTNE POČÍTAME',
'Chceme vedieť, koľko nových nezávislých hľadaní treba na zachytenie hypotetického rozdielu chyby oproti TPE. Neplánujeme počet tak, aby LLM muselo vyhrať. Väčšia vzorka môže potvrdiť aj horší výsledok LLM.',
'Päť rodín: Qwen3-8B, Ministral3-8B, Phi4-mini, Granite3.3-8B a Llama-3.1-8B. Hlavná metrika: pôvodná all-cell TEST MSE.',
'JEDNO HĽADANIE = 8 kandidátnych tréningov PINN + 5 tréningov vybranej konfigurácie + TEST týchto 5 modelov. LLM sa neučí. Každý PINN tréning: 400 epoch cez všetkých 640 TRAIN prípadov.',
'JEDEN PÁROVANÝ BLOK = jedno nové hľadanie každej z piatich LLM a jedno nové hľadanie TPE. TPE platíme raz a použijeme vo všetkých piatich kontrastoch. Celkom 6 hľadaní, 78 tréningov PINN na blok.','',
'HLAVNÁ PLÁNOVACIA TABUĽKA',
'Všetky počty nižšie sú CELÉ NOVÉ nezávislé potvrdenie; pôvodných päť hľadaní na metódu slúži ako pilot. Starých päť sa od týchto počtov neodpočítava.',
'Pevných 200 TEST prípadov, obojstranný párový t test, rodinná alpha 0,05 pre 5 porovnaní (konzervatívne alpha 0,01 na porovnanie), 90 % sila na jednotlivý kontrast.',
'Bodová plánovacia aproximácia variability zahŕňa neistotu výberu konfigurácie aj finálnych tréningových semien. Nie je to istota budúceho výsledku.',
'Rozdiel voči priemernej MSE TPE | Nové hľadania na KAŽDÚ z 6 metód | Hľadaní spolu | PINN tréningov | Očakávané rezervované GPU-h | Návrh limitu + TEST rezerva GPU-h'
]
for effect in [.05,.1,.2]:
 d=choose(effect);p=d['balanced_all_new']
 lines.append(f"{effect:.0%} | {d['balanced_search_blocks']} | {p['new_independent_searches_total']} | {p['total_PINN_trainings']} | {number(p['expected_reserved_total_GPU_h'])} | {number(p['proposed_6h30_per_training_job_cap_plus_TEST_reserve_GPU_h'])}")
lines+=['','Pri 80 % sile namiesto 90 %:']
for effect in [.05,.1,.2]:
 d=choose(effect,.8);p=d['balanced_all_new']
 lines.append(f"Rozdiel {effect:.0%}: {d['balanced_search_blocks']} nových hľadaní na každú metódu, {p['new_independent_searches_total']} spolu, asi {number(p['expected_reserved_total_GPU_h'])} GPU-h.")
d=choose(.1);p=d['balanced_all_new'];q=choose(.1,mode='fresh_independent_blocks_upper95_SD');u=q['balanced_all_new']
lines+=['','NEISTOTA TOHTO ODHADU',
'Máme iba päť hľadaní. Navyše sa vo všetkých pôvodných hľadaniach používala rovnaká sada piatich finálnych tréningových semien. Výpočet to zohľadňuje kríženým modelom; 25 finálnych checkpointov sa nevydáva za 25 nezávislých hľadaní.',
f"Pre rozdiel 10 % a silu 90 % vychádza bodová aproximácia na {d['balanced_search_blocks']} nových hľadaní na každú zo šiestich metód. Opatrnejší výpočet s hornou hranicou variability dáva až {q['balanced_search_blocks']} na metódu, teda {u['new_independent_searches_total']} hľadaní a asi {number(u['expected_reserved_total_GPU_h'])} GPU-h.",
f"Násobok plánovacieho SD v opatrnejšom variante: {summary['variance_upper95_multiplier']:.6f}. Je odvodený z dvoch 97,5 % horných normálnych rozptylových hraníc, spoločné pokrytie aspoň 95 % pre jeden kontrast. NIE je to simultánna 95 % záruka cez celú rodinu, ani zaručená 90 % sila.",
'Model predpokladá normálne nové blokové rozdiely a normálne krížené zložky v pilote; diskrétny výber konfigurácií môže tento predpoklad porušiť. Simulovaná kontrola šikmého šumu je v numerical_verification.json.',
'Prirodzené opakovanie fixného štartu štyrmi rodinami neposkytuje dôkaz, že sa pri ďalších semenách nikdy nevyberie iná konfigurácia. Päť pozorovaní neodhalí všetky zriedkavé zlyhania/výhry.',
'90 % je plánovaná sila jednotlivého porovnania, nie pravdepodobnosť, že naraz vyjdú všetky porovnania. Štatistická významnosť nie je ekvivalencia ani praktická dôležitosť.','',
'POZOR NA VEĽMI MALÉ N',
'Pri rodine piatich kontrastov je alpha 0,01 na porovnanie. Exaktný obojstranný sign-flip test má najmenšie možné p = 2/2^N; už na rozlíšenie p <= 0,01 treba aspoň 8 párov, navyše pri predpoklade symetrie pre tento test.',
'Preto bodový plán 6 blokov pre rozdiel 20 % stojí na parametrickom t modeli. Nedáva možnosť takejto exaktnej kontroly na alpha 0,01. Ak ju požadujeme, minimálne 8 nových blokov znamená 48 hľadaní, 624 tréningov PINN a približne '+number(8*sum(costs[m]['expected_reserved_GPU_h'] for m in ['qwen3_8b','ministral3_8b','phi4_mini','granite33_8b','Llama31_8B','tpe']))+' GPU-h; ani táto hranica sama osebe nezaručuje 90 % silu exaktného testu.',
'',
'MÔŽEME TO ZLACNIŤ?',
'1. Využiť pôvodných päť a zachovať presne pôvodnú sadu finálnych semien. Je to lacnejšie exploratívne doplnenie so záverom podmieneným týmto panelom, nie nové nezávislé potvrdenie cez ľubovoľné semená. Potrebuje iný, užší výklad výsledku:']
for effect in [.05,.1,.2]:
 c=choose(effect,mode='old_final_seed_panel_conditional');p=c['exploratory_add_to_old_five_same_final_seed_panel']
 lines.append(f"  Rozdiel {effect:.0%}, sila 90 %: cieľ spolu {c['balanced_search_blocks']} hľadaní/metódu, pridať {max(0,c['balanced_search_blocks']-5)}/metódu = {p['new_independent_searches_total']} nových hľadaní; asi {number(p['expected_reserved_total_GPU_h'])} GPU-h. Iba starý panel, modelový odhad.")
target=choose(.1,design='Ministral_vs_TPE_MSE_exploratory_targeted');tp=target['balanced_all_new']
lines += [
f"2. Samostatné nové porovnanie Ministral proti TPE, vybrané po starých výsledkoch: pri 10 % rozdiele a 90 % sile vychádza {target['balanced_search_blocks']} nových hľadaní na každú z týchto dvoch metód, {tp['new_independent_searches_total']} spolu, asi {number(tp['expected_reserved_total_GPU_h'])} GPU-h. Nie je to náhrada päťčlenného porovnania ani spätne primárna hypotéza.",
'3. Zmeniť počet kandidátov, finálnych opakovaní alebo prompt by zmenilo experiment; aktuálne N a cenu na taký protokol automaticky neprenášame.',
'Presné individuálne počty pre každú LLM a možnosť zdieľanej dlhšej série TPE sú v sample_size_designs.json (unequal_all_new_shared_TPE). Vyvážený variant je organizačne jednoduchší.','',
'AK BY SME CHCELI SÚČASNE VYHODNOCOVAŤ TRI CHYBOVÉ METRIKY AKO PRIMÁRNE',
'To je dodatočná 15-členná rodina; nemení pôvodnú hlavnú MSE analýzu:']
for effect in [.05,.1,.2]:
 d3=choose(effect,design='five_LLM_vs_TPE_three_metrics');p3=d3['balanced_all_new']
 lines.append(f"Rozdiel {effect:.0%}, sila 90 %: {d3['balanced_search_blocks']} nových hľadaní/metódu, {p3['new_independent_searches_total']} spolu, asi {number(p3['expected_reserved_total_GPU_h'])} GPU-h.")
lines+=['','Z ČOHO JE CENA',
'Metóda | Aktívne GPU-h na 8 kandidátov + 5 refitov | Z toho generovanie LLM | Aktívny TEST 5 modelov | Projekcia rezervovaných GPU-h vrátane réžie']
for method in ['qwen3_8b','ministral3_8b','phi4_mini','granite33_8b','Llama31_8B','tpe','random']:
 c=costs[method]
 lines.append(f"{method} | {c['mean_active_search_plus_5_refits_GPU_h']:.3f} | {c['mean_proposal_only_GPU_h']:.4f} | {c['mean_active_TEST_5_models_GPU_h']:.3f} | {c['expected_reserved_GPU_h']:.3f}")
cal=read(R/'observed_reservation_calibration.json')
lines += [
f"Šesť pôvodných spoločných GPU úloh rezervovalo {cal['old395_reserved_GPU_h']:.3f} GPU-h; súčet aktívnych 395 úloh bol {cal['old395_active_task_GPU_h']:.3f} GPU-h. Pomer {cal['old_pool_overhead_factor']:.6f} kalibruje nečinné konce/prevádzkovú réžiu v odhade.",
'Na nové individuálne úlohy navrhujeme limit 6,5 GPU-h na jeden tréning, teda 84,5 GPU-h na hľadanie plus 0,5 GPU-h rezerva TEST. Spolu 85 GPU-h na celé hľadanie; čas sa účtuje podľa skutočného behu, nie automaticky celý limit.',
'Ide o plánovací model z nameraných časov. Nie je to presný budúci účet portálu. Náklady sa neporovnávajú s neovereným aktuálnym zostatkom kvóty. Trvanie pri súbehu nie je totožné so súčtom GPU hodín.','',
'ČO ODPORÚČAM',
'Nevyhlasovať univerzálny počet povinných tréningov pre dizertáciu. Najprv pomenovať najmenší rozdiel, ktorý chceme spoľahlivo zachytiť. Pôvodný päťhľadaniový experiment je použiteľný s priznanou neistotou; analýza sily z neho sama nerobí neplatný experiment.',
'Ak chceme nové nezávislé potvrdenie všetkých piatich LLM, hlavná tabuľka je relevantný plán. Bodový variant je rozpočtový odhad pod predpokladmi; opatrnejší variant ukazuje, prečo nemožno z piatich hľadaní sľúbiť presný počet.',
'Ak celý plán nemá opodstatnenie, rozumnejšie je ponechať transparentne pôvodné závery alebo vopred ohraničiť nové cielené porovnanie než vyrábať ďalšie rovnaké checkpointy a počítať ich ako nezávislé hľadania.',
'Pevný TEST benchmark a 85 známych geometrií ostávajú; viac hľadaní neoprávňuje tvrdenia o úplne nových geometriách. Diagnostická hranica geometrickej neistoty je v geometry_power_ceiling_diagnostic.json.','',
'KONTROLY A ULOŽENIE',
f"Nezávislý audit: {audit['input_files_verified']} vstupných súborov, {audit['training_records']} hlavných tréningových záznamov, {audit['raw_final_models_reproduced']} finálnych modelov, {audit['sample_size_rows_verified']} výpočtov N a {audit['design_rows_verified']} súhrnných variantov.",
'Číselné údaje, zdroje a SHA ostávajú na Perune a sú publikované priamo Perun→GitHub. Na PC nevznikli nové vedecké dátové súbory.',
'Žiadne nové GPU tréningy LLM/PINN neboli spustené. Bežiace potvrdenie architektúr sa nemenilo.','',
'METODICKÉ ZDROJE',
'https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.solve_power.html',
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html',
'https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc232.htm',
'https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/sdconfli.htm',
'Vzorce pre nový päťsemenný blok a konzervatívnu kombináciu rozptylov sú odvodené a opísané v protokol.txt; ide o našu plánovaciu aproximáciu, nie tvrdenie autorov FlowBench.'
]
(R/'VYSLEDKY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
readme=['# Plánovanie ďalších nezávislých hľadaní LLM',
'',
'[Ľudsky čitateľné výsledky](VYSLEDKY.txt) · [Číselné varianty](sample_size_designs.json) · [Nezávislá kontrola](independent_verification.json)',
'',
'CPU analýza z pôvodných dát; žiadne nové GPU tréningy. Päť LLM proti spoločnej novej kontrole TPE. Rozlišujeme celé nové nezávislé potvrdenie a užšie exploratívne doplnenie so starým panelom tréningových semien.',
'',
'Jeden nový nezávislý blok používa päť čerstvých finálnych tréningových semien: párovaných medzi metódami, odlišných medzi blokmi. Závery sú podmienené tým istým TEST benchmarkom.',
'',
'[Protokol](../../03_methodology/power_analysis_20261124/protokol.txt) · [Reprodukovateľný zdroj](../../04_source/power_analysis_20261124)',
'',
'Výpočet N závisí od hypotetického efektu a odhadu variability z iba piatich hľadaní. Horná rozptylová citlivosť je podmienená normálnym kríženým modelom; nie je garanciou budúcej sily.',
'']
(R/'README.md').write_text('\n'.join(readme),encoding='utf-8')
save(R/'recommendation.json',dict(primary_point_10percent_90power=choose(.1),
 primary_cautious_10percent_90power=choose(.1,mode='fresh_independent_blocks_upper95_SD'),
 primary_point_20percent_90power=choose(.2),
 targeted_Ministral_10percent_90power=target,no_new_GPU_training_launched=True))
save(R/'OUTPUT_MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in R.iterdir() if p.is_file() and p.name!='OUTPUT_MANIFEST.json'})
print(json.dumps(dict(status='READABLE_REPORT_COMPLETE',recommendation=read(R/'recommendation.json')),ensure_ascii=False))
