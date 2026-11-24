# Porovnanie LLM pri konfigurácii PINN

## Úplné polia všetkých 155 PINN z experimentu s LLM

[Kompletné TEST polia, váhy, vstupy a referencie](01_results/polia_LLM/README.md) sú už doplnené a overené.

## Úplný prehľad dostupných dát

**[Katalóg výsledkov, váh a numerických polí](01_results/DOSTUPNE_DATA.md)** uvádza všetky publikované balíky, priame odkazy a aj výnimky z archivácie. Veľké súbory sú v Releases a nevyžadujú stiahnutie celého repozitára.

Qwen3 8B, Ministral3 8B, Phi4 mini a Granite3.3 8B navrhovali nastavenie jednej spoločnej PINN. Porovnávali sa s náhodným hľadaním, TPE a pevnou konfiguráciou. LLM sa netrénovali; trénovali sa siete PINN s 134 922 parametrami.

Hotových je **395 hlavných tréningov a finálny TEST 155 sietí na 200 prípadoch**. Každá vyhľadávacia metóda mala päť hľadaní, zvolená konfigurácia päť záverečných tréningov. Počet nezávislých hľadaní sa nezamieňa s počtom modelov alebo časovacích opakovaní.

TPE malo najnižšiu primárnu TEST MSE. Výsledok nie je všeobecným tvrdením, že LLM nemôžu pomôcť. Úplné vyhodnotenie vrátane obmedzení je v [záverečnom reporte](01_results/final/UZAVRETIE.txt), [tabuľke metód](01_results/final/test_method_metrics.csv) a [31 000 prípadových riadkoch](01_results/final/test_case_metrics.csv).

Používame ustálené 2D prúdenie v dutine s pohybujúcim sa horným vekom, FlowBench harmonics 512×512, 640 TRAIN / 160 VAL / 200 TEST. Vstup je geometria a Reynoldsovo číslo, výstup celé pole u/v/p. Model nepočíta časový priebeh. Všetky dvojice model/prípad splnili p99 ≤ 5 ms; štyri jednotlivé merania limit prekročili. Ide o GPU meranie s pripravenými vstupmi, bez prenosu do PC a kreslenia. TRAIN a TEST zdieľajú geometrie.

## Kde začať

| Priečinok | Čo v ňom nájdeš |
|---|---|
| [01_results](01_results) | Hotové porovnanie, grafy a číselné tabuľky |
| [02_models](02_models) | Index sietí a odkazy na skutočné váhy |
| [03_methodology](03_methodology) | Konfigurácie, metriky a podmienky experimentu |
| [04_source](04_source) | Vedecký zdrojový kód |
| [05_history](05_history) | Oddelená história pilotov a neúspešných pokusov |

## Modely a podrobné záznamy

[Stiahnuť modely a experimentálne záznamy](https://github.com/Infinitiv525/DP-LLM-comparison/releases/tag/research-artifacts-v1). Archívy sú prílohy Release; bežný `git clone` ich nesťahuje. [Index modelov](02_models/model_index.csv) určuje príslušný diel archívu, pôvod a SHA256 každého súboru. [Manifest](02_models/ARTIFACT_MANIFEST.json) pokrýva aj zverejnené experimentálne záznamy.

Zachovaných je **1557 checkpointových súborov**, vrátane vybraných váh, dostupných posledných stavov a obnovovacích stavov. Toto číslo nie je počtom nezávislých tréningov. Finálne siete a história sa v archíve aj indexe rozlišujú. Vývojové a neúspešné pokusy nenahrádzajú finálne výsledky.

Úplné veľké vstupné FlowBench/CFD polia a stiahnuté pôvodné LLM sa neduplikujú; ich zdroje a pripnuté verzie sú v metodike. Surové natívne predikčné polia možno znovu vypočítať zo zachovaných sietí. Kód obsahuje pôvodné pracovné cesty a vyžaduje nastavenie prostredia pre daný stroj. Archívy zachovávajú váhy byte za bytom.

## Súvisiace experimenty

- [3D experimenty](https://github.com/Infinitiv525/DP-3D-experiments)
- [Porovnanie LLM](https://github.com/Infinitiv525/DP-LLM-comparison)
- [Porovnanie architektúr](https://github.com/Infinitiv525/DP-architecture-comparison)

Verejný vedecký export oddeľuje tieto podklady od súkromnej rozpracovanej dizertácie a prevádzkovej histórie. Referenčné dáta pochádzajú z [FlowBench](https://huggingface.co/datasets/BGLab/FlowBench); metódy autorov opisuje [pôvodný projekt](https://github.com/baskargroup/flowbench-tools). Licencie a zdroje externých vstupov sú v [metodike](03_methodology/DATA_AND_REUSE.md).

## Doplnkové vedecké dáta

Ďalšie surové merania a podklady na reprodukciu sú v [doplnkovom balíku](01_results/doplnkove_data/README.md).

## Historické numerické výstupy

Pôvodné polia a diagnostika starších pokusov sú v [samostatnom balíku](01_results/historicke_numericke_vystupy/README.md).

## Ďalšie výskumné záznamy

Doplnené pôvodné diagnostiky, protokoly, obrázky a zdroje sú v [samostatnom balíku](01_results/dalsie_vyskumne_zaznamy/README.md).

## Pôvodné Slurm logy

Pôvodné výstupné a chybové záznamy jednotlivých úloh Slurm sú v [samostatnom balíku](01_results/slurm_logy/README.md); zahŕňajú aj prípravné a neúspešné pokusy.

## Doplnenia z 19. novembra 2026

- [Prečo tieto štyri LLM](03_methodology/research_review_20261119/preco_tieto_LLM.txt)
- [Rozbor všetkých 240 kandidátov a 140 návrhov LLM](01_results/analyza_navrhov_20261119/README.md)

Doplnenia zachovávajú pôvodné váhy a výsledky. Rozbory pracujú s pôvodnými TRAIN/VAL záznamami alebo so samostatnými diagnostickými kontrolami.

## Plné dodatočné porovnanie Llamy

Built with Llama. **Llama‑3.1‑8B‑Instruct** (pridaná 19. 11. 2026): technická kontrola prešla4/4 a bolo spustené hlavné porovnanie40+25 tréningov s rovnakým protokolom a najviac8GPU súčasne. Pôvodné výsledky štyroch rodín zostávajú zachované; nová kohorta má skutočné vlastné dátumy.

- [Protokol, rozsah a chronológia](03_methodology/Llama_extension_20261119/protokol_Llama.txt)
- [Výsledok technickej kontroly](01_results/Llama_extension_20261119/overenie_pilota.json)
- [Dátovaný stav pri spustení](01_results/Llama_extension_20261119/stav_pri_spusteni.json)
- [Zdrojové kódy rozšírenia](04_source/Llama_extension_20261119)

Nové hlavné výsledky a TEST hodnotenie budú doplnené po skončení a zmrazení všetkých25 finálnych modelov. Váhy Meta sa tu nešíria; ich presná verzia a hashy sú v metodike.

## Dokončenie tréningov a TEST — 23. novembra 2026

Všetkých **65 tréningov Llamy je dokončených a overených**: každý400epoch cez640TRAIN prípadov. Zmrazených je25 finálnych modelov. Záverečný TEST na200prípadoch bol spustený na4H200 po úspešných kontrolách všetkých4GPU. Ďalšie tréningy sa nespúšťajú.

- [Audit všetkých65 tréningov](01_results/Llama_extension_20261119/all65_completed_training_audit.json)
- [Protokol záverečného TESTu](03_methodology/Llama_final_test_20261123/PROTOKOL.txt)
- [Kontrola4GPU pred otvorením TEST](01_results/Llama_final_test_20261123/GPU_validation_preflight.json)

Ukladajú sa aj všetky5000predikované celé polia priamo naPerune. Výsledkové tabuľky a archív pribudnú po dokončení a kontrole TESTu; pôvodné štyri porovnania zostávajú zachované.

## Výsledok Llamy — TEST dokončený

**TEST prešiel25/25modelov, všetkých5000uložených polí bolo nezávisle skontrolovaných.** Llama je v presnosti prakticky zhodná sQwen/Phi/Granite. Ministral má opisne o3.28%nižšiu MSE, ale dodatočný párový interval rozdielu zahŕňa nulu. Llama teda nepreukázala zlepšenie oproti ostatnýmLLM ani pevnej konfigurácii. TPE má otestovanú nižšiu MSE (Llama+21.09%); všetkyLlamaPINN spĺňajú per-case p99≤5ms.

- [Ľudsky čitateľný výsledok a obmedzenia](01_results/Llama_final_test_20261123/VYSLEDOK_LLAMA.txt)
- [Spoločné presné výsledky všetkých metód](01_results/Llama_final_test_20261123/combined_summary.json)
- [Doplnkové párové porovnania Llamy s ostatnýmiLLM](01_results/Llama_final_test_20261123/pairwise_Llama_posthoc.json)
- [Všetkých5000nových výsledkov po prípadoch](01_results/Llama_final_test_20261123/case_metrics.csv)

Pôvodné kontrasty sa nemenia. Llama zostáva označená ako dodatočná kohorta; párové porovnania s ďalšímiLLM sú explicitne posthoc a exploratívne.

## Kompletný verejný archív Llamy

**Všetkých65tréningov,25finálnych modelov a5000celých TEST polí je verejne zachovaných.** [Zrozumiteľný katalóg a odkazy](01_results/Llama_final_test_20261123/ARCHIV.md). Iba SHA256-identické polia zdieľajú súbor; modelové identity a všetky váhy zostávajú samostatné. Priamy prenos Perun→GitHub, bez kopírovania doPC.

## Prečo majú štyri konfigurátory rovnaký výsledok

Dodatočný audit skontroloval175 pôvodných LLM odpovedí,325 tréningových výstupov,125 finálnych checkpointov a15 natívnych tokenizerových/gramatických skúšok. Nenašla sa zámena modelov ani prepísanie parametrov. **Rovnakú konfiguráciu vybral validačný program; Qwen/Llama pritom skúšali aj iné návrhy, Phi/Granite opakovali štart.** Párované semená a rovnaký postup viedli k rovnakým váham podľa semena.

[Čitateľný výsledok auditu](03_methodology/configuration_identity_audit_20261123/OVERENIE_ZHODY.txt) · [Podrobné číselné dôkazy](01_results/configuration_identity_audit_20261123)

## Analýza sily ďalších LLM hľadaní — 24. novembra 2026

[Čitateľný výsledok a GPU náklady](01_results/power_analysis_20261124/VYSLEDKY.txt) · [Úplné číselné varianty](01_results/power_analysis_20261124/sample_size_designs.json) · [Protokol](03_methodology/power_analysis_20261124/protokol.txt).

Doplňujúca CPU analýza plánuje nové nezávislé hľadania piatich LLM proti TPE. Rozlišuje neistotu výberu konfigurácie a finálnych tréningových semien; starých päť hľadaní nepočíta do nového potvrdenia. Uvádza modelový odhad aj opatrnejšiu rozptylovú citlivosť. Žiadne nové GPU experimenty sa týmto nespustili.
