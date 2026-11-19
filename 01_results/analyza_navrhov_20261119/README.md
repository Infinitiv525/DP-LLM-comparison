# Správanie konfigurátorov v uzavretom LLM experimente

Analyzovaných 240 kandidátskych tréningov: 6 metód ×5 hľadaní ×8 kandidátov. Štyri LLM vytvorili 140 adaptívnych návrhov (7 na hľadanie); kandidát 0 bol spoločný pevný štart. Ide o opis pôvodných TRAIN/VAL záznamov, bez nových návrhov, tréningov, TEST alebo zmeny výberov.

| Metóda | Priemer unikátnych konfigurácií /8 | Opakovaný návrh /35 | Nové najlepšie VAL /35 | Hľadaní bez zlepšenia nad toleranciu /5 | Priemerné zlepšenie nad štart |
|---|---:|---:|---:|---:|---:|
| qwen3_8b | 3.40 | 23 | 0 | 5 | 0.00 % |
| ministral3_8b | 8.00 | 0 | 5 | 1 | 4.14 % |
| phi4_mini | 1.00 | 35 | 0 | 5 | 0.00 % |
| granite33_8b | 1.00 | 35 | 0 | 5 | 0.00 % |
| random | 8.00 | 0 | 6 | 0 | 9.31 % |
| tpe | 8.00 | 0 | 8 | 0 | 17.85 % |

Tolerancia zlepšenia je absolútne 1e-12 MSE a slúži iba na odlíšenie numerického šumu; nejde o prah fyzikálnej kvality. Pôvodný striktne najnižší kandidát aj skóre zostávajú v dátach bez zmeny a pôvodné výbery sa nemenia.

Opakovaná konfigurácia spotrebovala pôvodný nový tréning, preto sa jej cena neodpočítava. Priemerné zlepšenie sa najprv počíta v každom hľadaní voči jeho vlastnému štartu, až potom priemeruje cez päť hľadaní. Best-so-far zahŕňa pôvodný štart; zhoršenie jednotlivého návrhu sa tým neskrýva v trials.json. Presná jedinečnosť sama nie je kvalita explorácie: TPE/random zapisujú spojitú LR s väčšou presnosťou. Sú uložené aj spoločné logaritmické LR koše a normalizované vzdialenosti piatich parametrov.

## Čas a zdroje konfigurátora

| LLM | Wall procesov návrhu [s] | Z toho generovanie [s] | Vstupné tokeny | Výstupné tokeny | Opravy |
|---|---:|---:|---:|---:|---:|
| qwen3_8b | 1910.9 | 1417.4 | 54485 | 4998 | 0 |
| ministral3_8b | 2192.6 | 1723.9 | 56147 | 6144 | 0 |
| phi4_mini | 1689.2 | 1236.5 | 38879 | 4328 | 0 |
| granite33_8b | 1997.9 | 1552.4 | 57460 | 5600 | 0 |

Wall zahŕňa načítanie modelu, prípravu schémy, generovanie a prípadné opravy. Súčet procesov nie je úplná faktúra alokácie Slurm; účtovanie pôvodných dávok je zachované samostatne. RAM/VRAM a podrobné časy sú v complete.json a trials.json.

## Spätná väzba a zdôvodnenia

Každý adaptívny vstup bol overený oproti skutočnej histórii dokončených pokusov a proti pôvodnému promptu. Parametre odpovede sa zhodujú s request.json a použitým tréningom. Vynútená JSON schéma a odkaz evidence_ids kontrolujú prenos/formát; nepreukazujú porozumenie ani príčinný vplyv spätnej väzby. Na ten by bol potrebný samostatný experiment bez histórie alebo s kontrolovane pozmenenou históriou. Žiadny taký test tento rozbor nepredstiera.

Predom určené príklady: prvý adaptívny návrh hľadania 0 každej rodiny. Texty sú deklarované dôvody modelu, nie skryté vnútorné uvažovanie.

**qwen3_8b**

- Dôvod: Increase physics_weight to emphasize flow dynamics while adjusting bc_weight to better capture boundary effects
- Očakávanie: Potential reduction in velocity and pressure errors by focusing more on fluid motion and boundary adherence
- Riziko: Higher computational cost and possible overfitting if physics_weight is too high
- Zmena VAL oproti predošlému najlepšiemu: -7.762 % (kladná je zlepšenie).

**ministral3_8b**

- Dôvod: Increase physics_weight to better constrain residuals, reduce pressure_weight to focus more on velocity, increase bc_str
- Očakávanie: Reduce divergence error and improve boundary adherence while maintaining velocity accuracy and lowering pressure oversho
- Riziko: Potential instability if physics_weight dominates too early during warmup, but reduced pressure_weight mitigates this
- Zmena VAL oproti predošlému najlepšiemu: -47.450 % (kladná je zlepšenie).

**phi4_mini**

- Dôvod: Maximizing min MSE
- Očakávanie: Minimize val MSE by increasing weightings slightly to boost overall accuracy while maintaining low errors across other P
- Riziko: Low - Slightly increased risk due to small adjustments without significant historical negative impact observed
- Zmena VAL oproti predošlému najlepšiemu: 0.000 % (kladná je zlepšenie).

**granite33_8b**

- Dôvod: Based on the given history, the current configuration seems balanced. We'll slightly increase the learning rate to avoid
- Očakávanie: improved convergence without compromising accuracy
- Riziko: potential overshoot if learning rate increased too much
- Zmena VAL oproti predošlému najlepšiemu: 0.000 % (kladná je zlepšenie).

Tieto jednotlivé texty neslúžia ako dôkaz príčiny úspechu. Napríklad zvýšenie násobnej váhy PDE chyby pri pevných 64 PDE bodoch samo nezvyšuje ich počet; výrok modelu o výpočtovej cene treba overiť proti kódu a meraniu. Hlavné závery o presnosti a štatistike zostávajú vo finálnom uzavretom TEST reporte.
