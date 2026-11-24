# Plánovanie ďalších nezávislých hľadaní LLM

[Ľudsky čitateľné výsledky](VYSLEDKY.txt) · [Číselné varianty](sample_size_designs.json) · [Nezávislá kontrola](independent_verification.json)

CPU analýza z pôvodných dát; žiadne nové GPU tréningy. Päť LLM proti spoločnej novej kontrole TPE. Rozlišujeme celé nové nezávislé potvrdenie a užšie exploratívne doplnenie so starým panelom tréningových semien.

Jeden nový nezávislý blok používa päť čerstvých finálnych tréningových semien: párovaných medzi metódami, odlišných medzi blokmi. Závery sú podmienené tým istým TEST benchmarkom.

[Protokol](../../03_methodology/power_analysis_20261124/protokol.txt) · [Reprodukovateľný zdroj](../../04_source/power_analysis_20261124)

Výpočet N závisí od hypotetického efektu a odhadu variability z iba piatich hľadaní. Horná rozptylová citlivosť je podmienená normálnym kríženým modelom; nie je garanciou budúcej sily.
