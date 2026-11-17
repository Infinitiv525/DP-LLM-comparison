# Zdrojový kód

Pôvodné vedecké implementácie sú v `original/test/`. Zachované názvy uľahčujú spojenie s konfiguráciami a históriou. Tréningové a vyhodnocovacie postupy používajú pôvodné absolútne pracovné cesty; pri reprodukcii ich nastav v pracovnej kópii. Prevádzkové prihlásenia a nasadzovacie pomocníky sa nezverejňujú.

## Hlavné súbory

- [SharedSteadyPINN](original/test/training/ldc_shared_pinn_v26.py) definuje spoločnú sieť.
- [Tréning hlavného experimentu](original/test/training/train_ldc_main_v33.py).
- [Konečné vyhodnotenie](original/test/training/evaluate_ldc_final_v49.py).
- Generovanie konfigurácií a riadenie hľadaní sú v `original/test/llm/`; pre hlavný experiment používaj súbory s `ldc` a verziou v33. Staršie FPO súbory patria k pilotom.
