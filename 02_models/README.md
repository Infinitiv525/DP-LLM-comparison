# Váhy a index sietí

Finálne vybraných modelov: 155. Celkom zachovaných checkpointových súborov: 1557.

[Archívy na stiahnutie](https://github.com/Infinitiv525/DP-LLM-comparison/releases/tag/research-artifacts-v1), [index všetkých checkpointov](model_index.csv), [úplný manifest](ARTIFACT_MANIFEST.json), [kontrolné súčty](SHA256SUMS.txt). Každý diel tar.gz sa samostatne rozbaľuje do rovnakého priečinka. Finálne siete sú pod `models/final`, ostatné pod `models/history`. Najprv over SHA dielu a následne SHA požadovaného súboru z manifestu.

`best` je vybraný checkpoint; `last` a `training_state` sú dostupné posledné/obnovovacie stavy. Nie sú to ďalšie nezávislé siete. Označenie final neznamená nulovú chybu. Bezpečne rozbaľuj napríklad cez Python 3.12+ `tarfile` s `filter="data"`. PyTorch checkpointy načítavaj len z dôveryhodného overeného pôvodu.

## Vyber konkrétnu rodinu alebo metódu

| Metóda | Vybrané záznamy | Váhy a seedy |
|---|---:|---|
| Pevná konfigurácia | 5 | [Otvoriť modely](fixed/README.md) |
| Granite3.3 8B | 25 | [Otvoriť modely](granite33_8b/README.md) |
| Ministral3 8B | 25 | [Otvoriť modely](ministral3_8b/README.md) |
| Phi4 mini | 25 | [Otvoriť modely](phi4_mini/README.md) |
| Qwen3 8B | 25 | [Otvoriť modely](qwen3_8b/README.md) |
| Náhodné hľadanie | 25 | [Otvoriť modely](random/README.md) |
| TPE | 25 | [Otvoriť modely](tpe/README.md) |
