# Úplné priestorové polia experimentu s LLM

**Všetkých 155 finálnych PINN má uložené celé u/v/p polia pre všetkých 200 TEST prípadov** v natívnom rozlíšení 512 × 512. Ide o 31 000 dvojíc model/prípad. Sú to ustálené riešenia rôznych prípadov, nie časová animácia.

[Celé vydanie](https://github.com/Infinitiv525/DP-LLM-comparison/releases/tag/native-llm-test-fields-20261118). Rovnaké súbory váh sa objavili pri viacerých konfigurátoroch. Preto 101 fyzických súborov predictions.npy pokrýva všetkých 155 modelových ID; zhodu určuje SHA256 celého checkpointu a explicitná tabuľka. Žiaden finálny model nebol vynechaný ani znovu trénovaný.

Polia boli dodatočne rekonštruované z uzavretých váh pôvodnou funkciou V49 v pôvodnom prostredí FP32 na H200. Každý prípad každého modelového ID prešiel zhodou štyroch pôvodných chybových metrík s pôvodným validačným tolerančným pravidlom rtol=0.001, atol=1e-9. Táto tolerancia kontroluje numerickú reprodukciu, nie požadovanú presnosť siete. Pôvodné výsledky a merania rýchlosti zostali nezmenené; časovanie sa neopakovalo.

## Obsah jedného modelového balíka

Každý modelový balík rozbaľ do samostatného priečinka podľa jeho názvu. Spoločné dátové balíky majú vlastné jedinečné podpriečinky.

- predictions.npy: float32 [200,512,512,3], osi prípad/y/x/pole, kanály u,v,p. Poradie prípadov je vo verification/reconstruction.json.
- model/best.pt, aliases.json a normalization.json: pôvodné váhy a všetky ekvivalentné modelové záznamy.
- original_evaluation/: pôvodné súhrny a surové časovania každého modelového ID v skupine.
- verification/: porovnanie rekonštruovaných polí s pôvodnými metrikami.

## Vstupy, referencie a reprodukcia

Osobitné balíky shared_data__train_*, shared_data__validation_* a shared_data__test_* zachovávajú všetkých 1000 použitých prípadov (640/160/200) v pôvodnom poradí daného splitu. Každý diel má najviac 100 prípadov a obsahuje reference_uvp.npy, geometry_sdf.npy, fluid_mask.npy, reynolds.npy, case_indices.npy a cases.json. Dva TEST diely sa zreťazia v číselnom poradí pozdĺž prvej osi, čím vznikne rovnaké poradie 200 prípadov ako v predictions.npy.

Balík shared_metadata_and_reproduction_source obsahuje pôvodnú normalizáciu, vstupnú schému, rozdelenie dát, protokol, manifest modelov, použité pôvodné zdrojové súbory a nové exportné skripty aj záznam prostredia. Referencie TRAIN/VAL sú presné hodnoty skutočných tréningových cache; TEST používa pôvodný canonical_target. Zachovaná je tlaková referencia, nulovanie v pevnej oblasti a pôvodné súradnice. Nevznikli nové CFD simulácie.

Zdrojom referencií je [BGLab/FlowBench, pripnutá verzia](https://huggingface.co/datasets/BGLab/FlowBench/tree/937c672604d4c9f39aceedf1fa2f0a603695264b), LDC_NS_2D/512x512 harmonics. Odvodený použitý výber zachováva pôvodné autorstvo a podmienky CC BY NC 4.0; pôvodná datasetová karta a citácia sú v spoločných metadátach.

Každý člen archívu aj archív má SHA256. Overené bolo úplné rozbalenie, serverový digest, spätné čítanie na Perune a dostupnosť odkazov bez prihlásenia. Veľké dáta neprechádzali cez PC.

## Modely

| Metóda | Model | Seed | Natívne polia a váhy |
|---|---|---:|---|
| fixed | fixed_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| fixed | fixed_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r01_k01__native_TEST_fields.tar.gz) |
| fixed | fixed_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k02__native_TEST_fields.tar.gz) |
| fixed | fixed_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| fixed | fixed_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/fixed_k04__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r02_k01__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k03__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r04_k00__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r04_k02__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| granite33_8b | granite33_8b_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r04_k04__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r00_k00__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r00_k01__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r00_k02__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r00_k03__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r00_k04__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r01_k02__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r02_k00__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r02_k01__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r02_k02__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r02_k03__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r02_k04__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r03_k00__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r03_k01__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r03_k02__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r03_k03__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r03_k04__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r04_k00__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r04_k01__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r04_k02__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r04_k03__native_TEST_fields.tar.gz) |
| ministral3_8b | ministral3_8b_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/ministral3_8b_r04_k04__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k01__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r01_k01__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r01_k03__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r02_k04__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r03_k01__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r03_k03__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r04_k01__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k02__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k03__native_TEST_fields.tar.gz) |
| phi4_mini | phi4_mini_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k01__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k02__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k03__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k00__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k01__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k02__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k03__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r01_k04__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r02_k00__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r02_k01__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r02_k02__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k00__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r03_k04__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r04_k00__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/granite33_8b_r00_k01__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k02__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/phi4_mini_r00_k03__native_TEST_fields.tar.gz) |
| qwen3_8b | qwen3_8b_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/qwen3_8b_r00_k04__native_TEST_fields.tar.gz) |
| random | random_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r00_k00__native_TEST_fields.tar.gz) |
| random | random_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r00_k01__native_TEST_fields.tar.gz) |
| random | random_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r00_k02__native_TEST_fields.tar.gz) |
| random | random_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r00_k03__native_TEST_fields.tar.gz) |
| random | random_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r00_k04__native_TEST_fields.tar.gz) |
| random | random_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r01_k00__native_TEST_fields.tar.gz) |
| random | random_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r01_k01__native_TEST_fields.tar.gz) |
| random | random_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r01_k02__native_TEST_fields.tar.gz) |
| random | random_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r01_k03__native_TEST_fields.tar.gz) |
| random | random_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r01_k04__native_TEST_fields.tar.gz) |
| random | random_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r02_k00__native_TEST_fields.tar.gz) |
| random | random_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r02_k01__native_TEST_fields.tar.gz) |
| random | random_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r02_k02__native_TEST_fields.tar.gz) |
| random | random_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r02_k03__native_TEST_fields.tar.gz) |
| random | random_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r02_k04__native_TEST_fields.tar.gz) |
| random | random_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r03_k00__native_TEST_fields.tar.gz) |
| random | random_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r03_k01__native_TEST_fields.tar.gz) |
| random | random_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r03_k02__native_TEST_fields.tar.gz) |
| random | random_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r03_k03__native_TEST_fields.tar.gz) |
| random | random_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r03_k04__native_TEST_fields.tar.gz) |
| random | random_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r04_k00__native_TEST_fields.tar.gz) |
| random | random_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r04_k01__native_TEST_fields.tar.gz) |
| random | random_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r04_k02__native_TEST_fields.tar.gz) |
| random | random_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r04_k03__native_TEST_fields.tar.gz) |
| random | random_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/random_r04_k04__native_TEST_fields.tar.gz) |
| tpe | tpe_r00_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r00_k00__native_TEST_fields.tar.gz) |
| tpe | tpe_r00_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r00_k01__native_TEST_fields.tar.gz) |
| tpe | tpe_r00_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r00_k02__native_TEST_fields.tar.gz) |
| tpe | tpe_r00_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r00_k03__native_TEST_fields.tar.gz) |
| tpe | tpe_r00_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r00_k04__native_TEST_fields.tar.gz) |
| tpe | tpe_r01_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r01_k00__native_TEST_fields.tar.gz) |
| tpe | tpe_r01_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r01_k01__native_TEST_fields.tar.gz) |
| tpe | tpe_r01_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r01_k02__native_TEST_fields.tar.gz) |
| tpe | tpe_r01_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r01_k03__native_TEST_fields.tar.gz) |
| tpe | tpe_r01_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r01_k04__native_TEST_fields.tar.gz) |
| tpe | tpe_r02_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r02_k00__native_TEST_fields.tar.gz) |
| tpe | tpe_r02_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r02_k01__native_TEST_fields.tar.gz) |
| tpe | tpe_r02_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r02_k02__native_TEST_fields.tar.gz) |
| tpe | tpe_r02_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r02_k03__native_TEST_fields.tar.gz) |
| tpe | tpe_r02_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r02_k04__native_TEST_fields.tar.gz) |
| tpe | tpe_r03_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r03_k00__native_TEST_fields.tar.gz) |
| tpe | tpe_r03_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r03_k01__native_TEST_fields.tar.gz) |
| tpe | tpe_r03_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r03_k02__native_TEST_fields.tar.gz) |
| tpe | tpe_r03_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r03_k03__native_TEST_fields.tar.gz) |
| tpe | tpe_r03_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r03_k04__native_TEST_fields.tar.gz) |
| tpe | tpe_r04_k00 | 150001 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r04_k00__native_TEST_fields.tar.gz) |
| tpe | tpe_r04_k01 | 150002 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r04_k01__native_TEST_fields.tar.gz) |
| tpe | tpe_r04_k02 | 150003 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r04_k02__native_TEST_fields.tar.gz) |
| tpe | tpe_r04_k03 | 150004 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r04_k03__native_TEST_fields.tar.gz) |
| tpe | tpe_r04_k04 | 150005 | [balík](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/native-llm-test-fields-20261118/tpe_r04_k04__native_TEST_fields.tar.gz) |
