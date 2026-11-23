# Kompletný archív experimentu s Llamou

Built with Llama. **Všetkých65 tréningov,25 finálnych modelov a5000 natívnych TEST polí je verejne uložených.** Modely a polia sa preniesli priamo z Perunu, bez kópie doPC.

## Čo otvoriť najprv

- [Výsledok porovnania](VYSLEDOK_LLAMA.txt)
- [Verejný archív – všetky súbory](https://github.com/Infinitiv525/DP-LLM-comparison/releases/tag/llama31-full-experiment-20261123)
- [Váhy a konfigurácie všetkých65 modelov](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_all65_best_models_and_configs.tar.gz)
- [Úplný strojovo čitateľný katalóg](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/LLAMA_COMPLETE_ARCHIVE_MANIFEST.json)

Balík váh má zrozumiteľné priečinky models_search a models_final, ďalej modelID. Obsahuje best.pt a config.json. V pôvodných úplných záznamoch zostali aj last.pt, stav optimalizátora/RNG, celá história, návrhy/dôvody LLM, skutočná spätná väzba a procesné záznamy.

Polia majú rozmer200×512×512×3, float32, kanály u,v,p.200 je počet rôznych TEST simulácií, nie časových krokov. Iba úplne byte-identické polia zdieľajú archív; všetky25 modelové identity a ich vlastné váhy ostali zachované.

| Hľadanie (1–5) | Seed PINN | Archív celých200 TEST polí |
|---|---|---|
| 1 | 150001 | [Llama31_8B_r00_k00](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k00__native_TEST_fields.tar.gz) |
| 1 | 150002 | [Llama31_8B_r00_k01](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k01__native_TEST_fields.tar.gz) |
| 1 | 150003 | [Llama31_8B_r00_k02](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k02__native_TEST_fields.tar.gz) |
| 1 | 150004 | [Llama31_8B_r00_k03](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k03__native_TEST_fields.tar.gz) |
| 1 | 150005 | [Llama31_8B_r00_k04](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k04__native_TEST_fields.tar.gz) |
| 2 | 150001 | [Llama31_8B_r01_k00](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k00__native_TEST_fields.tar.gz) |
| 2 | 150002 | [Llama31_8B_r01_k01](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k01__native_TEST_fields.tar.gz) |
| 2 | 150003 | [Llama31_8B_r01_k02](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k02__native_TEST_fields.tar.gz) |
| 2 | 150004 | [Llama31_8B_r01_k03](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k03__native_TEST_fields.tar.gz) |
| 2 | 150005 | [Llama31_8B_r01_k04](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k04__native_TEST_fields.tar.gz) |
| 3 | 150001 | [Llama31_8B_r02_k00](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k00__native_TEST_fields.tar.gz) |
| 3 | 150002 | [Llama31_8B_r02_k01](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k01__native_TEST_fields.tar.gz) |
| 3 | 150003 | [Llama31_8B_r02_k02](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k02__native_TEST_fields.tar.gz) |
| 3 | 150004 | [Llama31_8B_r02_k03](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k03__native_TEST_fields.tar.gz) |
| 3 | 150005 | [Llama31_8B_r02_k04](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k04__native_TEST_fields.tar.gz) |
| 4 | 150001 | [Llama31_8B_r03_k00](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r02_k00__native_TEST_fields.tar.gz) |
| 4 | 150002 | [Llama31_8B_r03_k01](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r03_k01__native_TEST_fields.tar.gz) |
| 4 | 150003 | [Llama31_8B_r03_k02](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k02__native_TEST_fields.tar.gz) |
| 4 | 150004 | [Llama31_8B_r03_k03](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r03_k03__native_TEST_fields.tar.gz) |
| 4 | 150005 | [Llama31_8B_r03_k04](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r03_k04__native_TEST_fields.tar.gz) |
| 5 | 150001 | [Llama31_8B_r04_k00](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r04_k00__native_TEST_fields.tar.gz) |
| 5 | 150002 | [Llama31_8B_r04_k01](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r01_k01__native_TEST_fields.tar.gz) |
| 5 | 150003 | [Llama31_8B_r04_k02](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r04_k02__native_TEST_fields.tar.gz) |
| 5 | 150004 | [Llama31_8B_r04_k03](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r04_k03__native_TEST_fields.tar.gz) |
| 5 | 150005 | [Llama31_8B_r04_k04](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_8B_r00_k04__native_TEST_fields.tar.gz) |

## Úplné pôvodné záznamy

- [Llama31_complete_training_records_part01_of02.tar.gz](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_complete_training_records_part01_of02.tar.gz)
- [Llama31_complete_training_records_part02_of02.tar.gz](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_complete_training_records_part02_of02.tar.gz)
- [Llama31_final_TEST_comparison_and_reproduction_source.tar.gz](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_final_TEST_comparison_and_reproduction_source.tar.gz)
- [Kód archivácie a plán](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/Llama31_archival_code_and_finalization_v2.tar.gz)
- [SHA256 kontrolné súčty](https://github.com/Infinitiv525/DP-LLM-comparison/releases/download/llama31-full-experiment-20261123/LLAMA_SHA256SUMS.txt)

Zdieľané vstupy, referencie a split640/160/200 sú už vo [verejnom archíve pôvodného experimentu](https://github.com/Infinitiv525/DP-LLM-comparison/releases/tag/native-llm-test-fields-20261118). Pochádzajú z [FlowBench](https://arxiv.org/abs/2409.18032), s pôvodnými podmienkami CC BY NC4.0 a citáciou. Váhy MetaLlama sa tu nešíria; presná verzia je v metodike, Llama3.1CommunityLicense a NOTICE sú priložené.

Každý archív prešiel kontrolou všetkých položiek po rozbalení, serverovým SHA256 aj úplným spätným prečítaním naPerune. Verejné odkazy boli overené bez prihlásenia.
