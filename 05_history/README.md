# Vývoj a pilotné pokusy

V archívoch sú od hlavného porovnania oddelené aj dostupné staršie siete, rozbehy a technické pokusy. Súbory best, last a training_state sú rôzne stavy tej istej siete; ich počet nie je počtom nezávislých experimentov. Presné zaradenie určuje index modelov a manifest príloh.

## Podklady jednotlivých pokusov

Cesty sú skrátené pre čitateľnosť a kompatibilitu s Windows. Obsah súborov ostal byte za bytom rovnaký; pôvodné cesty a SHA sú v `LOCAL_SCIENTIFIC_RECORDS.json`.

| Pokus | Počet súborov |
|---|---:|
| [PRECITAJ_MA.txt](pokusy/PRECITAJ_MA.txt/README.md) | 1 |
| [architektura_pinn_v26_20261105](pokusy/architektura_pinn_v26_20261105/README.md) | 2 |
| [audit_flowbench_fpo](pokusy/audit_flowbench_fpo/README.md) | 2 |
| [audit_fyziky_20261103_v18](pokusy/audit_fyziky_20261103_v18/README.md) | 52 |
| [audit_geometrie_20261103_v20](pokusy/audit_geometrie_20261103_v20/README.md) | 45 |
| [audit_hlbkovy_20261030](pokusy/audit_hlbkovy_20261030/README.md) | 2 |
| [audit_modelu_20261031_v9](pokusy/audit_modelu_20261031_v9/README.md) | 4 |
| [audit_replikovatelnosti_20261103.txt](pokusy/audit_replikovatelnosti_20261103.txt/README.md) | 1 |
| [dlhsia_historia_20261103_v22](pokusy/dlhsia_historia_20261103_v22/README.md) | 6 |
| [dlhsia_historia_20261103_v23](pokusy/dlhsia_historia_20261103_v23/README.md) | 304 |
| [dlhy_priebeh_20261031_v7](pokusy/dlhy_priebeh_20261031_v7/README.md) | 21 |
| [dokoncenie_20261107_v47](pokusy/dokoncenie_20261107_v47/README.md) | 6 |
| [doplnenie_24gpu_20261106_v46](pokusy/doplnenie_24gpu_20261106_v46/README.md) | 4 |
| [finalny_test_20261108_v49](pokusy/finalny_test_20261108_v49/README.md) | 18 |
| [gradient_20261031_v8](pokusy/gradient_20261031_v8/README.md) | 23 |
| [kompletny_audit_20261106_v44](pokusy/kompletny_audit_20261106_v44/README.md) | 19 |
| [kontrola_grid_fyziky_20261103_v19](pokusy/kontrola_grid_fyziky_20261103_v19/README.md) | 343 |
| [kontrola_kvality_20261104_v31](pokusy/kontrola_kvality_20261104_v31/README.md) | 20 |
| [kopie_manifest.json](pokusy/kopie_manifest.json/README.md) | 1 |
| [llm_konfigurator_20261104_v27](pokusy/llm_konfigurator_20261104_v27/README.md) | 4 |
| [llm_konfigurator_20261104_v28](pokusy/llm_konfigurator_20261104_v28/README.md) | 5 |
| [llm_konfigurator_20261104_v29](pokusy/llm_konfigurator_20261104_v29/README.md) | 2 |
| [llm_konfigurator_20261104_v30](pokusy/llm_konfigurator_20261104_v30/README.md) | 8 |
| [llm_ustalene_ldc_20261104](pokusy/llm_ustalene_ldc_20261104/README.md) | 5 |
| [paralelizacia_16gpu_20261104_v35](pokusy/paralelizacia_16gpu_20261104_v35/README.md) | 9 |
| [paralelizacia_24gpu_20261104_v36](pokusy/paralelizacia_24gpu_20261104_v36/README.md) | 8 |
| [pociatocny_stav_20261030_v4](pokusy/pociatocny_stav_20261030_v4/README.md) | 11 |
| [podklady_autorov_20261103](pokusy/podklady_autorov_20261103/README.md) | 6 |
| [postupny_pinn_20261031_v5](pokusy/postupny_pinn_20261031_v5/README.md) | 17 |
| [pravidla_vstupov.txt](pokusy/pravidla_vstupov.txt/README.md) | 1 |
| [predlohy_a_audit_20261101](pokusy/predlohy_a_audit_20261101/README.md) | 8 |
| [prenos_20261102_v16](pokusy/prenos_20261102_v16/README.md) | 33 |
| [presnost_a_inventura_20261030](pokusy/presnost_a_inventura_20261030/README.md) | 3 |
| [priama_rekonstrukcia_20261031_v11](pokusy/priama_rekonstrukcia_20261031_v11/README.md) | 18 |
| [priamy_curl_20261102_v15](pokusy/priamy_curl_20261102_v15/README.md) | 224 |
| [priestorove_vstupy_20261031_v10](pokusy/priestorove_vstupy_20261031_v10/README.md) | 23 |
| [priestorovy_kontext_20261103_v21](pokusy/priestorovy_kontext_20261103_v21/README.md) | 349 |
| [prisposobenie_datasetu_20261103.txt](pokusy/prisposobenie_datasetu_20261103.txt/README.md) | 1 |
| [rozsirenie_20261030_v2](pokusy/rozsirenie_20261030_v2/README.md) | 32 |
| [rozsirenie_20261030_v3](pokusy/rozsirenie_20261030_v3/README.md) | 15 |
| [rozsireny_trening_20261102_v17](pokusy/rozsireny_trening_20261102_v17/README.md) | 467 |
| [samostatne_modely_20261104_v24](pokusy/samostatne_modely_20261104_v24/README.md) | 94 |
| [setrenie_gpu_20261106_v43](pokusy/setrenie_gpu_20261106_v43/README.md) | 5 |
| [spolocny_ustaleny_pinn_20261104_v26](pokusy/spolocny_ustaleny_pinn_20261104_v26/README.md) | 10 |
| [stabilizacia_20261031_v6](pokusy/stabilizacia_20261031_v6/README.md) | 35 |
| [stav_20261106_v42](pokusy/stav_20261106_v42/README.md) | 2 |
| [stav_20261106_v45](pokusy/stav_20261106_v45/README.md) | 6 |
| [stav_20261107_v47](pokusy/stav_20261107_v47/README.md) | 7 |
| [stav_20261108_v48](pokusy/stav_20261108_v48/README.md) | 6 |
| [ustalene_ldc_20261104_v25](pokusy/ustalene_ldc_20261104_v25/README.md) | 9 |
| [uzavretie_experimentu_20261108_v50](pokusy/uzavretie_experimentu_20261108_v50/README.md) | 15 |
| [vacsia_siet_20261105_v39](pokusy/vacsia_siet_20261105_v39/README.md) | 5 |
| [vahy_fyziky_20261102_v14](pokusy/vahy_fyziky_20261102_v14/README.md) | 43 |
| [velkosti_dat_20261105](pokusy/velkosti_dat_20261105/README.md) | 2 |
| [velky_test_ustaleneho_pinn_20261104_v32](pokusy/velky_test_ustaleneho_pinn_20261104_v32/README.md) | 3 |
| [velky_test_ustaleneho_pinn_20261104_v33](pokusy/velky_test_ustaleneho_pinn_20261104_v33/README.md) | 14 |
| [viac_pripadov_20261031_v12](pokusy/viac_pripadov_20261031_v12/README.md) | 29 |
| [video_ustaleneho_prudenia_20261105_v41](pokusy/video_ustaleneho_prudenia_20261105_v41/README.md) | 5 |
| [vlastny_model_20261101_v13](pokusy/vlastny_model_20261101_v13/README.md) | 36 |
