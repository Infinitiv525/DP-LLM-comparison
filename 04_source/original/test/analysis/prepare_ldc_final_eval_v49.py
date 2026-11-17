"""Freeze the final protocol and source before the first TEST target access."""
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
for folder in ['analysis','llm','training']:sys.path.insert(0,str(ROOT/'test'/folder))
from ldc_main_common_v33 import WORK,read,save,digest,fingerprint,verify
from ldc_main_scheduler_v33 import locked

FINAL=ROOT/'results/ldc_final_test_20261108_v49'


def main():
    verify()
    assert not (FINAL/'freeze.json').exists() and not (FINAL/'test_access').exists()
    closure=read(FINAL/'training_closeout/summary.json')
    assert closure['status']=='TRAINING_AUDITED_MODEL_MANIFEST_READY' and closure['problems']==0
    manifest=read(FINAL/'models.json');assert len(manifest['models'])==155
    assert digest(FINAL/'models.json')==closure['models_sha256']
    main=read(WORK/'protocol.json');prepared=read(main['prepared_path'])
    split=read(main['split_path'])['indices'];inputs=read(Path(prepared['cache'])/'inputs.json')
    assert len(split['test'])==200 and not (set(split['test'])&set(split['train']+split['validation']))
    protocol=dict(version=49,task='Final inference of155 frozen shared steady2D PINNs on200 sealed TEST flowcases',
        prepared_path=main['prepared_path'],split_path=main['split_path'],test_ids=split['test'],
        test_geometries=[inputs['geometry_hashes'][i] for i in split['test']],
        search_methods=main['methods'],llm_methods=[m['label'] for m in main['models']],methods=main['methods']+['fixed'],
        final_models=155,test_cases=200,final_seeds=list(range(150001,150006)),
        training_completed=395,main_state_sha256=manifest['main_state_sha256'],
        models_sha256=digest(FINAL/'models.json'),normalization='Frozen TRAIN-only values; never refit on TEST',
        fields=['u','v','p'],grid=[512,512],coordinates='Same audited normalized[0,1]^2 and viscosity1/Re as V33',
        targets='Same canonical_target: float32 u/v/p, own fluid pressure mean, known solid zeros, property channel excluded',
        inference='Same frozen V26 query: native geometry pooling, encoder, coordinate MLP, interpolation, curl, own gauge and known mask',
        timing=dict(warmup=30,repeats=200,scope='Full GPU query with inputs onGPU; excludes host transfer/rendering/static coordinate setup',
                    threshold_p99_ms=5,precision='FP32 weights/data; matmulTF32False,cuDNNTF32True'),
        boundary=dict(outer_points_per_side=256,corner_exclusion=1/512,obstacle='All native mask half-cell faces'),
        metrics='Same unweighted all-cell u/v/p MSE, fluid velocity/pressure relativeL2, all-fieldL2, near-obstacle error, FD NS/divergence, boundary errors',
        analysis=dict(primary_metric='Mean unweighted all-cell TEST MSE, equal weighting of200cases,5seeds then5searches',
            primary_contrasts='Each of4LLM methods minusTPE; negative favoursLLM. Random/fixed and all other metrics descriptive.',
            bootstrap='Paired crossed resampling:5search IDs and85TEST geometry clusters; same draws for every method. Include all cases per sampled cluster with original equal-case weights. Average5finalseeds first, never treat25models as25independent searches.',
            iterations=10000,bootstrap_seed=510001,family_alpha=.05,contrasts=4,ci_percent=98.75,
            quantiles=[.00625,.99375],quantile_method='linear',
            limits='Approximate bootstrap coverage, only5 independent searches. Known-geometry flowcase split; no unseen-geometry claim. Formal dissertation hypothesis wording remains a later writing task.'),
        failure_policy='Preserve failure and partial outputs, halt new claims for review; no fabricated metrics, no replacement model or hidden retry; comparative report requires155x200valid outcomes',
        preflight='Four GPU validation-only correctness checks against saved VAL metrics before any TEST target file is opened; not a resource-estimation pilot',
        schedule=dict(gpus=4,cpus_per_gpu=4,unique_H200_required=True,wall_limit_hours=8),
        no_training=True,no_new_LLM_proposals=True,no_selection_on_TEST=True,
        raw_arrays_policy='Fields/weights/rawtimings remain onPerun; local metadata/tables/plots only',
        resource_estimation_pilot=False,TEST_opened_at_freeze=False,
        frozen_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat())
    assert len(set(protocol['test_geometries']))==85
    save(FINAL/'protocol.json',protocol)
    (FINAL/'PROTOKOL.txt').write_text('''FINÁLNY TEST SPOLOČNÉHO 2D PINN — 08.11.2026

Všetkých 395 tréningov je dokončených. Vyhodnotí sa 155 finálnych sietí
(6 metód x 5 hľadaní x 5 čerstvých tréningových seedov + 5 fixed baseline)
na 200 doteraz odložených prípadoch. Váhy, konfigurácie, seedy a normalizácia
sú uzavreté v models.json. Už sa nič netrénuje ani nevyberá podľa TEST.

Primárna metrika je priemerná MSE celého poľa u/v/p, rovnako ako pri VAL.
Ďalej sa uložia relatívne chyby rýchlosti a tlaku, rezíduá Navierových–Stokesových
rovníc a divergencie, chyby na stenách a prekážke, aj surové časy výpočtu.
Každý prípad má rovnakú váhu. Pole má natívne rozlíšenie 512 x 512.

Rýchlosť: 30 zahriatí a 200 meraní na každý model/prípad; požiadavka p99 <= 5 ms.
Meria sa celý GPU výpočet s už pripravenými vstupmi na GPU; bez prenosu do PC,
vykreslenia a prípravy statických súradníc. Vstupná geometria sa v každom meraní
znovu spracuje. Nastavenia FP32/TF32 a tlaková referencia sú rovnaké ako vo V33.
Pred prvým otvorením TEST prejdú všetky 4 GPU kontrolou správnosti na VAL.
Ide o overenie výpočtu, nie o používateľom vynechaný pilot odhadu spotreby.

Primárne 4 porovnania: jednotlivé LLM mínus TPE v priemernej TEST MSE.
Najprv sa spriemeruje 5 finálnych seedov v každom hľadaní. Následne párový
crossed bootstrap (10 000 opakovaní, seed 510001) vzorkuje 5 hľadaní a
85 geometrických zhlukov TEST. Použije všetky prípady vybratej geometrie
so zachovanou rovnakou váhou prípadu a rovnaké vzorkovanie pre všetky metódy.
Bonferroniho 98,75 % intervaly zohľadnia 4 primárne porovnania.
Nejde o 25 nezávislých hľadaní. Pokrytie intervalov je približné a len
5 nezávislých hľadaní je obmedzenie. Random, fixed a ostatné metriky sa
uvedú tiež. Formálne znenie hypotéz zostáva na neskoršie písanie dizertácie;
číselný postup sa uzatvára teraz, pred prvým výsledkom TEST.

Nevolí sa najlepší finálny seed. Zlyhanie sa uloží a zastaví ďalšie úlohy,
bez vymyslených hodnôt a skrytého opakovania. Porovnávací report vyžaduje
všetkých 155 x 200 platných výsledkov. TRAIN a TEST zdieľajú geometrie,
preto sa nevyvodzuje schopnosť predikovať úplne nevidené geometrie.
Predikujú sa ustálené polia. Veľké polia, váhy a surové časy zostanú na Perune.
''',encoding='utf-8')
    files=[FINAL/'models.json',FINAL/'protocol.json',FINAL/'PROTOKOL.txt',FINAL/'training_closeout/summary.json']
    files += [ROOT/'test'/p for p in [
        'training/ldc_final_v49.py','training/evaluate_ldc_final_v49.py','training/test_ldc_final_v49.py',
        'analysis/review_ldc_final_v49.py','analysis/collect_ldc_final_v49.py','analysis/prepare_ldc_final_eval_v49.py',
        'scripts/269_ldc_final_test_v49.sbatch','scripts/270_ldc_final_collector_v49.sbatch']]
    save(FINAL/'freeze.json',dict(files={str(p):digest(p) for p in files},
        input_fingerprints={prepared['rawtest_targets_sealed_in']:fingerprint(prepared['rawtest_targets_sealed_in'])},
        original_main_freeze_sha256=digest(WORK/'freeze.json'),TEST_opened=False))
    tasks={m['id']:dict(status='PENDING',model=m) for m in manifest['models']}
    save(FINAL/'evaluation/state.json',dict(tasks=tasks,attention_required=False))
    with locked(POOL:=WORK/'operational_pool_v36/control'):
        path=WORK/'operational_pool_v36/current_group.json'
        save(FINAL/'training_closeout/operational_index_before.json',read(path))
        index=read(path);index['status']='TRAINING_COMPLETE_FINAL_TEST_PREPARED'
        index['final_evaluation']=str(FINAL);save(path,index)
    verify();print('FINAL_PROTOCOL_FROZEN_TEST_NOT_OPENED',digest(FINAL/'freeze.json'),flush=True)


if __name__=='__main__':main()
