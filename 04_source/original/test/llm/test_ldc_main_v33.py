"""Meaningful guards for sealed feedback, dependency ordering and sampler replay."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from ldc_main_common_v33 import ROOT, read
from ldc_main_contract_v33 import BASELINE,validate_feedback,parse_reply,training_config,messages
from ldc_main_controls_v33 import propose_control
from ldc_controls_v30 import TPESearch,RandomSearch
from ldc_main_scheduler_v33 import make_plan,initialize_state,claim,finish,choose_selection


def history_row(identifier='qwen3_8b_r00_c00',phase='integration'):
    return dict(id=identifier,parameters=BASELINE.copy(),status='COMPLETE',score=.02,velocity_error=.2,
        pressure_error=.3,momentum_rms=.4,divergence_rms=.00001,boundary_error=.02,training_seconds=100.,
        training_seed=320001 if phase=='integration' else 140001,epochs=2 if phase=='integration' else 400,error_codes=[])


def fixture(phase='integration'):
    h=history_row(phase=phase);budget=3 if phase=='integration' else 8
    feedback=dict(contract_version=33,phase=phase,epochs=h['epochs'],candidate_budget=budget,
                  remaining_attempts=budget-1,latest_required_id=h['id'],history=[h])
    reply=dict(parameters=BASELINE.copy(),reason='Measured start.',evidence_ids=[h['id']],
               expected_effect='Test balance.',risk='May not improve.')
    return feedback,reply


def race_worker(directory,index,queue):
    claimed=[]
    for _ in range(500):
        payload,status=claim(directory,index,'unit-fixture')
        if status=='ALL_DONE':break
        if payload is None:time.sleep(.05);continue
        task=payload['task'];row=history_row(task['id']);row['parameters']=BASELINE.copy()
        time.sleep(.003)
        finish(directory,task['id'],index,'unit-fixture',dict(status='COMPLETE',history=row,trial_summary_sha256='unit_fixture_only'))
        claimed.append(task['id'])
    queue.put(claimed)


class ContractTests(unittest.TestCase):
    def test_main_and_integration_budgets_cannot_be_mixed(self):
        for phase in ['integration','main']:
            f,r=fixture(phase);self.assertFalse(parse_reply(json.dumps(r),f)[1])
        f,_=fixture('main');f['epochs']=2
        with self.assertRaises(ValueError):messages(f)

    def test_no_test_or_future_fields(self):
        f,_=fixture();f['test_scores']=[0]
        with self.assertRaises(ValueError):messages(f)
        f,_=fixture();f['history'][0]['test_error']=0
        with self.assertRaises(ValueError):messages(f)

    def test_failure_cannot_be_given_invented_metrics(self):
        f,_=fixture();r=f['history'][0];r['status']='TRAINING_FAILED'
        with self.assertRaises(ValueError):validate_feedback(f)
        for k in ['score','velocity_error','pressure_error','momentum_rms','divergence_rms','boundary_error']:r[k]=None
        r['error_codes']=['NONFINITE_OPTIMIZATION'];validate_feedback(f)

    def test_real_latest_id_and_bounded_parameters_required(self):
        f,r=fixture();r['evidence_ids']=['future']
        self.assertTrue(parse_reply(json.dumps(r),f)[1])
        f,r=fixture();r['parameters']['physics_weight']=0
        self.assertTrue(parse_reply(json.dumps(r),f)[1])
        f,r=fixture();r['parameters']['learning_rate']={'min':.0001,'max':.002}
        self.assertTrue(parse_reply(json.dumps(r),f)[1])

    def test_exact_main_training_and_schedule(self):
        base=dict(width=192,batch_cases=4,data_points=2048,physics_points=64)
        main=training_config(base,BASELINE,'main',140001);tiny=training_config(base,BASELINE,'integration',320001)
        self.assertEqual((main['epochs'],main['assess_epochs'],main['warmup'],main['ramp']),(400,10,2000,2000))
        self.assertEqual((tiny['epochs'],tiny['warmup'],tiny['ramp']),(2,10,10))
        for key in base:self.assertEqual(main[key],tiny[key])

    def test_different_sampling_seed_in_same_search_rejected(self):
        f,_=fixture('main');r=copy.deepcopy(f['history'][0]);r.update(id='next',training_seed=999)
        f['history'].append(r);f['latest_required_id']='next';f['remaining_attempts']=6
        with self.assertRaises(ValueError):validate_feedback(f)


class SamplerTests(unittest.TestCase):
    def test_exact_random_and_tpe_replay(self):
        for method in ['random','tpe']:
            history=[history_row()]
            persistent=RandomSearch(73) if method=='random' else TPESearch(73,history)
            for i in range(3):
                expected=persistent.ask();actual,_=propose_control(method,73,history)
                self.assertEqual(actual,expected)
                value=.02-i*.001;persistent.tell(expected,value)
                r=history_row('fixture_'+str(i));r.update(parameters=expected,score=value);history.append(r)

    def test_replay_detects_changed_past_parameters(self):
        history=[history_row(),history_row('other')]
        with self.assertRaises(ValueError):propose_control('tpe',73,history)


class SchemaTests(unittest.TestCase):
    def test_actual_main_token_bridge_accepts_complete_reply(self):
        import torch
        from test_ldc_schema_v30 import CharacterTokenizer
        from ldc_main_schema_v33 import prepare_tokenizer,build_constraint,CompleteJSON
        f,r=fixture('main');tok=CharacterTokenizer();text=json.dumps(r)
        allowed=build_constraint(prepare_tokenizer(tok),f);tokens=[48]
        for character in text:
            self.assertIn(ord(character),allowed(0,torch.tensor(tokens)));tokens.append(ord(character))
        self.assertIn(128,allowed(0,torch.tensor(tokens)))
        stop=CompleteJSON(tok,1)
        self.assertTrue(bool(stop(torch.tensor([tokens]),None)[0]))

    def test_schema_does_not_accept_range_objects(self):
        import torch
        from test_ldc_schema_v30 import CharacterTokenizer
        from ldc_main_schema_v33 import prepare_tokenizer,build_constraint
        f,r=fixture('main');r['parameters']['learning_rate']={'min':.0001,'max':.002}
        tok=CharacterTokenizer();allowed=build_constraint(prepare_tokenizer(tok),f);tokens=[48];accepted=True
        for character in json.dumps(r):
            if ord(character) not in allowed(0,torch.tensor(tokens)):
                accepted=False;break
            tokens.append(ord(character))
        self.assertFalse(accepted)


class QueueTests(unittest.TestCase):
    def test_plan_counts_and_paired_final_seeds(self):
        for phase,count in [('integration',25),('main',395)]:
            plan=make_plan(phase);self.assertEqual(len(plan),count)
            self.assertEqual(len({t['id'] for t in plan}),count)
        plan=make_plan('main')
        self.assertEqual({t['seed'] for t in plan if t['kind']=='fixed'},
                         {t['seed'] for t in plan if t['kind']=='retrain'})
        self.assertTrue(all(len(t['dependencies'])==8 for t in plan if t['kind']=='retrain'))

    def test_eight_processes_never_double_claim_or_skip_dependency(self):
        runtime=ROOT/'test/runtime';runtime.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='main33_queue_',dir=runtime) as directory:
            plan=make_plan('integration');initialize_state(directory,plan)
            queue=mp.Queue();processes=[mp.Process(target=race_worker,args=(directory,i,queue)) for i in range(8)]
            for p in processes:p.start()
            try:
                deadline=time.monotonic()+180
                claimed=sum((queue.get(timeout=max(.1,deadline-time.monotonic())) for _ in processes),[])
                for p in processes:
                    p.join(max(.1,deadline-time.monotonic()));self.assertEqual(p.exitcode,0)
            finally:
                # Stop and reap children BEFORE the temporary directory cleanup.
                for p in processes:
                    if p.is_alive():p.terminate()
                for p in processes:p.join(10)
                queue.close();queue.join_thread()
            self.assertEqual(len(claimed),25);self.assertEqual(len(set(claimed)),25)
            state=read(Path(directory)/'state.json')
            self.assertTrue(all(r['status']=='DONE' for r in state['tasks'].values()))
            for row in state['tasks'].values():
                for parent in row['task']['dependencies']:
                    self.assertLessEqual(state['tasks'][parent]['finished_unix'],row['claimed_unix'])

    def test_wrong_worker_cannot_complete_task(self):
        with tempfile.TemporaryDirectory(prefix='main33_owner_',dir=ROOT/'test/runtime') as directory:
            initialize_state(directory,make_plan('integration'));payload,_=claim(directory,0,'job')
            with self.assertRaises(ValueError):finish(directory,payload['task']['id'],1,'job',{})

    def test_selection_excludes_failures_and_uses_earliest_tie(self):
        task=dict(dependencies=['a','b','c']);state=dict(tasks={})
        for key,score,status in [('a',.02,'COMPLETE'),('b',None,'TRAINING_FAILED'),('c',.02,'COMPLETE')]:
            row=history_row(key);row.update(score=score,status=status)
            state['tasks'][key]=dict(task=dict(id=key,candidate=ord(key)-ord('a')),
                outcome=dict(status=status,history=row,trial_summary_sha256=key))
        self.assertEqual(choose_selection(state,task)['source_task'],'a')


if __name__=='__main__':unittest.main()
