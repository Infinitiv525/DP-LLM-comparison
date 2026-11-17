import copy
import json
import unittest
from ldc_contract_v27 import BASELINE,parse_reply,messages,training_config,validate_parameters
from ldc_controls_v27 import RandomSearch,TPESearch


def fixture():
    r=dict(id='common_0',parameters=BASELINE.copy(),score=.02,velocity_error=.9,pressure_error=.8,
        momentum_rms=1.,divergence_rms=.001,training_seconds=50.,training_seed=270001,epochs=2)
    f=dict(contract_version=27,phase='technical_preflight',epochs=2,remaining_attempts=2,latest_required_id='common_0',history=[r])
    reply=dict(parameters=BASELINE.copy(),reason='Measured baseline used.',evidence_ids=['common_0'],expected_effect='Improve fit.',risk='Might not improve.')
    return f,reply


class ContractTests(unittest.TestCase):
    def test_range_and_physics_cannot_be_disabled(self):
        for key,value in [('physics_weight',0),('physics_weight',True),('learning_rate',float('nan')),('bc_weight',-1),('warmup_fraction',.5)]:
            p=BASELINE.copy();p[key]=value
            with self.assertRaises(ValueError):validate_parameters(p)
        with self.assertRaises(ValueError):validate_parameters(dict(BASELINE,width=999))

    def test_transport_does_not_extract_or_change_parameters(self):
        f,r=fixture()
        parsed,errors,transport=parse_reply('```json\n'+json.dumps(r)+'\n```',f)
        self.assertFalse(errors);self.assertEqual(parsed,r);self.assertNotEqual(transport,'strict_json')
        self.assertTrue(parse_reply('Explanation: '+json.dumps(r),f)[1])
        self.assertTrue(parse_reply(json.dumps(r).replace('"parameters":','"parameters": {}, "parameters":',1),f)[1])
        self.assertTrue(parse_reply(json.dumps(r).replace('0.001','NaN',1),f)[1])

    def test_second_proposal_must_cite_real_latest_feedback(self):
        f,r=fixture();row=copy.deepcopy(f['history'][0]);row['id']='qwen_r1';f['history'].append(row);f['latest_required_id']=row['id'];f['remaining_attempts']=1
        self.assertTrue(parse_reply(json.dumps(r),f)[1])
        r['evidence_ids'].append(row['id']);self.assertFalse(parse_reply(json.dumps(r),f)[1])
        f['history'][-1]['score']=float('inf')
        with self.assertRaises(ValueError):messages(f)

    def test_test_labels_and_extra_fields_cannot_enter_prompt(self):
        f,r=fixture();f['test_scores']=[1]
        with self.assertRaises(ValueError):messages(f)
        f,_=fixture();f['history'][0]['test_error']=0
        with self.assertRaises(ValueError):messages(f)

    def test_common_budget_architecture_and_scaled_schedule(self):
        base=dict(width=192,epochs=400,batch_cases=4,data_points=2048,physics_points=64,body_points=64,wall_points_per_side=16)
        short=training_config(base,BASELINE);full=training_config(base,BASELINE,400)
        self.assertEqual((short['warmup'],short['ramp']),(10,10));self.assertEqual((full['warmup'],full['ramp']),(2000,2000))
        for k in ['width','batch_cases','data_points','physics_points','body_points','wall_points_per_side']:
            self.assertEqual(short[k],full[k])

    def test_control_samplers_use_same_space_and_real_tell(self):
        a,b=RandomSearch(42),RandomSearch(42)
        for _ in range(10):self.assertEqual(a.ask(),b.ask())
        f,_=fixture();t=TPESearch(42,f['history']);p=t.ask();validate_parameters(p)
        with self.assertRaises(ValueError):t.ask()
        t.tell(p,.015);self.assertEqual(t.export()['trials'][-1]['score'],.015)


if __name__=='__main__':unittest.main()
