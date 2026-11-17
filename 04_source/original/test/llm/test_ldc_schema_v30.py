import json
import unittest
import torch
from test_ldc_contract_v30 import fixture
from ldc_contract_v30 import parse_reply
from ldc_schema_v30 import prepare_tokenizer,build_constraint,CompleteJSON


class CharacterTokenizer:
    eos_token_id=128
    all_special_ids=[128]
    def __len__(self):return 129
    def encode(self,s,**kwargs):return [ord(c) for c in s]
    def decode(self,tokens,**kwargs):return ''.join(chr(int(t)) for t in tokens if int(t)!=128)


class SchemaTests(unittest.TestCase):
    def accepted(self,text):
        f,_=fixture();tok=CharacterTokenizer();allowed=build_constraint(prepare_tokenizer(tok),f);sequence=[48]
        for c in text:
            if ord(c) not in allowed(0,torch.tensor(sequence)):return False
            sequence.append(ord(c))
        return 128 in allowed(0,torch.tensor(sequence))

    def test_real_prefix_bridge_and_stop_at_complete_object(self):
        _,r=fixture();raw=json.dumps(r);self.assertTrue(self.accepted(raw))
        tok=CharacterTokenizer();stop=CompleteJSON(tok,1);tokens=torch.tensor([[48]+tok.encode(raw)])
        self.assertFalse(bool(stop(tokens[:,:-1],None)[0]));self.assertTrue(bool(stop(tokens,None)[0]))

    def test_prevents_ranges_missing_fields_extra_text_and_wrong_categories(self):
        f,r=fixture();r['parameters']['learning_rate']={'min':.0001,'max':.002};self.assertFalse(self.accepted(json.dumps(r)))
        _,r=fixture();del r['evidence_ids'];self.assertFalse(self.accepted(json.dumps(r)))
        _,r=fixture();self.assertFalse(self.accepted(json.dumps(r)+' commentary'))
        r['parameters']['physics_weight']=0;self.assertFalse(self.accepted(json.dumps(r)))

    def test_schema_length_and_independent_numeric_range_guard(self):
        f,r=fixture();r['reason']='x'*121;self.assertFalse(self.accepted(json.dumps(r)))
        _,r=fixture();r['parameters']['learning_rate']=.5
        # The library enforces numeric type, not number ranges; our guard remains authoritative.
        self.assertTrue(parse_reply(json.dumps(r),f)[1])


if __name__=='__main__':unittest.main()
