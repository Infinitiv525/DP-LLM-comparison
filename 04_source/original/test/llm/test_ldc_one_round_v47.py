"""CPU-only policy tests; no real claims, training or scheduler calls."""
import unittest
from worker_ldc_one_round_v47 import OneRoundClaim


class OneRoundTests(unittest.TestCase):
    def test_never_claims_a_second_task(self):
        calls=[]
        payload={'task':{'id':'fixed_k00'},'selection':None}
        gate=OneRoundClaim(lambda *args:(calls.append(args) or payload,'CLAIMED'))
        self.assertEqual(gate('d',0,'1'),(payload,'CLAIMED'))
        self.assertEqual(gate('d',0,'1'),(None,'ALL_DONE'))
        self.assertEqual(len(calls),1)

    def test_waiting_does_not_hold_idle_gpu(self):
        gate=OneRoundClaim(lambda *args:(None,'WAITING'))
        self.assertEqual(gate(),(None,'ALL_DONE'))
        self.assertIsNone(gate.task)

    def test_attention_is_preserved(self):
        gate=OneRoundClaim(lambda *args:(None,'ATTENTION_REQUIRED'))
        self.assertEqual(gate(),(None,'ATTENTION_REQUIRED'))

    def test_payload_is_passed_through_without_changes(self):
        payload={'task':{'id':'qwen3_8b_r04_k04','seed':150005},
                 'selection':{'parameters':{'learning_rate':.001}},'history':[]}
        gate=OneRoundClaim(lambda *args:(payload,'CLAIMED'))
        returned,status=gate()
        self.assertIs(returned,payload)
        self.assertEqual(status,'CLAIMED')


if __name__=='__main__':unittest.main()
