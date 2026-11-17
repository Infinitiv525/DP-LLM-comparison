import unittest
import torch
from fpo_rollout_v7 import own_chunks


class RolloutTests(unittest.TestCase):
    def test_own_states_survive_parameter_update_without_old_graph(self):
        weight=torch.nn.Parameter(torch.tensor(2.))
        initial=torch.tensor(1.,requires_grad=True)
        optimizer=torch.optim.SGD([weight],lr=.1)
        chunks=own_chunks(torch.tensor(0.),initial,lambda a,b:weight*b,4,2)
        start,previous,current,pred=next(chunks)
        torch.testing.assert_close(torch.stack(pred),torch.tensor([2.,4.]))
        sum(pred).backward();torch.testing.assert_close(weight.grad,torch.tensor(5.))
        old_gradient=initial.grad.clone();optimizer.step();optimizer.zero_grad(set_to_none=True)
        start,previous,current,pred=next(chunks)
        self.assertEqual(start,2);self.assertFalse(previous.requires_grad);self.assertFalse(current.requires_grad)
        torch.testing.assert_close(current,torch.tensor(4.))
        torch.testing.assert_close(torch.stack(pred),torch.tensor([6.,9.]))
        sum(pred).backward();torch.testing.assert_close(weight.grad,torch.tensor(16.))
        torch.testing.assert_close(initial.grad,old_gradient)

    def test_chunking_preserves_second_order_trajectory(self):
        for chunk in [1,3,8,10]:
            values=[]
            for _,_,_,pred in own_chunks(torch.tensor(0.),torch.tensor(1.),lambda a,b:2*b-a,10,chunk):values.extend(pred)
            torch.testing.assert_close(torch.stack(values),torch.arange(2.,12.))

    def test_partial_last_block_has_no_padding_or_missing_target(self):
        blocks=list(own_chunks(torch.tensor(0.),torch.tensor(1.),lambda a,b:b+1,10,8))
        self.assertEqual([a[0] for a in blocks],[0,8]);self.assertEqual([len(a[3]) for a in blocks],[8,2])
        self.assertEqual(float(blocks[-1][-1][-1]),11.)

    def test_invalid_lengths_rejected(self):
        for h,c in [(0,8),(8,0),(-1,8)]:
            with self.assertRaises(ValueError):next(own_chunks(torch.tensor(0.),torch.tensor(1.),lambda a,b:b,h,c))


if __name__=='__main__':unittest.main()
