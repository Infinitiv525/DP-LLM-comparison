import unittest
import torch
from fpo_gradient_v8 import matched_window


class MatchedGradientTests(unittest.TestCase):
    def rollout(self,count,horizon):
        weight=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));blocks=[];predictions=[]
        def objective(offset,previous,current,pred):
            blocks.append(offset);predictions.extend(p.detach().clone() for p in pred)
            return torch.stack(pred).mean()
        previous,current,loss=matched_window(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64),
                                              lambda a,b:weight*b,objective,count,horizon)
        loss.backward()
        return previous,current,loss.detach(),weight.grad,blocks,torch.stack(predictions)

    def test_same_values_and_objective_different_exact_gradient(self):
        short=self.rollout(32,8);long=self.rollout(32,32)
        torch.testing.assert_close(short[2],long[2],rtol=0,atol=0)
        torch.testing.assert_close(short[5],long[5],rtol=0,atol=0)
        self.assertEqual(short[4],long[4]);self.assertEqual(short[4],[0,8,16,24])
        torch.testing.assert_close(short[3],torch.tensor(4.5,dtype=torch.float64))
        torch.testing.assert_close(long[3],torch.tensor(16.5,dtype=torch.float64))

    def test_partial14_steps_weighted_without_padding(self):
        short=self.rollout(14,8);long=self.rollout(14,32)
        self.assertEqual(len(short[5]),14);self.assertEqual(short[4],[0,8])
        torch.testing.assert_close(short[2],torch.tensor(1.,dtype=torch.float64))
        torch.testing.assert_close(short[3],torch.tensor(57/14,dtype=torch.float64))
        torch.testing.assert_close(long[3],torch.tensor(7.5,dtype=torch.float64))

    def test_no_gradient_graph_leaks_between_optimizer_updates(self):
        for horizon in [8,32]:
            w=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.SGD([w],lr=.01)
            previous,current=torch.tensor(0.),torch.tensor(1.)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                previous,current,loss=matched_window(previous,current,lambda a,b:w*b,
                            lambda o,a,b,p:torch.stack(p).mean(),32,horizon)
                self.assertFalse(previous.requires_grad);self.assertFalse(current.requires_grad)
                loss.backward();self.assertTrue(torch.isfinite(w.grad));optimizer.step()

    def test_same_forward_and_loss_for_nonconstant_signal(self):
        results=[]
        for horizon in [8,32]:
            w=torch.nn.Parameter(torch.tensor(.91,dtype=torch.float64))
            result=matched_window(torch.tensor(.1,dtype=torch.float64),torch.tensor(.2,dtype=torch.float64),
                   lambda a,b:w*b+.03*a+.01,
                   lambda o,a,b,p:torch.stack([v.square() for v in p]).mean(),32,horizon)
            results.append([v.detach() for v in result])
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_lengths(self):
        for count,h in [(0,8),(33,32),(8,16)]:
            with self.assertRaises(ValueError):matched_window(torch.tensor(0.),torch.tensor(1.),lambda a,b:b,lambda o,a,b,p:p[0],count,h)


if __name__=='__main__':unittest.main()
