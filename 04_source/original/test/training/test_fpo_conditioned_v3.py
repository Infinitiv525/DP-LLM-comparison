"""Meaningful analytic checks before any GPU optimizer steps."""
import unittest
import numpy as np
import torch
from fpo_conditioned_v3 import ConditionalPINN, residual, errors
from fpo_common_v3 import coords


class Physics(unittest.TestCase):
    def test_variable_reynolds_poiseuille(self):
        for re in (100., 382., 1000.):
            q = torch.randn(32,3,dtype=torch.float64,requires_grad=True)
            x,y,t = q.unbind(1)
            zero = 0*(x**3+y**3+t**3)
            pred = torch.stack((1-y*y+zero,zero,-2*x/re+zero),1)
            self.assertLess(residual(q,pred,re).abs().max().item(),1e-12)

    def test_time_derivative_pressure_balance(self):
        q=torch.randn(32,3,dtype=torch.float64,requires_grad=True)
        x,y,t=q.unbind(1); zero=0*(x**3+y**3+t**3)
        self.assertLess(residual(q,torch.stack((t+zero,zero,-x+zero),1),382).abs().max().item(),1e-12)

    def test_physical_coordinates(self):
        q=coords(np.array([2,241]),np.array([0,255]),np.array([0,1023]))
        np.testing.assert_allclose(q[0],[4+1/128,6+1/128,0])
        np.testing.assert_allclose(q[1],[20-1/128,10-1/128,11.95],rtol=1e-6)

    def test_pressure_gauge(self):
        rng=np.random.default_rng(12); ref=rng.normal(size=(100,3)); pred=ref.copy(); pred[:,2]+=123
        self.assertLess(errors(pred,ref)['pressure_relative_l2'],1e-12)

    def test_conditioning_and_second_derivatives(self):
        torch.manual_seed(5)
        config=dict(conditional=True,frequencies=[1.,2.],latent=8,width=16,depth=2)
        model=ConditionalPINN(config,[0,0,0],[1,1,1])
        geometry=torch.randn(1,2,32,128)
        c=model.context(geometry,382)
        self.assertFalse(torch.allclose(c,model.context(geometry,900)))
        self.assertFalse(torch.allclose(c,model.context(geometry.flip(-1),382)))
        q=torch.rand(8,3,requires_grad=True)*torch.tensor([16,4,11.95])+torch.tensor([4,6,0])
        pred=model(q,c); loss=residual(q,pred,382).square().mean();loss.backward()
        gradients=[p.grad for p in model.encoder.parameters()]
        self.assertTrue(any(g is not None and g.abs().max()>0 for g in gradients))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))


if __name__=='__main__':
    unittest.main()
