import unittest
import torch
from fpo_own_v13 import OwnPINN
from fpo_state_v4 import bicubic,residual


class OwnTests(unittest.TestCase):
    def config(self):return dict(state_conditioning=True,width=12,depth=2,latent=8,frequencies=[1.,2.])

    def test_initial_and_cached_consistency(self):
        torch.manual_seed(7);m=OwnPINN(self.config(),[0,0,0],[1,1,1]).double()
        raw=torch.randn(256,1024,6,dtype=torch.float64)*.01
        q=torch.tensor([[6.123,7.123,0.],[9.234,8.234,.5]],dtype=torch.float64)
        context=torch.randn(1,9,dtype=torch.float64);cached=bicubic(raw,q)
        a=m(q,context,raw);b=m(q,context,cached_features=cached)
        torch.testing.assert_close(a,b);torch.testing.assert_close(a[0],cached[0,:3])

    def test_physics_gradient_reaches_encoder_and_every_condition_block(self):
        torch.manual_seed(8);m=OwnPINN(self.config(),[0,0,0],[1,1,1]).double()
        raw=torch.randn(256,1024,6,dtype=torch.float64)*.001
        geo=torch.randn(1,2,32,128,dtype=torch.float64);low=torch.randn(1,6,32,128,dtype=torch.float64)
        c=m.context(geo,low,382)
        q=torch.tensor([[6.123,7.123,.2],[9.234,8.234,.5]],dtype=torch.float64,requires_grad=True)
        residual(q,m(q,c,raw),382).square().mean().backward()
        for group in [m.encoder]+[b.condition for b in m.blocks]:
            gs=[p.grad for p in group.parameters() if p.grad is not None]
            self.assertTrue(gs);self.assertTrue(all(torch.isfinite(g).all() for g in gs));self.assertTrue(any(g.abs().max()>0 for g in gs))

    def test_context_changes_future_but_not_initial(self):
        torch.manual_seed(9);m=OwnPINN(self.config(),[0,0,0],[1,1,1])
        raw=torch.zeros(2,6);q=torch.tensor([[6.,7.,0.],[6.,7.,1.]])
        a=m(q,torch.zeros(1,9),cached_features=raw);b=m(q,torch.ones(1,9),cached_features=raw)
        torch.testing.assert_close(a[0],b[0]);self.assertGreater(float((a[1]-b[1]).abs().max().detach()),0.)


if __name__=='__main__':unittest.main()
