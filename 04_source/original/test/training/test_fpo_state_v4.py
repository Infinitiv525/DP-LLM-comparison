"""Interpolation derivatives, initial condition, caching equivalence and PDE gradients."""
import unittest
import sys
from pathlib import Path
import numpy as np
import torch
from fpo_state_v4 import StatePINN,bicubic,residual
from fpo_common_v4 import coords
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from prepare_fpo_state_v4 import prefix_features


class StateChecks(unittest.TestCase):
    def test_prefix_reader_cannot_access_future(self):
        seen=[]
        class Reader:
            def __getitem__(self,k):
                if k not in [2,3]:raise AssertionError('Future frame accessed')
                seen.append(k);a=np.full((256,1024,3),float(k));a[:,:,2]=1234+k
                return a
        features,low=prefix_features(Reader(),np.ones((256,1024),dtype=bool))
        self.assertEqual(seen,[2,3])
        np.testing.assert_allclose(features[0,0],[3,3,0,1,1,0])
        self.assertEqual(low.shape,(6,32,128))

    def test_quadratic_interpolation_derivatives(self):
        y,x=torch.meshgrid(6+(torch.arange(256,dtype=torch.float64)+.5)/64,
                           4+(torch.arange(1024,dtype=torch.float64)+.5)/64,indexing='ij')
        field=torch.stack((x*x+y*y,x*y,x+2*y),-1)
        q=torch.tensor([[7.123,7.623,.7],[12.567,8.345,2.]],dtype=torch.float64,requires_grad=True)
        value=bicubic(field,q)
        torch.testing.assert_close(value[:,0],q[:,0]**2+q[:,1]**2,rtol=1e-10,atol=1e-10)
        g=torch.autograd.grad(value[:,0].sum(),q,create_graph=True)[0]
        torch.testing.assert_close(g[:,:2],2*q[:,:2],rtol=1e-9,atol=1e-9)
        lap=torch.autograd.grad(g[:,0].sum(),q,retain_graph=True)[0][:,0]+torch.autograd.grad(g[:,1].sum(),q)[0][:,1]
        torch.testing.assert_close(lap,torch.full_like(lap,4.),rtol=1e-7,atol=1e-7)

    def test_cell_centre_cache_equivalence_and_initial_anchor(self):
        torch.manual_seed(3)
        cfg=dict(width=16,depth=2,latent=8,frequencies=[1.,2.],state_conditioning=True)
        model=StatePINN(cfg,[.8,.1,.2],[.4,.2,.5])
        state=torch.randn(256,1024,6);geo=torch.randn(1,2,32,128);low=torch.randn(1,6,32,128)
        c=model.context(geo,low,382)
        y=np.array([12,25,40]);x=np.array([16,29,35]);q=torch.tensor(coords(np.array([3,3,3]),y,x))
        raw=state[y,x]
        torch.testing.assert_close(model(q,c,state),raw[:,:3],rtol=1e-6,atol=1e-6)
        q[:,2]=.7
        torch.testing.assert_close(model(q,c,state),model(q,c,cached_features=raw),rtol=1e-6,atol=1e-6)

    def test_no_state_control_cannot_read_flow_input(self):
        torch.manual_seed(3)
        model=StatePINN(dict(width=16,depth=2,latent=8,frequencies=[1.],state_conditioning=False),[0,0,0],[1,1,1])
        geo=torch.randn(1,2,32,128)
        a=model.context(geo,torch.randn(1,6,32,128),382)
        b=model.context(geo,torch.randn(1,6,32,128)*1000,382)
        torch.testing.assert_close(a,b)
        q=torch.tensor([[7.,8.,1.]])
        torch.testing.assert_close(model(q,a,None),model(q,b,torch.randn(256,1024,6)))

    def test_pde_second_derivatives_reach_encoder(self):
        torch.manual_seed(3)
        model=StatePINN(dict(width=16,depth=2,latent=8,frequencies=[1.],state_conditioning=True),[0,0,0],[1,1,1])
        c=model.context(torch.randn(1,2,32,128),torch.randn(1,6,32,128),382)
        q=torch.tensor([[7.123,7.623,.7],[12.567,8.345,2.]],requires_grad=True)
        field=torch.randn(256,1024,6)*.01
        loss=residual(q,model(q,c,field),382).square().mean();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in model.encoder.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
