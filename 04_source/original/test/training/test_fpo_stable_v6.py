import unittest
import numpy as np
import torch
from fpo_stable_v6 import StablePINN,curl_grid,divergence_grid,grid_physics,grid,stable_step,conditioning,features,residual


class StableTests(unittest.TestCase):
    def config(self,kind):return dict(state_conditioning=True,state_update=kind,width=8,depth=2,latent=8,frequencies=[1.,2.])

    def test_known_state_mandatory(self):
        with self.assertRaises(ValueError):StablePINN(dict(self.config('curl'),state_conditioning=False),[0,0,0],[1,1,1])

    def test_discrete_curl_commutes(self):
        torch.manual_seed(1);psi=torch.randn(64,128,dtype=torch.float64)
        self.assertLess(float(divergence_grid(curl_grid(psi)).abs().max()),1e-10)

    def test_initial_defaults(self):
        raw=torch.randn(64,128,6);q=grid('cpu',64,128);geo=torch.randn(1,2,8,16);weight=torch.ones(64,128)
        for kind in ['ar2','first_order','curl']:
            model=StablePINN(self.config(kind),[0,0,0],[1,1,1]);c=conditioning(model,raw,geo,382)
            p=model.grid_prediction(q,c,raw,weight)
            expected=raw[:,:,:3]+(raw[:,:,3:] if kind=='ar2' else 0.)
            torch.testing.assert_close(p,expected)

    def test_sparse_matches_grid_even_at_crop_edges(self):
        torch.manual_seed(2);model=StablePINN(self.config('curl'),[0,0,0],[1,1,1]).double()
        with torch.no_grad():model.net[-1].weight.normal_(0,.03)
        raw=torch.randn(64,128,6,dtype=torch.float64);w=torch.rand(64,128,dtype=torch.float64)
        geo=torch.randn(1,2,8,16,dtype=torch.float64);c=conditioning(model,raw,geo,382)
        q=grid('cpu',64,128).double();full=model.grid_prediction(q,c,raw,w)
        y=torch.tensor([0,63,0,63,20]);x=torch.tensor([0,0,127,127,40]);point=q.reshape(64,128,3)[y,x]
        torch.testing.assert_close(model.sparse_prediction(point,c,raw,w,y,x),full[y,x],rtol=1e-9,atol=1e-9)

    def test_masked_curl_preserves_interior_divergence(self):
        torch.manual_seed(3);model=StablePINN(self.config('curl'),[0,0,0],[1,1,1]).double()
        with torch.no_grad():model.net[-1].weight.normal_(0,.03)
        mask=torch.ones(64,128,dtype=torch.bool);mask[28:34,40:46]=False
        current=torch.randn(64,128,3,dtype=torch.float64)*mask[:,:,None];previous=current*.99
        valid=mask.clone()
        for sy in range(-3,4):
            for sx in range(-3,4):valid &= torch.roll(mask,(sy,sx),(0,1))
        valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
        case=dict(geometry=torch.randn(1,2,8,16,dtype=torch.float64),record=dict(reynolds=382),fluid=mask,weight=mask.double())
        pred=stable_step(model,previous,current,case,grid('cpu',64,128).double())
        change=divergence_grid(pred[:,:,:2])-divergence_grid(current[:,:,:2])
        self.assertLess(float(change[valid].abs().max().detach()),1e-9)
        self.assertEqual(float(pred[~mask].abs().max().detach()),0.)

    def test_continuous_curl_and_third_derivative_backprop(self):
        torch.manual_seed(4);model=StablePINN(self.config('curl'),[0,0,0],[1,1,1]).double()
        with torch.no_grad():model.net[-1].weight.normal_(0,.02)
        yy,xx=torch.meshgrid(6+(torch.arange(256,dtype=torch.float64)+.5)/64,4+(torch.arange(1024,dtype=torch.float64)+.5)/64,indexing='ij')
        raw=torch.stack((-.1*(yy-8),.1*(xx-6),0*xx,0*xx,0*xx,0*xx),2)
        w=torch.ones(256,1024,dtype=torch.float64)
        c=conditioning(model,raw,torch.randn(1,2,32,128,dtype=torch.float64),382)
        q=torch.tensor([[7.123,7.623,0.],[12.567,8.345,.027]],dtype=torch.float64,requires_grad=True)
        p=model(q,c,raw,w)
        gu=torch.autograd.grad(p[:,0].sum(),q,create_graph=True)[0]
        gv=torch.autograd.grad(p[:,1].sum(),q,create_graph=True)[0]
        self.assertLess(float((gu[:,0]+gv[:,1]).abs().max().detach()),1e-8)
        torch.testing.assert_close(p[0],torch.tensor([-.1*(7.623-8),.1*(7.123-6),0.],dtype=torch.float64),rtol=1e-9,atol=1e-9)
        residual(q,p,382).square().mean().backward()
        self.assertTrue(all(torch.isfinite(a.grad).all() for a in model.parameters() if a.grad is not None))
        self.assertTrue(any(a.grad is not None and a.grad.abs().max()>0 for a in model.encoder.parameters()))

    def test_own_state_midpoint_physics_known_solutions(self):
        q=grid('cpu').double().reshape(256,1024,3);x,y=q[:,:,0],q[:,:,1]
        valid=torch.ones(256,1024,dtype=torch.bool);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
        for re in [100.,382.,1000.]:
            case=dict(record=dict(reynolds=re),interior_gpu=valid)
            field=torch.stack((1-(y-8)**2,0*x,-2*x/re),2)
            m,d=grid_physics(field,field,case,torch.tensor(1.))
            self.assertLess(float(m),1e-12);self.assertLess(float(d),1e-12)
            old=torch.stack((0*x,0*x,-x),2);new=old.clone();new[:,:,0]=.05
            m,d=grid_physics(old,new,case,torch.tensor(1.))
            self.assertLess(float(m),1e-12);self.assertLess(float(d),1e-12)

    def test_own_state_physics_has_gradient(self):
        q=grid('cpu');f=torch.sin(q[:,0]).reshape(256,1024)
        old=torch.stack((f,f*.1,f*.2),2);new=(old+.01*torch.cos(old)).requires_grad_(True)
        valid=torch.ones(256,1024,dtype=torch.bool);valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
        m,d=grid_physics(old,new,dict(record=dict(reynolds=382),interior_gpu=valid),torch.tensor(.3))
        (m+d).backward();self.assertTrue(torch.isfinite(new.grad).all());self.assertGreater(float(new.grad.abs().max()),0.)

    def test_step_cannot_access_reference_labels(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key in ['data','reference']:raise AssertionError('Reference label accessed by predictor')
                return super().__getitem__(key)
        model=StablePINN(self.config('curl'),[0,0,0],[1,1,1])
        current=torch.randn(64,128,3)
        case=Guard(geometry=torch.randn(1,2,8,16),record=dict(reynolds=382),
                   fluid=torch.ones(64,128,dtype=torch.bool),weight=torch.ones(64,128))
        stable_step(model,current*.99,current,case,grid('cpu',64,128))


if __name__=='__main__':unittest.main()
