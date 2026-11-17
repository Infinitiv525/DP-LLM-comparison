import unittest
import numpy as np
import torch
from fpo_step_v5 import StepPINN,grid,step,trajectory,project_known_constraints,residual


class StepTests(unittest.TestCase):
    def config(self):return dict(width=8,depth=2,latent=8,frequencies=[1.,2.],state_conditioning=True)

    def test_requires_initial_state(self):
        with self.assertRaises(ValueError):StepPINN(dict(self.config(),state_conditioning=False),[0,0,0],[1,1,1])

    def test_causal_state_machine(self):
        inputs=[]
        def next_step(a,b):inputs.append((a,b));return 2*b-a
        self.assertEqual(list(trajectory(1,2,next_step,4)),[3,4,5,6])
        self.assertEqual(inputs,[(1,2),(2,3),(3,4),(4,5)])

    def test_transition_indices_cover_retained_interval(self):
        k=list(range(1,240-1))
        self.assertEqual(len(k),238)
        self.assertEqual((k[0]-1+2,k[0]+2,k[0]+1+2),(2,3,4))
        self.assertEqual(k[-1]+1+2,241)
        self.assertEqual(list(range(1,240-4))[-1]+4+2,241)

    def test_zero_correction_linear_prediction_and_initial_anchor(self):
        torch.manual_seed(4);model=StepPINN(self.config(),[0,0,0],[1,1,1])
        raw=torch.randn(3,6);c=model.context(torch.randn(1,2,8,16),torch.randn(1,6,8,16),382)
        q=torch.tensor([[7.,8.,0.],[9.,7.,.025],[12.,9.,.05]])
        actual=model(q,c,cached_features=raw)
        torch.testing.assert_close(actual,raw[:,:3]+q[:,2:3]/.05*raw[:,3:])

    def test_full_step_uses_geometry_and_own_pressure_gauge(self):
        model=StepPINN(self.config(),[0,0,0],[1,1,1]);prev=torch.randn(64,128,3)*.1;curr=prev*1.1
        mask=torch.ones(64,128,dtype=torch.bool);mask[20:23,30:35]=False
        geo=torch.randn(1,2,8,16)
        expected=project_known_constraints(2*curr-prev,mask)
        actual=step(model,prev,curr,geo,382,mask,grid('cpu',64,128))
        torch.testing.assert_close(actual,expected)
        self.assertLess(abs(float(actual[:,:,2][mask].mean())),1e-6)
        self.assertEqual(float(actual[~mask].abs().max()),0.)

    def test_local_physical_pde_backward(self):
        model=StepPINN(self.config(),[0,0,0],[1,1,1])
        with torch.no_grad():model.net[-1].weight.fill_(.01)
        field=torch.randn(256,1024,6)*.001
        c=model.context(torch.randn(1,2,8,16),torch.randn(1,6,8,16),382)
        q=torch.tensor([[7.123,7.623,.017],[12.567,8.345,.039]],requires_grad=True)
        loss=residual(q,model(q,c,field),382).square().mean();loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in model.encoder.parameters()))


if __name__=='__main__':unittest.main()
