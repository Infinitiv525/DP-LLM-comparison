import unittest
import torch
from fpo_spatial_v10 import SpatialPINN,expanded_features,features,conditioning,stable_step
from fpo_stable_v6 import StablePINN,grid,stable_step as parent_step,residual


class SpatialTests(unittest.TestCase):
    def test_pretrained_mapping_and_continuous_gradients(self):
        torch.manual_seed(8)
        config=dict(state_conditioning=True,state_update='curl',width=8,depth=2,latent=8,frequencies=[1.,2.])
        parent=StablePINN(config,[0,0,0],[1,1,1])
        with torch.no_grad():parent.net[-1].weight.normal_(0,.02)
        current=torch.randn(256,1024,3)*.01;previous=current*.99
        case=dict(geometry=torch.randn(1,2,32,128),record=dict(reynolds=382),fluid=torch.ones(256,1024,dtype=torch.bool),weight=torch.ones(256,1024))
        with torch.no_grad():base=parent_step(parent,previous,current,case,grid('cpu'))
        for mode in ['zeros','fd']:
            model=SpatialPINN(dict(config,spatial_mode=mode),[0,0,0],[1,1,1]);model.load_state_dict(parent.state_dict())
            with torch.no_grad():torch.testing.assert_close(stable_step(model,previous,current,case,grid('cpu')),base,atol=1e-6,rtol=1e-5)
            raw=features(previous,current);c=conditioning(model,raw,case['geometry'],382)
            q=torch.tensor([[7.13,7.22,.025],[12.12,8.23,.05]],requires_grad=True)
            residual(q,model(q,c,raw,case['weight']),382).square().mean().backward()
            extra=model.net[0].weight.grad[:,-9:]
            self.assertTrue(torch.isfinite(extra).all())
            if mode=='zeros':self.assertEqual(float(extra.abs().max()),0.)
            else:self.assertGreater(float(extra.abs().max()),0.)

    def test_manufactured_spatial_features(self):
        q=grid('cpu').double().reshape(256,1024,3);x,y=q[:,:,0],q[:,:,1]
        f=torch.stack((2*x+3*y,x*x,y*y),2);raw=expanded_features(f,f,'fd')
        torch.testing.assert_close(raw[3:-3,3:-3,6],torch.full_like(x[3:-3,3:-3],.2))
        torch.testing.assert_close(raw[3:-3,3:-3,9],torch.full_like(x[3:-3,3:-3],.3))
        torch.testing.assert_close(raw[3:-3,3:-3,13:],torch.full_like(raw[3:-3,3:-3,13:],.02))


if __name__=='__main__':unittest.main()
