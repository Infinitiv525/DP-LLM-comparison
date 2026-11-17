import unittest
import torch
from ldc_shared_pinn_v26 import SharedSteadyPINN,regular_grid,steady_residual


class SteadyTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(26)
        config=dict(width=12,depth=2,latent=4,frequencies=[1.,2.])
        normal=dict(target_std=[1.,1.,1.],geometry_mean=[0.,0.],geometry_std=[1.,1.])
        model=SharedSteadyPINN(config,normal).double()
        with torch.no_grad():model.net[-1].weight.normal_(0,.05)
        return model,torch.randn(1,2,64,64,dtype=torch.float64)

    def test_native_and_sparse_curl_agree_including_corners(self):
        model,geometry=self.fixture();context=model.context(geometry,200.)
        q=regular_grid(32,32,'cpu',torch.float64);whole=model.grid_prediction(q,context,32,32)
        y=torch.tensor([0,0,31,31,10,20]);x=torch.tensor([0,31,0,31,17,12])
        torch.testing.assert_close(model.sampled_prediction(q[y*32+x],context,y,x,32,32),whole[y,x],rtol=1e-10,atol=1e-10)
        u,v=whole[:,:,0],whole[:,:,1]
        div=(torch.roll(u,-1,1)-torch.roll(u,1,1)+torch.roll(v,-1,0)-torch.roll(v,1,0))*16
        torch.testing.assert_close(div[3:-3,3:-3],torch.zeros_like(div[3:-3,3:-3]),rtol=0,atol=1e-10)

    def test_true_steady_physics_derivatives_reach_encoder(self):
        model,geometry=self.fixture();context=model.context(geometry,200.)
        q=torch.tensor([[.213,.317],[.387,.421],[.631,.781]],dtype=torch.float64,requires_grad=True)
        loss=steady_residual(q,model(q,context),200.).square().mean();loss.backward()
        grad=model.encoder[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0.)

    def test_analytic_steady_couette_and_no_time_input(self):
        q=torch.tensor([[.2,.3],[.5,.6]],dtype=torch.float64,requires_grad=True)
        value=torch.stack((q[:,1],q[:,0]*0,q[:,0]*0),1)
        torch.testing.assert_close(steady_residual(q,value,100.),torch.zeros(2,3,dtype=torch.float64),rtol=0,atol=0)
        model,geometry=self.fixture()
        with self.assertRaises(ValueError):model.heads(torch.zeros(2,3,dtype=torch.float64),model.context(geometry,100.))

    def test_pressure_gradient_balances_nonzero_viscosity(self):
        q=torch.tensor([[.2,.3],[.5,.6]],dtype=torch.float64,requires_grad=True)
        value=torch.stack((q[:,1]*(1-q[:,1]),q[:,0]*0,-2*q[:,0]/100),1)
        torch.testing.assert_close(steady_residual(q,value,100.),torch.zeros(2,3,dtype=torch.float64),rtol=0,atol=1e-14)
        self.assertGreater(float(steady_residual(q,value,200.).abs().max().detach()),.009)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
