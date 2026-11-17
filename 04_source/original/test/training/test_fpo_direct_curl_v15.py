import unittest
import torch
from fpo_direct_curl_v15 import DirectPINN
from fpo_stable_v6 import grid,conditioning,divergence_grid,residual
from fpo_step_v5a import project_known_constraints


class DirectTests(unittest.TestCase):
    def model(self,kind):return DirectPINN(dict(state_conditioning=True,width=8,depth=2,latent=8,frequencies=[1.,2.],representation=kind),[0,0,0],[1,1,1]).double()

    def test_full_vs_sampled_stencils_and_initial(self):
        torch.manual_seed(1);raw=torch.randn(64,128,6,dtype=torch.float64)*.01
        geo=torch.randn(1,2,8,16,dtype=torch.float64);y=torch.tensor([0,63,0,63,24]);x=torch.tensor([0,0,127,127,54])
        for kind in ['primitive','curl']:
            m=self.model(kind)
            with torch.no_grad():m.net[-1].weight.normal_(0,.02)
            c=conditioning(m,raw,geo,382)
            for t in [0.,.05,11.9]:
                q=grid('cpu',64,128).double();q[:,2]=t
                full=m.grid_prediction(q,c,raw);sample=m.sampled_prediction(q.reshape(64,128,3)[y,x],c,raw,y,x)
                torch.testing.assert_close(sample,full[y,x],rtol=1e-9,atol=1e-9)
                if t==0:torch.testing.assert_close(full,raw[:,:,:3])

    def test_masked_curl_increment_keeps_initial_interior_divergence(self):
        torch.manual_seed(2);m=self.model('curl')
        with torch.no_grad():m.net[-1].weight.normal_(0,.02)
        fluid=torch.ones(64,128,dtype=torch.bool);fluid[28:34,40:46]=False
        current=torch.randn(64,128,3,dtype=torch.float64)*fluid[:,:,None];raw=torch.cat((current,current*.01),2)
        c=conditioning(m,raw,torch.randn(1,2,8,16,dtype=torch.float64),382)
        q=grid('cpu',64,128).double();q[:,2]=11.9
        p=project_known_constraints(m.grid_prediction(q,c,raw),fluid)
        valid=fluid.clone()
        for dy in range(-3,4):
            for dx in range(-3,4):valid &= torch.roll(fluid,(dy,dx),(0,1))
        valid[:3]=False;valid[-3:]=False;valid[:,:3]=False;valid[:,-3:]=False
        difference=divergence_grid(p[:,:,:2])-divergence_grid(current[:,:,:2])
        self.assertLess(float(difference[valid].abs().max().detach()),1e-9)

    def test_continuous_curl_third_derivative_gradient(self):
        torch.manual_seed(3);m=self.model('curl')
        with torch.no_grad():m.net[-1].weight.normal_(0,.02)
        yy,xx=torch.meshgrid(6+(torch.arange(256,dtype=torch.float64)+.5)/64,4+(torch.arange(1024,dtype=torch.float64)+.5)/64,indexing='ij')
        raw=torch.stack((-.1*(yy-8),.1*(xx-6),0*xx,0*xx,0*xx,0*xx),2)
        c=conditioning(m,raw,torch.randn(1,2,32,128,dtype=torch.float64),382)
        q=torch.tensor([[7.123,7.623,.2],[12.567,8.345,5.]],dtype=torch.float64,requires_grad=True)
        p=m(q,c,raw);gu=torch.autograd.grad(p[:,0].sum(),q,create_graph=True)[0];gv=torch.autograd.grad(p[:,1].sum(),q,create_graph=True)[0]
        self.assertLess(float((gu[:,0]+gv[:,1]).abs().max().detach()),1e-8)
        residual(q,p,382).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in m.encoder.parameters()))


if __name__=='__main__':unittest.main()
