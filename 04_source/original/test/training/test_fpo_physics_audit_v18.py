import unittest
import torch
from fpo_physics_audit_v18 import continuous_terms,grid_terms,time_stencil,temporal_difference
from fpo_direct_curl_v15 import DirectPINN
from fpo_state_v4 import residual


def manufactured(q):
    x,y,t=q.unbind(-1)
    return torch.stack((x*x+t*y+t*t,y*y-t*x+.5*t*t,x*x+y*y),-1)


class AuditTests(unittest.TestCase):
    def test_manufactured_continuous_equations(self):
        torch.manual_seed(18);q=torch.rand(17,3,dtype=torch.float64,requires_grad=True)
        x,y,t=q.unbind(1);value=manufactured(q);u,v,_=value.unbind(1);re=123.
        expected=torch.stack((y+2*t+u*2*x+v*t+2*x-2/re,
                              -x+t-u*t+v*2*y+2*y-2/re),1)
        terms=continuous_terms(q,value,re)
        torch.testing.assert_close(terms['momentum'],expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(terms['divergence'][:,0],2*x+2*y)

    def test_grid_time_stencils_match_polynomial_solution(self):
        y,x=torch.meshgrid(6+(torch.arange(24,dtype=torch.float64)+.5)/64,
                           4+(torch.arange(32,dtype=torch.float64)+.5)/64,indexing='ij')
        def field(t):return manufactured(torch.stack((x,y,torch.full_like(x,t)),2))
        yx=torch.tensor([[3,4],[9,15],[20,27]])
        for t in [.05,1.,11.9]:
            q=torch.stack((x[yx[:,0],yx[:,1]],y[yx[:,0],yx[:,1]],torch.full((3,),t,dtype=torch.float64)),1).requires_grad_()
            exact=continuous_terms(q,manufactured(q),123.)
            for dt in [.025,.0025]:
                times,backward=time_stencil(t,dt)
                temporal=temporal_difference(field(t),field(times[0]),field(times[1]),dt,backward)
                measured=grid_terms(field(t),temporal,yx,123.)
                for k in exact:torch.testing.assert_close(measured[k],exact[k],rtol=1e-9,atol=1e-9)
        for t in [0.,11.91]:
            with self.assertRaises(ValueError):time_stencil(t,.025)

    def test_audit_matches_frozen_training_residual(self):
        torch.manual_seed(18)
        config=dict(state_conditioning=True,width=12,depth=2,latent=4,frequencies=[1.])
        for kind in ['primitive','curl']:
            model=DirectPINN(dict(config,representation=kind),[0.,0.,0.],[1.,1.,1.]).double()
            with torch.no_grad():model.net[-1].weight.normal_(0,.1)
            model.requires_grad_(False)
            raw=torch.randn(16,16,6,dtype=torch.float64)*.1;context=torch.randn(1,5,dtype=torch.float64)
            q=torch.rand(8,3,dtype=torch.float64);q[:,:2]=q[:,:2]*.1+q.new_tensor([4.01,6.01]);q[:,2]+=1;q.requires_grad_()
            value=model(q,context,raw);terms=continuous_terms(q,value,123.)
            training=residual(q,value,123.)
            torch.testing.assert_close(terms['momentum'],training[:,:2],rtol=1e-11,atol=1e-11)
            torch.testing.assert_close(terms['divergence'][:,0],training[:,2],rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
