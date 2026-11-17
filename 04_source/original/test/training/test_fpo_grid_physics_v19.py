import unittest
import torch
from fpo_direct_curl_v15 import DirectPINN
from fpo_step_v5a import grid,conditioning,project_known_constraints
from fpo_physics_audit_v18 import grid_terms
from fpo_grid_physics_v19 import sampled_grid_terms,momentum_loss


class GridPhysicsTests(unittest.TestCase):
    def setup_case(self,kind='curl'):
        torch.manual_seed(19)
        cfg=dict(representation=kind,state_conditioning=True,width=12,depth=2,latent=4,frequencies=[1.])
        model=DirectPINN(cfg,[0.,0.,0.],[1.,1.,1.]).double()
        with torch.no_grad():model.net[-1].weight.normal_(0,.1)
        raw=torch.randn(32,128,6,dtype=torch.float64)*.1
        geometry=torch.randn(1,2,4,16,dtype=torch.float64)
        mask=torch.ones(32,128,dtype=torch.bool);mask[14:18,62:66]=False
        yx=torch.tensor([[5,10],[10,35],[24,103]])
        return model,raw,geometry,mask,yx

    def full_terms(self,model,raw,context,mask,yx,t):
        q=grid('cpu',32,128).double()
        def at(time):
            z=q.clone();z[:,2]=time
            return project_known_constraints(model.grid_prediction(z,context,raw),mask)
        value=at(t);temporal=(at(t+.025)[:,:,:2]-at(t-.025)[:,:,:2])/.05
        return grid_terms(value,temporal,yx,123.,h=.125)

    def test_sparse_physics_matches_actual_projected_grid(self):
        for kind in ['primitive','curl']:
            model,raw,geometry,mask,yx=self.setup_case(kind)
            context=conditioning(model,raw,geometry,123.)
            for t in [.025,1.,11.875]:
                sparse=sampled_grid_terms(model,raw,context,yx[:,0],yx[:,1],torch.full((3,),t,dtype=torch.float64),123.)
                exact=self.full_terms(model,raw,context,mask,yx,t)
                for key in exact:torch.testing.assert_close(sparse[key],exact[key],rtol=1e-9,atol=1e-9)

    def test_independent_times_per_point(self):
        model,raw,geometry,mask,yx=self.setup_case()
        context=conditioning(model,raw,geometry,123.);times=torch.tensor([.05,1.,11.875],dtype=torch.float64)
        sparse=sampled_grid_terms(model,raw,context,yx[:,0],yx[:,1],times,123.)
        for i,t in enumerate(times.tolist()):
            exact=self.full_terms(model,raw,context,mask,yx[i:i+1],t)
            for key in exact:torch.testing.assert_close(sparse[key][i:i+1],exact[key],rtol=1e-9,atol=1e-9)

    def test_parameter_gradient_and_encoder_backprop(self):
        model,raw,geometry,mask,yx=self.setup_case();times=torch.tensor([.05,1.,5.],dtype=torch.float64)
        def objective():
            context=conditioning(model,raw,geometry,123.)
            return momentum_loss(sampled_grid_terms(model,raw,context,yx[:,0],yx[:,1],times,123.),model.std[:2].mean())
        value=objective();value.backward()
        grads=[p.grad for p in model.encoder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        p=model.net[-1].weight;analytic=float(p.grad[0,0]);eps=1e-6
        with torch.no_grad():
            p[0,0]+=eps;plus=float(objective());p[0,0]-=2*eps;minus=float(objective());p[0,0]+=eps
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),delta=max(1e-6,abs(analytic)*1e-4))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
