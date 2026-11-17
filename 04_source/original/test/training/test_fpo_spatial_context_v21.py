import unittest
import torch
from fpo_spatial_context_v21 import SpatialPINN,spline_points,spline_numerators,axis_weights
from fpo_direct_curl_v15 import DirectPINN
from fpo_step_v5a import grid,conditioning
from fpo_state_v4 import residual


class SpatialContextTests(unittest.TestCase):
    def fixture(self,enabled=True):
        torch.manual_seed(21)
        cfg=dict(representation='curl',local_context=enabled,state_conditioning=True,width=12,depth=2,latent=4,frequencies=[1.])
        model=SpatialPINN(cfg,[0.,0.,0.],[1.,1.,1.]).double()
        raw=torch.randn(64,256,6,dtype=torch.float64)*.1
        geometry=torch.randn(1,2,8,32,dtype=torch.float64)
        return model,raw,geometry,cfg

    def test_partition_constant_and_three_derivatives_at_seams(self):
        numerators=torch.tensor(spline_numerators())
        x=torch.tensor([4.2,5.,7.3,12.5,19.9],dtype=torch.float64,requires_grad=True)
        ids,w=axis_weights(x,4.,16.,8,numerators)
        torch.testing.assert_close(w.sum(1),torch.ones_like(x),rtol=1e-12,atol=1e-12)
        feature=torch.full((3,8,8),2.,dtype=torch.float64)
        q=torch.stack((x,x*0+7.,x*0+1.),1)
        torch.testing.assert_close(spline_points(feature,q,numerators),torch.full((5,3),2.,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        torch.manual_seed(21);feature=torch.randn(2,8,8,dtype=torch.float64)
        def derivatives(position):
            q=torch.tensor([[position,7.13,1.]],dtype=torch.float64,requires_grad=True)
            value=spline_points(feature,q,numerators).sum();values=[float(value.detach())]
            for _ in range(3):
                value=torch.autograd.grad(value,q,create_graph=True)[0][0,0];values.append(float(value.detach()))
            return torch.tensor(values,dtype=torch.float64)
        # A coarse-cell knot: interpolation is smooth through the third derivative.
        torch.testing.assert_close(derivatives(9.-1e-7),derivatives(9.+1e-7),rtol=1e-5,atol=1e-5)

    def test_separable_grid_matches_point_sampler(self):
        model,raw,geometry,cfg=self.fixture();context=conditioning(model,raw,geometry,123.)
        q=grid('cpu',64,256).double()
        feature=model.grid_features(context['spatial'],64,256)
        indices=torch.tensor([0,255,63*256,64*256-1,13*256+45,32*256+128])
        torch.testing.assert_close(feature.reshape(-1,16)[indices],spline_points(context['spatial'],q[indices],model.spline_numerators),rtol=1e-11,atol=1e-11)

    def test_zero_extension_preserves_source_and_hard_initial_state(self):
        model,raw,geometry,cfg=self.fixture();source=DirectPINN(cfg,[0.,0.,0.],[1.,1.,1.]).double()
        with torch.no_grad():source.net[-1].weight.normal_(0,.1)
        model.load_source(source.state_dict());q=grid('cpu',64,256).double();q[:,2]=1.
        c=conditioning(model,raw,geometry,123.);old=conditioning(source,raw,geometry,123.)
        torch.testing.assert_close(model.grid_prediction(q,c,raw),source.grid_prediction(q,old,raw),rtol=1e-10,atol=1e-10)
        q[:,2]=0.
        torch.testing.assert_close(model.grid_prediction(q,c,raw),raw[:,:,:3],rtol=0,atol=0)

    def test_nonzero_features_sparse_grid_and_curl_continuity(self):
        model,raw,geometry,cfg=self.fixture()
        with torch.no_grad():
            model.net[0].weight[:,model.original_input_features:].normal_(0,.1)
            model.net[-1].weight.normal_(0,.1)
        context=conditioning(model,raw,geometry,123.);q=grid('cpu',64,256).double();q[:,2]=1.
        value=model.grid_prediction(q,context,raw);yx=torch.tensor([[0,0],[0,255],[63,0],[63,255],[20,60],[35,160]])
        y,x=yx.T
        torch.testing.assert_close(model.sampled_prediction(q[y*256+x],context,raw,y,x),value[y,x],rtol=1e-10,atol=1e-10)
        delta=value[:,:,:2]-raw[:,:,:2]
        div=(torch.roll(delta[:,:,0],-1,1)-torch.roll(delta[:,:,0],1,1))/(2*16/256)+(torch.roll(delta[:,:,1],-1,0)-torch.roll(delta[:,:,1],1,0))/(2*4/64)
        torch.testing.assert_close(div[3:-3,3:-3],torch.zeros_like(div[3:-3,3:-3]),rtol=0,atol=1e-11)

    def test_continuous_ns_third_derivatives_reach_spatial_encoder(self):
        model,raw,geometry,cfg=self.fixture()
        with torch.no_grad():
            model.net[0].weight[:,:model.original_input_features].zero_()
            model.net[0].weight[:,model.original_input_features:].normal_(0,.1)
            model.net[-1].weight.normal_(0,.1)
        q=torch.tensor([[4.13,6.15,.5],[4.2,6.2,1.],[4.3,6.3,3.]],dtype=torch.float64,requires_grad=True)
        context=conditioning(model,raw,geometry,123.)
        loss=residual(q,model(q,context,raw),123.).square().mean();loss.backward()
        gradients=[p.grad for layer in list(model.encoder)[:6] for p in layer.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
