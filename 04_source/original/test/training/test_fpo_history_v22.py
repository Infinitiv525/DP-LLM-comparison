import unittest
import numpy as np
import torch
from fpo_history_v22 import (HistoryPINN, conditioning, attach_history,
    read_known_history, target_frame, TARGET_COUNT, END_TIME, pool_history)
from fpo_spatial_context_v21 import SpatialPINN
from fpo_step_v5a import grid, conditioning as old_conditioning
from fpo_state_v4 import residual


class HistoryTests(unittest.TestCase):
    def fixture(self, enabled):
        torch.manual_seed(22)
        cfg=dict(representation='curl',local_context=True,history_enabled=enabled,
            state_conditioning=True,width=12,depth=2,latent=4,frequencies=[1.])
        model=HistoryPINN(cfg,[.1,.2,.3],[1.,2.,3.]).double()
        history=torch.randn(20,64,256,3,dtype=torch.float64)*.1
        case=attach_history(dict(geometry=torch.randn(1,2,8,32,dtype=torch.float64),
            record={'reynolds':123.},fluid=torch.ones(64,256,dtype=torch.bool)),history)
        return model,case,cfg

    def test_no_future_validation_reads_and_gauge(self):
        class GuardedReference:
            def __init__(self):self.reads=[]
            def __getitem__(self, key):
                if not isinstance(key,slice) or key.start!=2 or key.stop!=22:
                    raise AssertionError('Reading beyond declared prefix')
                self.reads.append((key.start,key.stop))
                a=np.ones((20,8,16,3),dtype=np.float64)
                a[:,:,:,2]=np.arange(20)[:,None,None]+2
                return a
        ref=GuardedReference();fluid=np.ones((8,16),dtype=bool);fluid[3,4]=False
        history=read_known_history(ref,fluid,'cpu')
        self.assertEqual(ref.reads,[(2,22)])
        self.assertEqual(history.shape,(20,8,16,3))
        self.assertTrue(torch.all(history[:,:, :,2]==0))
        self.assertTrue(torch.all(history[:,3,4]==0))
        self.assertEqual([target_frame(k) for k in range(1,TARGET_COUNT+1)],list(range(22,242)))
        self.assertEqual(END_TIME,11.)
        for bad in [0,221]:
            with self.assertRaises(ValueError):target_frame(bad)

    def test_pooling_and_history_channel_order(self):
        model,c,cfg=self.fixture(True)
        low=pool_history(c['history'])
        torch.testing.assert_close(low[0,:,0,0],c['history'][0,:8,:8].mean((0,1)))
        captured=[]
        h=model.encoder[0].register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        conditioning(model,c);h.remove()
        expected=(low[:-2]-low[-1:])/model.std[None,:,None,None]
        torch.testing.assert_close(captured[0][:,8:],expected.reshape(1,54,8,32))

    def test_source_preservation_and_hard_anchor(self):
        model,c,cfg=self.fixture(True)
        source=SpatialPINN(cfg,model.mean.tolist(),model.std.tolist()).double()
        with torch.no_grad():source.net[-1].weight.normal_(0,.1)
        model.load_source(source.state_dict())
        now=conditioning(model,c)
        old=old_conditioning(source,c['raw'],c['geometry'],123.)
        q=grid('cpu',64,256).double();q[:,2]=1.
        torch.testing.assert_close(model.grid_prediction(q,now,c['raw']),source.grid_prediction(q,old,c['raw']),rtol=1e-10,atol=1e-10)
        q[:,2]=0.
        torch.testing.assert_close(model.grid_prediction(q,now,c['raw']),c['history'][-1],rtol=0,atol=0)

    def test_control_ignores_older_history_and_active_uses_it(self):
        for enabled in [False,True]:
            model,c,cfg=self.fixture(enabled)
            with torch.no_grad():
                model.encoder[0].weight[:,8:].normal_(0,.1)
                model.net[-1].weight.normal_(0,.1)
                model.net[0].weight[:,model.original_input_features:].normal_(0,.1)
            first=conditioning(model,c)
            c['history']=c['history'].clone();c['history'][:-2]+=1.
            second=conditioning(model,c)
            difference=float((first['spatial']-second['spatial']).detach().abs().max())
            self.assertGreater(difference,1e-6) if enabled else self.assertEqual(difference,0.)

    def test_active_history_grid_sparse_and_physics_gradients(self):
        model,c,cfg=self.fixture(True)
        with torch.no_grad():
            model.encoder[0].weight[:,8:].normal_(0,.1)
            model.net[-1].weight.normal_(0,.1)
            model.net[0].weight[:,model.original_input_features:].normal_(0,.1)
        context=conditioning(model,c);q=grid('cpu',64,256).double();q[:,2]=1.
        whole=model.grid_prediction(q,context,c['raw'])
        y=torch.tensor([0,63,20,35]);x=torch.tensor([255,0,60,160])
        torch.testing.assert_close(model.sampled_prediction(q[y*256+x],context,c['raw'],y,x),whole[y,x],rtol=1e-10,atol=1e-10)
        qc=torch.tensor([[4.13,6.15,.5],[4.2,6.2,1.],[4.3,6.3,3.]],dtype=torch.float64,requires_grad=True)
        loss=residual(qc,model(qc,context,c['raw']),123.).square().mean();loss.backward()
        gradient=model.encoder[0].weight.grad[:,8:]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0.)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
