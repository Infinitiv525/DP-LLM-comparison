import unittest
import numpy as np
from ldc_data_v25 import standard_split,input_fields,canonical_target,reference_physics


class DataTests(unittest.TestCase):
    def test_standard_counts_disjoint_reproducible(self):
        split=standard_split();self.assertEqual([len(split[k]) for k in ['train','validation','test']],[640,160,200])
        sets=[set(split[k]) for k in split];self.assertEqual(set.union(*sets),set(range(1000)))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);self.assertEqual(split,standard_split())

    def test_input_channels_detect_actual_order_and_sign(self):
        y,x=np.mgrid[:32,:32];sdf=np.sqrt((x-16)**2+(y-16)**2)-5
        a=np.stack((np.full((32,32),100.),sdf,(sdf>0)*255.))
        re,fluid,s,meta=input_fields(a);self.assertEqual(re,100.);self.assertTrue(fluid[0,0]);self.assertFalse(fluid[16,16])
        self.assertEqual(meta['sdf_channel'],1)
        a=a[[0,2,1]];_,other,_,meta=input_fields(a);np.testing.assert_array_equal(fluid,other);self.assertEqual(meta['sdf_channel'],2)

    def test_target_gauge_and_solid_values(self):
        fluid=np.ones((32,32),dtype=bool);fluid[10:20,10:20]=False
        a=np.ones((4,32,32));a[2]=7.;a[:3,~fluid]=np.nan
        value=canonical_target(a,fluid);self.assertTrue(np.isfinite(value).all());self.assertTrue((value[2]==0).all());self.assertTrue((value[:,~fluid]==0).all())

    def test_steady_residual_matches_analytic_couette(self):
        n=32;y=(np.arange(n)+.5)/n;u=np.broadcast_to(y[:,None],(n,n))
        value=np.stack((u,np.zeros_like(u),np.zeros_like(u)))
        r=reference_physics(value,np.ones((n,n),dtype=bool),100.)
        self.assertAlmostEqual(r['momentum_rms'],0.);self.assertAlmostEqual(r['divergence_rms'],0.)


if __name__=='__main__':unittest.main()
