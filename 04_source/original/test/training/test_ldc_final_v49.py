"""CPU-only pure metric/canonicalization checks, no TEST reference access."""
import unittest
import numpy as np
from ldc_final_v49 import aggregate,metrics
from ldc_data_v25 import canonical_target


class FinalMetricTests(unittest.TestCase):
    def fixture(self):
        fluid=np.ones((32,32),dtype=bool);fluid[12:20,12:20]=False
        reference=np.zeros((32,32,3),dtype=np.float32);reference[fluid,0]=1
        return fluid,reference

    def test_identical_fields_have_zero_error(self):
        f,r=self.fixture();m=metrics(r.copy(),r,f,100)
        for k in ['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']:self.assertEqual(m[k],0)

    def test_primary_is_unweighted_all_cell_mse(self):
        f,r=self.fixture();p=r.copy();p[f,2]=2
        m=metrics(p,r,f,100)
        self.assertAlmostEqual(m['mse_all'],4*f.sum()/(32*32*3))

    def test_nonfinite_is_rejected_not_clipped(self):
        f,r=self.fixture();p=r.copy();p[0,0,0]=np.nan
        with self.assertRaises(ValueError):metrics(p,r,f,100)

    def test_reference_gauge_uses_only_fluid_and_excludes_property(self):
        f,r=self.fixture();raw=np.zeros((4,32,32));raw[2]=7;raw[2,~f]=1234;raw[3]=1e9
        c=canonical_target(raw,f)
        self.assertEqual(c.shape,(3,32,32));self.assertTrue(np.all(c[:,~f]==0))
        self.assertAlmostEqual(float(c[2,f].mean()),0)

    def test_aggregation_does_not_choose_best_case(self):
        keys=['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']
        result=aggregate([{k:1. for k in keys},{k:3. for k in keys}])
        self.assertTrue(all(v==2 for v in result.values()))


if __name__=='__main__':unittest.main()
