import unittest
import numpy as np
from fpo_context_audit_v20 import zones_from_distance,regional_errors,ZONE_NAMES


class ContextAuditTests(unittest.TestCase):
    def fixture(self):
        fluid=np.array([[False,True,True],[True,True,True]])
        distance=np.array([[0,.125,.126],[.5,.501,1.]])
        zones=zones_from_distance(fluid,distance)
        ref=np.arange(18,dtype=float).reshape(2,3,3)/10
        pred=ref.copy();pred[:,:,0]+=.2;pred[:,:,1]-=.3;pred[1,:,2]+=.4
        return pred,ref,fluid,zones

    def test_partition_thresholds_and_solid_exclusion(self):
        p,r,fluid,zones=self.fixture()
        self.assertEqual(zones.sum((1,2)).tolist(),[1,2,2])
        np.testing.assert_array_equal(zones.sum(0),fluid)
        self.assertFalse(zones[:,0,0].any())

    def test_global_pressure_gauge_invariance_and_region_offsets(self):
        p,r,fluid,zones=self.fixture();a=regional_errors(p,r,fluid,zones)
        p[:,:,2]+=17;r[:,:,2]-=33;b=regional_errors(p,r,fluid,zones)
        for region in a:
            self.assertAlmostEqual(a[region]['pressure_rmse'],b[region]['pressure_rmse'],places=12)
        self.assertGreater(a[ZONE_NAMES[0]]['pressure_rmse'],0.)

    def test_error_energy_additivity(self):
        p,r,fluid,zones=self.fixture();a=regional_errors(p,r,fluid,zones)
        for key in ['velocity_squared_error','pressure_squared_error']:
            self.assertAlmostEqual(sum(a[name][key] for name in ZONE_NAMES),a['all_fluid'][key],places=12)
        for key in ['area_fraction','velocity_error_fraction','pressure_error_fraction']:
            self.assertAlmostEqual(sum(a[name][key] for name in ZONE_NAMES),1.,places=12)


if __name__=='__main__':unittest.main()
