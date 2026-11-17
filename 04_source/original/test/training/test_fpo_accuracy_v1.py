import unittest
import numpy as np
from evaluate_fpo_pinn_pilot_v1 import errors,roles
class Tests(unittest.TestCase):
 def test_roles(self):
  r=roles();self.assertEqual([len(r[k]) for k in r],[40,40,160]);self.assertEqual(set.union(*r.values()),set(range(2,242)));self.assertFalse(r['train']&r['heldout_time'])
 def test_gauge(self):
  r=np.column_stack([np.ones(10),np.ones(10),np.arange(10)]);p=r.copy();p[:,2]+=100
  m=errors(p,r);self.assertEqual(m['velocity_relative_l2'],0);self.assertEqual(m['pressure_gauge_aligned_relative_l2'],0)
 def test_known_velocity(self):
  r=np.column_stack([np.ones(10),np.ones(10),np.arange(10)]);p=r.copy();p[:,:2]*=1.1
  self.assertAlmostEqual(errors(p,r)['velocity_relative_l2'],.1)
if __name__=='__main__':unittest.main(verbosity=2)
