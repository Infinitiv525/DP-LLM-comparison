import unittest
import numpy as np
from audit_fpo_deep_v1 import derivative_check
class Tests(unittest.TestCase):
 def test_known_time_scale(self):
  a=np.zeros((242,64,64,3));x=4+(np.arange(64)+.5)*16/1024
  for k in range(242):a[k,:,:,0]=1+.1*k*.05;a[k,:,:,2]=-.1*x[None,:]
  for stride in [1,2]:
   for r in derivative_check(a,np.ones((64,64),bool),stride):
    self.assertAlmostEqual(r['estimated_snapshot_dt'],.05,places=8)
    self.assertLess(r['relative_residual'],1e-8)
if __name__=='__main__':unittest.main(verbosity=2)
