import unittest
import torch
from train_fpo_pinn_pilot_v1 import ns_residual
class Tests(unittest.TestCase):
 def test_solid_rotation(self):
  q=torch.rand(10,3,dtype=torch.float64,requires_grad=True);x,y,t=q.unbind(1);z=0*(x**3+y**3+t**3)
  pred=torch.stack([-y+z,x+z,.5*(x*x+y*y)+z],1)
  self.assertLess(float(ns_residual(q,pred).detach().abs().max()),1e-10)
 def test_uniform_acceleration(self):
  q=torch.rand(10,3,dtype=torch.float64,requires_grad=True);x,y,t=q.unbind(1);z=0*(x**3+y**3+t**3)
  pred=torch.stack([1+.1*t+z,z,-.1*x+z],1)
  self.assertLess(float(ns_residual(q,pred).detach().abs().max()),1e-10)
if __name__=='__main__':unittest.main(verbosity=2)
