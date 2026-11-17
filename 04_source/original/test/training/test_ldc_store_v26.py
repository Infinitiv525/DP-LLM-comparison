import unittest
import numpy as np
import torch
from ldc_store_v26 import Store,conditioning,interior_mask,obstacle_faces


class StoreTests(unittest.TestCase):
    def fixture(self):
        s=Store.__new__(Store)
        s.train_ids=[10];s.validation_ids=[20];s.allowed={10,20}
        s.known_cache={i:dict(geometry_native=torch.ones(1,2,16,16)*i,
            fluid=torch.ones(16,16,dtype=torch.bool),reynolds=100.) for i in [10,20]}
        # Even a contaminated target dictionary cannot bypass the role check.
        s.targets={i:torch.ones(16,16,3)*i for i in [10,20,30]}
        s.validation_slots={20:0};s.y_validation=np.ones((1,3,16,16))
        return s

    def test_only_train_labels_are_available_to_optimizer(self):
        s=self.fixture();y=x=torch.tensor([1])
        self.assertEqual(float(s.training_target(10,y,x)[0,0]),10.)
        for i in [20,30]:
            with self.assertRaises(ValueError):s.training_target(i,y,x)
        with self.assertRaises(ValueError):s.validation_reference(10)
        with self.assertRaises(ValueError):s.validation_reference(30)
        self.assertEqual(s.validation_reference(20).shape,(3,16,16))

    def test_known_inputs_have_no_targets_and_no_test_access(self):
        s=self.fixture()
        self.assertEqual(set(s.known(20)),{'geometry_native','fluid','reynolds'})
        with self.assertRaises(ValueError):s.known(30)
        class Echo:
            def context(self,g,r):return g,r
        g,r=conditioning(Echo(),s.known(20))
        self.assertEqual(tuple(g.shape),(1,2,2,2));self.assertEqual(r,100.)
        # A changed input is pooled afresh, not served from a feature cache.
        s.known(20)['geometry_native'].zero_()
        g,_=conditioning(Echo(),s.known(20));self.assertEqual(float(g.sum()),0.)

    def test_body_faces_and_eroded_collocation_cells(self):
        fluid=np.ones((32,32),dtype=bool);fluid[12:20,12:20]=False
        interior=interior_mask(fluid)
        self.assertFalse(interior[:3].any());self.assertFalse(interior[16,9:23].any());self.assertFalse(interior[9:23,16].any())
        points=obstacle_faces(fluid)
        self.assertEqual(len(points),32)
        self.assertTrue(np.all((points>0)&(points<1)))
        self.assertTrue(np.all(np.isclose(points,.375).any(1)|np.isclose(points,.625).any(1)))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
