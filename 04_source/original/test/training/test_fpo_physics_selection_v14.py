import unittest
from fpo_physics_selection_v14 import checkpoint_key


def report(error,div,limit=1.):return dict(score=error,interior_div_rms=div,mass_limit=limit,mass_pass=div<=limit,failure=False)


class SelectionTests(unittest.TestCase):
    def test_physics_qualified_preferred_but_accuracy_still_reported(self):
        self.assertLess(checkpoint_key([report(.2,.9)]),checkpoint_key([report(.01,1.1)]))

    def test_without_pass_reduce_worst_normalized_violation(self):
        self.assertLess(checkpoint_key([report(.3,2.),report(.3,2.)]),checkpoint_key([report(.01,1.),report(.01,3.)]))

    def test_among_physics_qualified_select_accuracy(self):
        self.assertLess(checkpoint_key([report(.05,.9)]),checkpoint_key([report(.2,.1)]))

    def test_failed_forecast_not_qualified(self):
        self.assertGreater(checkpoint_key([dict(score=1e6,failure=True)]),checkpoint_key([report(.2,.9)]))


if __name__=='__main__':unittest.main()
