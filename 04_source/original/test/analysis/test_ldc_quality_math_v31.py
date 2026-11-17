"""Manufactured NS field verifies axis, sign, viscosity and pressure gauge."""
import unittest
import numpy as np
from ldc_quality_math_v31 import differential


class ManufacturedField(unittest.TestCase):
    def setUp(self):
        y, x = np.meshgrid((np.arange(32) + .5) / 32, (np.arange(32) + .5) / 32, indexing='ij')
        self.x, self.y = x, y
        self.value = np.stack((x**2, -2*x*y, x+y), -1)

    def test_known_navier_stokes_residual(self):
        result, _, _, _ = differential(self.value, 17.)
        expected = np.stack((2*self.x**3 + 1 - 2/17., 2*self.x**2*self.y + 1, self.x*0), -1)
        np.testing.assert_allclose(result[3:-3, 3:-3], expected[3:-3, 3:-3], rtol=0, atol=1e-11)

    def test_pressure_constant_cancels(self):
        shifted = self.value.copy()
        shifted[..., 2] += 19.
        a = differential(self.value, 17.)[0]
        b = differential(shifted, 17.)[0]
        np.testing.assert_allclose(a[3:-3, 3:-3], b[3:-3, 3:-3], rtol=0, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
