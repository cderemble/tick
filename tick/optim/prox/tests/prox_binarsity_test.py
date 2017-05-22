import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.optim.prox import ProxBinarsity
from tick.optim.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxBinarsity(self):
        """...Test of ProxBinarsity
        """
        coeffs = self.coeffs.copy()
        l_binarsity = 0.5
        t = 1.7
        out = np.array([0., 0., 0., -0.72681389, -0.72681389, 0.4845426,
                        0.4845426, 0.4845426, 0., 0.])
        blocks_start = [0, 3, 8]
        blocks_length = [3, 5, 2]
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length)

        val = 0
        for j, d_j in enumerate(blocks_length):
            start = blocks_start[j]
            val += np.abs(coeffs[start + 1:start + d_j]
                          - coeffs[start:start + d_j - 1]).sum()
        val *= l_binarsity
        self.assertAlmostEqual(prox.value(coeffs), val, delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)

        coeffs = self.coeffs.copy()
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length, positive=True)

        val = 0
        for j, d_j in enumerate(blocks_length):
            start = blocks_start[j]
            val += np.abs(coeffs[start + 1:start + d_j]
                          - coeffs[start:start + d_j - 1]).sum()
        val *= l_binarsity
        self.assertAlmostEqual(prox.value(coeffs), val, delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)

if __name__ == '__main__':
    unittest.main()
