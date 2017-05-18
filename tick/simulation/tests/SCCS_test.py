import unittest
import numpy as np

from tick.simulation import SimuSCCS


class Test(unittest.TestCase):

    def test_censoring(self):
        array_list = [np.ones((2, 3)) for i in range(3)]
        expected = [np.zeros((2, 3)) for i in range(3)]
        for i in range(1, 3):
            expected[i][:i] += 1
        censoring = np.arange(3)

        output = SimuSCCS._censor_array_list(array_list, censoring)

        for i in range(3):
            np.testing.assert_equal(output[i], expected[i])

    def test_filter_non_positive_samples(self):
        features = [np.ones((2, 3)) * i for i in range(10)]
        labels = [np.zeros((2, 1)) for i in range(10)]
        censoring = np.full((10, 1), 2)

        expected_idx = np.sort(np.random.choice(np.arange(10), 5, False))
        for i in expected_idx:
            labels[i][i % 2] = 1
        expect_feat = [features[i] for i in expected_idx]
        expect_lab = [labels[i] for i in expected_idx]
        expect_cens = censoring[expected_idx]

        out_feat, out_lab, out_cens, out_idx = SimuSCCS\
            ._filter_non_positive_samples(features, labels, censoring)

        np.testing.assert_array_equal(expect_cens, out_cens)
        for i in range(len(expect_cens)):
            np.testing.assert_array_equal(expect_feat[i], out_feat[i])
            np.testing.assert_array_equal(expect_lab[i], out_lab[i])
            self.assertGreater(expect_lab[i].sum(), 0)

if __name__ == '__main__':
    unittest.main()
