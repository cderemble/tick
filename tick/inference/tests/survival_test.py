import unittest

import numpy as np

from tick.inference import Survival

class Test(unittest.TestCase):

    def test_hazard_rate_from_survival_function(self):
        n_observations = 100

        timestamps = np.random.uniform(size=n_observations)
        observations = np.ones(n_observations)

        hzrd = Survival.nelson_aalen_hazard_rate(timestamps, observations)
        surv = Survival.kaplan_meier_survival_function(timestamps, observations)

        surv_from_hzrd = np.exp(-hzrd)

        self.assertTrue(np.allclose(surv, surv_from_hzrd, atol=1.e-2))

