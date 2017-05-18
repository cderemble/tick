"""AR-SCCS simulations utility."""
from .base.simu import Simu
from operator import itemgetter
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from tick.preprocessing.longitudinal_features_lagger\
    import LongitudinalFeaturesLagger


class SimuSCCS(Simu):
    _attrinfos = {
        "block_size": {
            "writable": False
        },
        "positive_sample_idx": {
            "writable": False
        }
    }

    def __init__(self, n_samples, n_intervals, n_features, n_lags, sparse=True,
                 exposure_type="infinite", distribution="multinomial",
                 first_tick_only=True, censoring=True, censoring_prob=.7,
                 censoring_intensity=.9, coeffs=None, seed=None, verbose=True,
                 batch_size=None):
        if exposure_type not in ["infinite", "short"]:
            raise ValueError("exposure_type can be only 'infinite' or 'short'")

        if distribution not in ["multinomial", "poisson"]:
            raise ValueError("distribution can be only 'multinomial' or\
             'poisson'")

        if coeffs is not None and coeffs.shape != (n_features * (n_lags + 1),):
            raise ValueError("Coeffs should be of shape\
             (n_features * (n_lags + 1),).")
        super(SimuSCCS, self).__init__(seed, verbose)
        self.n_samples = n_samples
        self.n_intervals = n_intervals
        self.n_features = n_features
        self.n_lags = n_lags
        self.sparse = sparse
        self.exposure_type = exposure_type
        self.distribution = distribution
        self.first_tick_only = first_tick_only
        self.censoring = censoring
        self.censoring_prob = censoring_prob
        self.censoring_intensity = censoring_intensity
        self.coeffs = coeffs
        if (batch_size is None) or (batch_size > n_samples):
            self.batch_size = int(min(2000, n_samples))
        else:
            self.batch_size = int(batch_size)

    def _simulate(self):
        # Simulate coefficients according to a gaussian if they are not
        # specified.
        n_lagged_features = (self.n_lags + 1) * self.n_features
        if self.coeffs is None:
            self.coeffs = np.random.normal(1e-3, 1.1, n_lagged_features)

        features = []
        outcomes = []
        out_censoring = np.zeros((self.n_samples, 1), dtype="uint64")
        sample_count = 0
        while sample_count < self.n_samples:
            X_temp, y_temp, _censoring, _ = self._simulate_batch()
            n_new_samples = len(y_temp)
            expected_count = sample_count + n_new_samples
            if expected_count >= self.n_samples:
                n = n_new_samples - (expected_count - self.n_samples)
            else:
                n = n_new_samples

            features.extend(X_temp[0:n])
            outcomes.extend(y_temp[0:n])
            out_censoring[sample_count:sample_count+n] = _censoring[0:n]
            sample_count += n_new_samples

        return features, outcomes, out_censoring, self.coeffs

    def _simulate_batch(self):
        _censoring = np.full((self.batch_size, 1), self.n_intervals,
                             dtype="uint64")
        # No censoring right now
        X_temp = self._simulate_sccs_features(self.batch_size)
        y_temp = self._simulate_outcomes(X_temp, _censoring)

        if self.censoring:
            censored_idx = np.random.binomial(1, self.censoring_prob,
                                              size=self.batch_size
                                              ).astype("bool")
            _censoring[censored_idx] -= np.random.poisson(
                lam=self.censoring_intensity, size=(censored_idx.sum(), 1)
                ).astype("uint64")
            X_temp = self._censor_array_list(X_temp, _censoring)
            y_temp = self._censor_array_list(y_temp, _censoring)

        return self._filter_non_positive_samples(X_temp, y_temp,
                                                 _censoring)

    def _simulate_sccs_features(self, n_samples):
        if self.exposure_type == "infinite":
            sim_feat = lambda: self._sim_infinite_exposures(self.sparse)
        elif self.exposure_type == "short":
            sim_feat = lambda: self._sim_short_exposures(self.sparse)

        return [sim_feat() for i in range(n_samples)]

    def _sim_short_exposures(self, sparse):
        features = np.random.randint(2,
                                     size=(self.n_intervals, self.n_features),
                                     dtype="float64")
        if sparse:
            features = csr_matrix(features, dtype="float64")
        return features

    def _sim_infinite_exposures(self, sparse):
        if not sparse:
            raise ValueError("'infinite' exposures can only be simulated as \
            sparse feature matrices")
        # Select features for which there is exposure
        n_exposures = np.random.randint(1, self.n_features, 1)
        cols = np.random.choice(np.arange(self.n_features, dtype="int64"),
                                n_exposures, replace=False)
        rows = np.random.randint(self.n_intervals, size=n_exposures)
        data = np.ones_like(cols, dtype="float64")
        return csr_matrix((data, (rows, cols)),
                          shape=(self.n_intervals, self.n_features),
                          dtype="float64")

    def _simulate_outcomes(self, features, censoring):
        """Distribution must be equal to 'poisson' or 'multinomial'."""
        features = LongitudinalFeaturesLagger(n_lags=self.n_lags).\
            fit_transform(features, censoring)

        if self.distribution == "poisson":
            y = self._simulate_outcome_from_poisson(features, self.coeffs,
                                                    self.first_tick_only)
        else:
            y = self._simulate_outcome_from_multi(features, self.coeffs,)
        return y

    @staticmethod
    def _simulate_outcome_from_multi(features, coeffs):
        inner_product = [f.dot(coeffs) for f in features]

        def simulate(inner_prod):
            inner_prod -= inner_prod.max()
            probabilities = np.exp(inner_prod) / \
                            np.sum(np.exp(inner_prod))
            y = np.random.multinomial(1, probabilities)
            return y.astype("uint64")

        return [simulate(i) for i in inner_product]

    @staticmethod
    def _simulate_outcome_from_poisson(features, coeffs, first_tick_only=True):
        inner_product = [f.dot(coeffs) for f in features]

        def simulate(inner_prod):
            intercept = -inner_prod.max()
            intensities = np.exp(intercept + inner_prod)
            ticks = np.random.poisson(lam=intensities)
            if first_tick_only:
                first_tick_idx = np.argmax(ticks > 0)
                y = np.zeros_like(intensities)
                if ticks.sum() > 0:
                    y[first_tick_idx] = 1
            else:
                y = ticks
            return y.astype("uint64")

        return [simulate(i) for i in inner_product]

    @staticmethod
    def _censor_array_list(array_list, censoring):
        def censor(array, censoring_idx):
            if sps.issparse(array):
                array = array.tolil()
                array[int(censoring_idx):] = 0
                array = array.tocsr()
            else:
                array[int(censoring_idx):] = 0
            return array
        return [censor(l, censoring[i]) for i, l in enumerate(array_list)]

    @staticmethod
    def _filter_non_positive_samples(features, labels, censoring):
        """Filter out samples which don't tick in the observation window.
        Parameters
        ----------
        features : List[{2d array, csr matrix containing float64
                   of shape (n_intervals, n_features)}]
            The features matrix
        labels : List[{1d array, csr matrix of shape (n_intervals, 1)]
            The labels vector
        """
        nnz = [np.nonzero(arr)[0] for arr in labels]
        positive_sample_idx = [i for i, arr in enumerate(nnz) if
                               len(arr) > 0]
        if len(positive_sample_idx) == 0:
            raise ValueError("There should be at least one positive sample")
        pos_samples_filter = itemgetter(*positive_sample_idx)
        return list(pos_samples_filter(features)), \
               list(pos_samples_filter(labels)), \
               censoring[positive_sample_idx], \
               np.array(positive_sample_idx, dtype="uint64")
