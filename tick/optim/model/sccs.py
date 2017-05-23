import numpy as np
from .base import ModelFirstOrder, ModelLipschitz
from .build.model import ModelSCCS as _ModelSCCS
from tick.preprocessing.utils import check_longitudinal_features_consistency, \
    check_censoring_consistency


class ModelSCCS(ModelFirstOrder, ModelLipschitz):
    """SCCS model. This class gives first order information (gradient and loss)
    for this model.

    Parameters
    ----------
    n_intervals : `int`
        Number of time intervals observed for each sample.

    n_lags : `int`
        Number of lags. The model will regress labels on the last observed
        values of the features over the `n_lags` time intervals. `n_lags`
         must be between 0 and `n_intervals` - 1

    Attributes
    ----------
    features : List[{2d array, csr matrix of shape (n_intervals, n_features)}]
            The features matrix

    labels : List[{1d array, csr matrix of shape (n_intervals, 1)]
        The labels vector

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model
    """

    _attrinfos = {
        "labels": {
            "writable": False
        },
        "features": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "tick_index": {
            "writable": False
        },
        "positive_sample_idx": {
            "writable": False
        },
        "_n_features": {
            "writable": False
        },
        "_n_samples": {
            "writable": False
        },
        "_n_lags": {
            "writable": False
        },
        "_n_intervals": {
            "writable": False
        }
    }

    def __init__(self, n_intervals: int, n_lags: int):
        if n_lags >= n_intervals:
            raise ValueError("n_lags should be < n_intervals")

        ModelFirstOrder.__init__(self)
        ModelLipschitz.__init__(self)
        self.n_lags = n_lags
        self.n_intervals = n_intervals

    def fit(self, features, labels, censoring=None):
        """Set the data into the model object

        Parameters
        ----------
        features : List[{2d array, csr matrix containing float64
                   of shape (n_intervals, n_features)}]
            The features matrix

        labels : List[{1d array, csr matrix of shape (n_intervals,)]
            The labels vector
            
        censoring : 1d array of shape (n_samples,)
            The censoring vector

        Returns
        -------
        output : `ModelSCCS`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels, censoring)
        ModelLipschitz.fit(self, features, labels)

        self._set("_model",
                  _ModelSCCS(features=self.features,
                             labels=self.labels,
                             censoring=self.censoring,
                             n_lags=self.n_lags)
                  )

        return self

    def _set_data(self, features, labels, censoring):
        """Set the data to the model.

        Parameters
        ----------
        features : List[{2d array, csr matrix containing float64
                   of shape (n_intervals, n_features)}] containing n_samples
                   elements
            The features matrix

        labels : List[{1d array, csr matrix of shape (n_intervals,)]
                 containing n_samples elements
            The labels vector

        censoring : 1d array of shape (n_samples,)
            The censoring vector
        """
        base_shape = features[0].shape
        self._set("n_intervals", base_shape[0])
        self._set("n_features", base_shape[1])
        self._set("n_samples", len(features))
        if len(labels) != self.n_samples:
            raise ValueError("Features and labels lists should have the same\
             length.")
        if censoring is None:
            censoring = np.full(self.n_samples, self.n_intervals,
                                dtype="uint64")
        censoring = check_censoring_consistency(censoring, self.n_samples)
        features = check_longitudinal_features_consistency(features,
                                                           base_shape,
                                                           "float64")
        labels = check_longitudinal_features_consistency(labels,
                                                         (self.n_intervals,),
                                                         "int32")

        self._set("labels", labels)
        self._set("features", features)
        self._set("censoring", censoring)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()

    def _get_lip_best(self):
        return self.get_lip_max()

    @property
    def _epoch_size(self):
        return self._model.get_epoch_size()

    @property
    def _rand_max(self):
        return self._model.get_rand_max()
