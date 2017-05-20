import numpy as np

from tick.base import actual_kwargs
from tick.inference.base import LearnerGLM
from tick.optim.model import ModelPoisReg
from tick.optim.solver import GD, AGD, SGD, BFGS, SVRG, AdaGrad


class PoissonRegression(LearnerGLM):
    """
    This is the Poisson regression model, with exponential link function.
    It supports several solvers and several penalizations.
    Note that for this model, there is no way to tune
    automatically the `step` of this learner. Thus, the default for `step`
    might work, or not, so that several values should be tried out.

    Parameters
    ----------
    step : `float`, default=1e-3
        Step-size to be used for the solver. For Poisson regression there is no
        way to tune it automatically. The default tuning might work, or not...

    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'tv'}, default='l2'
        The penalization to use. Default is ridge penalization

    solver : {'gd', 'agd', 'bfgs', 'svrg', 'sgd'}, default='svrg'
        The name of the solver to use

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    tol : `float`, default=1e-6
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=1
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : `int` seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers

    Attributes
    ----------
    weights : `numpy.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    intercept : `float` or None
        The intercept, if ``fit_intercept=True``, otherwise `None`
    """

    _solvers = {
        'gd': GD,
        'agd': AGD,
        'sgd': SGD,
        'svrg': SVRG,
        'bfgs': BFGS,
    }

    _attrinfos = {
        "_actual_kwargs": {"writable": False}
    }

    @actual_kwargs
    def __init__(self, step=1e-3, fit_intercept=True, penalty='l2', C=1e3,
                 tol=1e-5, max_iter=100, solver='svrg', verbose=False,
                 warm_start=False, print_every=10, record_every=1,
                 elastic_net_ratio=0.95, random_state=None):

        self._actual_kwargs = PoissonRegression.__init__.actual_kwargs
        # extra_model_kwargs = {'fit_intercept': fit_intercept}

        LearnerGLM.__init__(self, step=step, fit_intercept=fit_intercept,
                            penalty=penalty, C=C, solver=solver, tol=tol,
                            max_iter=max_iter, verbose=verbose,
                            warm_start=warm_start, print_every=print_every,
                            record_every=record_every,
                            elastic_net_ratio=elastic_net_ratio,
                            random_state=random_state)

    def _construct_model_obj(self, fit_intercept=True):
        return ModelPoisReg(fit_intercept=fit_intercept, link='exponential')


    # def _encode_labels_vector(self, labels):
    #     """Encodes labels values to canonical labels -1 and 1
    #
    #     Parameters
    #     ----------
    #     labels : `np.array`, shape=(n_samples,)
    #         Labels vector
    #
    #     Returns
    #     -------
    #     output : `np.array`, shape=(n_samples,)
    #         Encoded labels vector which takes values -1 and 1
    #     """
    #     # If it is already in the canonical shape return it
    #     # Additional check as if self.classes.dtype is not a number it raises
    #     # a warning
    #     if np.issubdtype(self.classes.dtype, np.number) and \
    #             np.array_equal(self.classes, [-1, 1]):
    #         return labels
    #
    #     mapper = {
    #         self.classes[0]: -1.,
    #         self.classes[1]: 1.
    #     }
    #     y = np.vectorize(mapper.get)(labels)
    #     return y

    def fit(self, X: object, y: np.array):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`,, shape=(n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : `np.array`, shape=(n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : `PoissonRegression`
            The fitted instance of the model
        """
        return LearnerGLM.fit(self, X, y)

    # def decision_function(self, X):
    #     """
    #     Predict scores for given samples
    #
    #     The confidence score for a sample is the signed distance of that
    #     sample to the hyperplane.
    #
    #     Parameters
    #     ----------
    #     X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
    #         Samples.
    #
    #     Returns
    #     -------
    #     output : `np.array`, shape=(n_samples,)
    #         Confidence scores.
    #     """
    #     if not self._fitted:
    #         raise ValueError("You must call ``fit`` before")
    #     else:
    #         X = self._safe_array(X)
    #         z = X.dot(self.weights)
    #         if self.intercept:
    #             z += self.intercept
    #         return z

    def predict(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Samples.

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Returns predicted values.
        """
        pass

        # TODO
        # scores = self.decision_function(X)
        # indices = (scores > 0).astype(np.int)
        # return self.classes[indices]

    # def predict_proba(self, X):
    #     """
    #     Probability estimates.
    #
    #     The returned estimates for all classes are ordered by the
    #     label of classes.
    #
    #     Parameters
    #     ----------
    #     X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
    #         Input features matrix
    #
    #     Returns
    #     -------
    #     output : `np.ndarray`, shape=(n_samples, 2)
    #         Returns the probability of the sample for each class
    #         in the model in the same order as in `self.classes`
    #     """
    #     if not self._fitted:
    #         raise ValueError("You must call ``fit`` before")
    #     else:
    #         score = self.decision_function(X)
    #         n_samples = score.shape[0]
    #         probs_class_1 = np.empty((n_samples,))
    #         ModelLogReg.sigmoid(score, probs_class_1)
    #         probs = np.empty((n_samples, 2))
    #         probs[:, 1] = probs_class_1
    #         probs[:, 0] = 1. - probs_class_1
    #         return probs
