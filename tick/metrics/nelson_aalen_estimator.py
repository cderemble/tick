import numpy as np


class NelsonAalenEstimator:
    """A class that implements the Nelson-Aalen cumulative hazard rate estimator 
    given by:

    .. math::
        \\Lambda{(t_{(i)})} = \\sum_{j=1}^i \\frac{d_j}{n_j}
        
    where
    
      - :math:`d_j` are the number of deaths at :math:`t_{(j)}`
      - :math:`n_j` are the number of patients alive just before :math:`t_{(j)}` 
        
    """

    def fit(self, timestamps, observations, censored_observation=0):
        """Reseeds all simulations such that each simulation is started with a
        unique seed. The random selection of new seeds is seeded with the value
        given in 'seed'.

        Parameters
        ----------
        spectral_radius : `float`
            The targeted spectral radius
        """

        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)

        if isinstance(observations, list):
            observations = np.array(observations)

        unique_timestamps = np.concatenate((np.zeros(1), np.unique(timestamps)))

        return unique_timestamps, np.cumsum(
            np.fromiter((
                np.sum(observations[t == timestamps]) /
                np.sum(t <= timestamps)
                for t in unique_timestamps
            ), dtype='float', count=unique_timestamps.size))

