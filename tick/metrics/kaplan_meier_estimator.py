import numpy as np


class KaplanMeierEstimator:

    def __init__(self):
        pass

    def fit(self, timestamps, observations, censored_observation=0):
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)

        if isinstance(observations, list):
            observations = np.array(observations)

        unique_timestamps = np.concatenate((np.zeros(1), np.unique(timestamps)))

        return unique_timestamps, np.cumprod(
            np.fromiter((
                1.0 -
                np.sum(observations[t == timestamps] != censored_observation) /
                np.sum(t <= timestamps)
                for t in unique_timestamps
            ), dtype='float', count=unique_timestamps.size))

