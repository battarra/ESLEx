""" Reinvents the wheel of an abstract classifier for illustrative purposes."""

import numpy as np


class AbstractClassifier(object):
    """ Abstract classifier class."""

    def __init__(self):

        self._n_fit_points = None
        self._n_features = None

        self._n_levels = None
        self._sorted_levels = None

    @property
    def n_levels(self) -> int:
        """ Returns the number of levels in the target categorical variable, as seen during the fit."""

        return self._n_levels

    @property
    def n_fit_points(self) -> int:
        """ Returns the number of datapoints seen in the fit."""

        return self._n_fit_points

    @property
    def n_features(self) -> int:
        """ Returns the number of features seen in the fit."""

        return self._n_features

    @property
    def sorted_levels(self) -> np.ndarray:
        """ Returns the array of (distinct) levels of the target categorical variable, as seen during the fit."""

        return self._sorted_levels

    def train(self, X: np.ndarray, G: np.ndarray) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G: array of categorical labels.
        """

        assert G.shape[0] > 0, "Zero length array cannot be used to train the classifier."

        self._n_fit_points = X.shape[0]
        self._n_features = X.shape[1]

        # preparing one-hot encoding, by hand because that's what this is about
        self._sorted_levels = np.array(sorted(list(set(G))))
        self._n_levels = len(self._sorted_levels)
        G_one_hot = G[:, np.newaxis] == self._sorted_levels[np.newaxis, :]

        self._train(X, G_one_hot)

    def _train(self, X: np.ndarray, G_one_hot: np.ndarray) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hot: array of one-hot-encoded categorical values, dimensions ``(N, n_levels)``.
        """

        raise NotImplementedError("AbstractClassifier does not implement _train - did you forget something?")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the most likely category, returning an array of level labels."""

        return self._sorted_levels[self._predict_level_indices(X)]

    def _predict_level_indices(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the most likely category, returning an array of level indices, sorted as per ``sorted_levels``."""

        raise NotImplementedError("AbstractClassifier does not implement _predict_level_indices - "
                                  "did you forget something?")

