""" Reinvents the OLS classifier wheel for illustrative purposes."""

import numpy as np
from .abstract_classifier import AbstractClassifier


class OLSClassifier(AbstractClassifier):
    """ Classifier based on OLS of one-hot class encodings."""

    def __init__(self):

        super(OLSClassifier, self).__init__()

        self.__beta = None

    @property
    def beta(self) -> np.ndarray:
        """ Returns the 2D array of OLS coefficients, shape ``(1 + n_features, n_levels)``."""

        return self.__beta

    def _train(self, X: np.ndarray, G_one_hot: np.ndarray) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hot: array of one-hot-encoded categorical values, dimensions ``(N, n_levels)``.
        """

        # including the constant in the regression
        Xo = np.ones((self._n_fit_points, self._n_features + 1))
        Xo[:, 1:] = X

        # performing the regression
        cov = np.dot(Xo.T, Xo) / self._n_fit_points
        inv_cov = np.linalg.inv(cov)

        self.__beta = np.dot(inv_cov, np.dot(Xo.T, G_one_hot))

    def _predict_level_indices(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the most likely category, returning an array of level indices, sorted as per ``sorted_levels``."""

        Xo = np.ones((X.shape[0], self._n_features + 1))
        Xo[:, 1:] = X
        Y = np.dot(Xo, self.__beta)

        return np.argmax(Y, axis=1)[::1]
