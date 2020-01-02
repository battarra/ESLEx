""" Reinvents the OLS classifier wheel for illustrative purposes."""

import numpy as np
from typing import List
from .esl_classifier import EslClassifier


class OLSClassifier(EslClassifier):
    """ Classifier based on OLS of one-hot class encodings."""

    def __init__(self):

        super(OLSClassifier, self).__init__()

        self.__betas = None   # type: List[np.ndarray]

    def beta(self, i_lab: int) -> np.ndarray:
        """ Returns the 2D array of OLS coefficients, shape ``(1 + n_features, n_levels)``, for one of the labels."""

        return self.__betas[i_lab]

    def _fit(self, X: np.ndarray, G_one_hots: List[np.ndarray]) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hots: list (one element per label) of 2d arrays of one-hot-encoded categorical values,
                dimensions ``(N, n_levels)``.
        """

        # including the constant in the regression
        Xo = np.ones((self._n_fit_points, self._n_features + 1))
        Xo[:, 1:] = X

        # performing the regression
        cov = np.dot(Xo.T, Xo) / self._n_fit_points
        inv_cov = np.linalg.inv(cov)

        self.__betas = [np.dot(inv_cov, np.dot(Xo.T, G_one_hot)) for G_one_hot in G_one_hots]

    def _predict_level_indices(self, X: np.ndarray, i_lab: int) -> np.ndarray:
        """ Predicts the most likely levels, returning an array of level indices, for one of the labels."""

        Xo = np.ones((X.shape[0], self._n_features + 1))
        Xo[:, 1:] = X

        Y_hat = np.dot(Xo, self.__betas[i_lab])
        return np.argmax(Y_hat, axis=1)
