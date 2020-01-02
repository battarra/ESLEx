""" Reinvents the KNN classifier wheel for illustrative purposes."""

import numpy as np
from typing import List
from .esl_classifier import EslClassifier


class KNNClassifier(EslClassifier):
    """ Classifier based on OLS of one-hot class encodings."""

    def __init__(self, k: int):

        super(KNNClassifier, self).__init__()

        assert k > 0, "Invalid k value {k}".format(k=k)

        self.k = k

        self.__X_train = None
        self.__X2_train = None
        self.__G_indices_fit = None

    def _fit(self, X: np.ndarray, G_one_hots: List[np.ndarray]) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hots: list (one element per label) of 2d arrays of one-hot-encoded categorical values,
                dimensions ``(N, n_levels)``.
        """

        self.__X_train = X.copy()
        self.__X2_train = np.sum(X * X, axis=1, keepdims=True)
        self.__G_indices_fit = [np.dot(G_one_hot, np.arange(G_one_hot.shape[1])) for G_one_hot in G_one_hots]

    def _predict_level_indices(self, X: np.ndarray, i_lab: int) -> np.ndarray:
        """ Predicts the most likely levels, returning an array of level indices, for one of the labels."""

        m_dist = self.__X2_train - 2 * np.dot(self.__X_train, X.T)  # no need to include the X^2 terms!

        # shape (X.shape[0], k)
        neighb_indices = np.argsort(m_dist, axis=0)[:self.k, :].T
        neighb_level_indices = self.__G_indices_fit[i_lab][neighb_indices]

        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=1,
            arr=neighb_level_indices
        )
