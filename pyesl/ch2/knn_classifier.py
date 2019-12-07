""" Reinvents the KNN classifier wheel for illustrative purposes."""

import numpy as np
from .abstract_classifier import AbstractClassifier


class KNNClassifier(AbstractClassifier):
    """ Classifier based on OLS of one-hot class encodings."""

    def __init__(self, k: int):

        super(KNNClassifier, self).__init__()

        assert k > 0, "Invalid k value {k}".format(k=k)

        self.k = k

        self.__X_train = None
        self.__X2_train = None
        self.__G_one_hot_train = None

    def _train(self, X: np.ndarray, G_one_hot: np.ndarray) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hot: array of one-hot-encoded categorical values, dimensions ``(N, n_levels)``.
        """

        self.__X_train = X.copy()
        self.__X2_train = np.sum(X * X, axis=1, keepdims=True)
        self.__G_indices = np.dot(G_one_hot, np.arange(self.n_levels))

    def _predict_level_indices(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the most likely category, returning an array of level indices, sorted as per ``sorted_levels``."""

        m_dist = self.__X2_train - 2 * np.dot(self.__X_train, X.T)  # no need to include the X^2 terms!

        # shape (X.shape[0], k)
        neighb_indices = np.argsort(m_dist, axis=0)[:self.k, :].T
        neighb_level_indices = self.__G_indices[neighb_indices]

        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=1,
            arr=neighb_level_indices
        )
