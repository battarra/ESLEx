""" Reinvents the wheel of an abstract classifier for illustrative purposes."""

import numpy as np
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin


class EslClassifier(BaseEstimator, ClassifierMixin):
    """ Abstract classifier class for ESL."""

    def __init__(self):

        self._n_features = None

        self._n_labels = None
        self._fit_labels_shape = None

        self._n_fit_points = None

        self._n_levels_by_label = None
        self._sorted_levels_by_label = None

    @property
    def n_features(self) -> int:
        """ Returns the number of features seen in the fit."""

        return self._n_features

    @property
    def n_fit_points(self) -> int:
        """ Returns the number of data points seen in the fit."""

        return self._fit_labels_shape[0]

    def n_levels(self, i_label: int = 0) -> int:
        """ Returns the number of levels observed during fit in one of the target categorical variables.

        Args:
            i_label: index of the label in the training data.
        """

        return self._n_levels_by_label[i_label]

    @property
    def n_labels(self) -> int:
        """ Returns the number of responses."""

        return self._n_labels

    def sorted_levels(self, i_lab: int) -> np.ndarray:
        """ Returns the array of (distinct) levels of the target categorical variable, as seen during the fit."""

        return self._sorted_levels_by_label[i_lab]

    def fit(self, X: np.ndarray, G: np.ndarray) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G: 1d or 2d array of categorical labels.
        """

        self._fit_labels_shape = G.shape

        # if single response, add axis
        G = G if len(G.shape) == 2 else G[:, np.newaxis]
        self._n_fit_points, self._n_labels = G.shape

        self._n_features = X.shape[1]

        # preparing one-hot encoding, by hand because that's what this is about
        self._sorted_levels_by_label = [np.array(sorted(list(set(G[:, i_lab])))) for i_lab in range(self._n_labels)]
        self._n_levels_by_label = [len(levels) for levels in self._sorted_levels_by_label]

        # for each label, create a 2d array of one-hot variables for fitting
        G_one_hots = [
            G[:, i_lab:i_lab + 1] == self._sorted_levels_by_label[i_lab][np.newaxis, :] \
                for i_lab in range(self._n_labels)
        ]

        self._fit(X, G_one_hots)

    def _fit(self, X: np.ndarray, G_one_hots: List[np.ndarray]) -> None:
        """ Trains the classifier.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            G_one_hots: list (one element per response) of 2d arrays of one-hot-encoded categorical values,
                dimensions ``(N, n_levels)``.
        """

        raise NotImplementedError("EslClassifier does not implement _train - did you forget something?")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the most likely category, returning an array of level labels."""

        predicted_levels = np.full(shape=(X.shape[0], self._n_labels), fill_value=None, dtype=object)
        for i_lab in range(self._n_labels):
            predicted_indices = self._predict_level_indices(X, i_lab)
            predicted_levels[:, i_lab] = self._sorted_levels_by_label[i_lab][predicted_indices]

        # return 1d array if 1d array had been given as input to fit
        return predicted_levels if len(self._fit_labels_shape) == 2 else predicted_levels[:, 0]

    def _predict_level_indices(self, X: np.ndarray, i_lab: int) -> np.ndarray:
        """ Predicts the most likely levels, returning an array of level indices, for one of the labels."""

        raise NotImplementedError("EslClassifier does not implement _predict_level_indices - "
                                  "did you forget something?")

    def score(self, X, y=None):

        raise NotImplementedError("score not implemented in abstract class EslClassifier")

