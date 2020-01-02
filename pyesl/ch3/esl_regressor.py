""" Minimal abstract regressor class for ESL exercises."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class EslRegressor(BaseEstimator, RegressorMixin):
    """ Abstract regressor class for ESL."""

    def __init__(self):

        self._n_fit_points = None
        self._n_features = None

        self._fit_responses_shape = None
        self._n_responses = None

    @property
    def n_fit_points(self) -> int:
        """ Returns the number of datapoints seen in the fit."""

        return self._n_fit_points

    @property
    def n_features(self) -> int:
        """ Returns the number of features seen in the fit."""

        return self._n_features

    @property
    def n_responses(self) -> int:
        """ Returns the number of responses seen in the fit."""

        return self._n_responses

    def fit(self, X: np.ndarray, Y: np.ndarray) -> RegressorMixin:
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: 1d or 2d numpy matrix of responses.
        """

        self._fit_responses_shape = Y.shape

        Y = Y if len(Y.shape) == 2 else Y[:, np.newaxis]

        self._n_fit_points, self._n_responses = Y.shape
        self._n_features = X.shape[1]

        self._fit(X, Y)

        return self

    def _fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: 2d numpy array of responses, dimensions ``(N, n_responses)``.
        """

        raise NotImplementedError("EslRegressor does not implement _train - did you forget something?")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 2d array."""

        raise NotImplementedError("EslRegressor does not implement predict - did you forget something?")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 1d or 2d array depending on the shape of responses seen during fit."""

        Y = self._predict(X)

        return Y if len(self._fit_responses_shape) == 2 else Y[:, 0]

    def score(self, X, y=None):

        raise NotImplementedError("score not implemented in abstract class EslRegressor")

