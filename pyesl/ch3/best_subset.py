""" Implements exhaustive best subset regression for ESL."""

import numpy as np
import copy
import itertools as itr
from typing import List
from sklearn.linear_model import LinearRegression
from .esl_regressor import EslRegressor


class BestSubsetRegression(EslRegressor):
    """ Exhaustive best subset regression for ESL."""

    def __init__(self, subset_size: int):
        """ Instantiates a Best Subset regressor.

        Args:
            regressor: regressor used for regression after subset selection.
            subset_size: subset size.
        """

        self.subset_size = subset_size

        self.__best_models = None   # type: List[LinearRegression]
        self.__best_preds = None    # type: np.ndarray   # shape: (n_responses, subset_size)

    def best_preds(self, i_resp: int):
        """ Returns the array of best predictors for a specific response."""

        return self.__best_preds[i_resp, :]

    def _fit(self, X: np.ndarray, Y: np.ndarray = None):
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: 2d numpy array of responses, dimensions ``(N, n_responses)``.
        """

        best_scores = np.full(shape=(self._n_responses,), fill_value=-np.inf)
        self.__best_models = [None] * self._n_responses
        self.__best_preds = np.zeros((self._n_responses, self.subset_size), dtype=int)

        regressor = LinearRegression(fit_intercept=True)

        for preds in itr.combinations(np.arange(X.shape[1]), self.subset_size):

            for i_resp in range(self._n_responses):
                regressor.fit(X[:, list(preds)], Y[:, i_resp])

                score = regressor.score(X[:, list(preds)], Y[:, i_resp])
                if score > best_scores[i_resp]:
                    best_scores[i_resp] = score
                    self.__best_models[i_resp] = copy.deepcopy(regressor)
                    self.__best_preds[i_resp, :] = list(preds)

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 2d array."""

        Yhat = np.zeros((X.shape[0], self._n_responses))

        for i_resp in range(self._n_responses):
            Yhat[:, i_resp] = self.__best_models[i_resp].predict(X[:, self.__best_preds[i_resp, :]])

        return Yhat

    @property
    def coeffs(self):

        coef = np.zeros((self.n_responses, self.n_features))
        for i_resp in range(self._n_responses):
            coef[i_resp, self.__best_preds[i_resp, :]] = self.__best_models[i_resp].coef_

        return coef if len(self._fit_responses_shape) == 2 else coef[0, :]

    @property
    def intercept(self):

        intercept = np.array([self.__best_models[i_resp].intercept_ for i_resp in range(self._n_responses)])

        return intercept if len(self._fit_responses_shape) == 2 else intercept[0]
