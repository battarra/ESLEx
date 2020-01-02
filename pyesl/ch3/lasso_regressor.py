""" Re-parametrization of LASSO regression for ESL."""

import numpy as np
from .esl_regressor import EslRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, lars_path


class LassoRegressor(EslRegressor):
    """ LASSO regression.

    By default, predictors are normalized before regression.
    """

    def __init__(self, shrinkage: float):
        """ Constructs a LASSO regressor.

        Args:
            shrinkage: desired ratio between L1 norm of coefficients under regularization and L1 norm
               of OLS coefficient - 1 for no regularization.
        """

        super(LassoRegressor, self).__init__()

        self.shrinkage = shrinkage

        self._fit_response_shape_length = None

        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def lasso_shrinkage(X: np.ndarray, y: np.ndarray, ols_beta_l1: float, alpha: float):
        """ Computes the lasso shrinkage corresponding to a value of the regularization parameter ``alpha``."""

        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)

        return np.linalg.norm(lasso.coef_, ord=1) / ols_beta_l1

    def _fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: numpy matrix of responses, dimensions ``(N, n_responses)``.
        """

        self._fit_response_shape_length = len(Y.shape)

        assert len(Y.shape) == 1 or Y.shape[1] == 1

        # including the constant in the regression
        scaler = StandardScaler()
        Xc = scaler.fit_transform(X)

        self.intercept_ = np.average(Y) if len(Y.shape) == 1 else np.average(Y, axis = 0)

        if len(Y.shape) == 1:
            norm_coef = self._fit_single_Y(X, Y)

            self.coef_ = norm_coef / scaler.scale_
            self.intercept_ -= np.sum(self.coef_ * scaler.mean_)

        else:
            norm_coef = np.zeros((Y.shape[1], X.shape[1]))

            for i_resp in range(Y.shape[1]):
                norm_coef[i_resp, :] = self._fit_single_Y(X, Y[:, i_resp])

            self.coef_ = norm_coef / scaler.scale_[np.newaxis, :]
            self.intercept_ -= np.dot(self.coef_, scaler.mean_)

    def _fit_single_Y(self, X: np.ndarray, y: np.ndarray):

        if self.shrinkage == 0:
            return np.zeros((X.shape[1],))

        alphas, active, coefs = lars_path(X, y, method='lasso')

        coefs_ols = coefs[:, -1]

        if self.shrinkage == 1:
            return coefs[:, -1]

        coefs_ols_l1 = np.linalg.norm(coefs_ols, ord=1)

        shrinkages = np.array([np.linalg.norm(coefs[:, i], ord=1) for i in range(coefs.shape[1])]) / coefs_ols_l1

        i = np.searchsorted(shrinkages, self.shrinkage)
        if self.shrinkage == shrinkages[i]:
            return coefs[:, i]

        else:
            ls, rs = shrinkages[i - 1], shrinkages[i]

            return coefs[:, i - 1] + (self.shrinkage - ls) / (rs - ls) * (coefs[:, i] - coefs[:, i - 1])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 2d array."""

        return self.intercept_[np.newaxis, :] + np.dot(X, self.coef_.T)

    @property
    def coeffs(self):
        return self.coef_

    @property
    def intercept(self):
        return self.intercept_