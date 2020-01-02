""" Ridge regression wrapper for ESL."""

import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class RidgeRegressor(Ridge):
    """ Ridge regression parametrized with the number of degrees of freedom."""

    def __init__(self, n_dofs: float):

        super(RidgeRegressor, self).__init__(normalize=True)

        self.n_dofs = n_dofs

        self.scaler = None

    @staticmethod
    def ridge_n_dofs(s2: np.ndarray, N: int, alpha: float):
        """ Returns the number of degrees of freedom for a given regularization parameter."""

        return np.sum(s2 / (s2 + N * alpha))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: numpy matrix of responses, dimensions ``(N, n_responses)``.
        """

        # including the constant in the regression
        self.scaler = StandardScaler()
        Xc = self.scaler.fit_transform(X)

        s2 = np.linalg.eigvalsh(np.dot(Xc.T, Xc))

        if self.n_dofs == X.shape[1]:
            self.alpha = 0

        elif self.n_dofs < X.shape[1]:
            n_dof_goal = lambda log_alpha: RidgeRegressor.ridge_n_dofs(s2, Xc.shape[0], np.exp(log_alpha)) - self.n_dofs

            # notice the absence of N factor - when normalize=True, predictors are normalized to have unit L2 norm,
            # not unit std!
            self.alpha = np.exp(
                scipy.optimize.root_scalar(n_dof_goal, method='bisect', bracket=[-20, 20], xtol=1e-5).root
            )

        super(RidgeRegressor, self).fit(X, Y)

    @property
    def coeffs(self):
        return self.coef_

    @property
    def intercept(self):
        return self.intercept_
