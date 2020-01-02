""" PCR regression for ESL."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .esl_regressor import EslRegressor


class PcrRegressor(EslRegressor):
    """ PCR regression for ESL."""

    def __init__(self, n_components: int):
        """ Instantiates a ``PcrRegressor`` object.

        Args:
            regressor: regressor to be used with PCA components.
            n_components: number of components to be used.
        """

        super(PcrRegressor).__init__()

        self.n_components = n_components

        self.scaler = None    # type: StandardScaler
        self.pca = None

        self.regressor = None   # type: LinearRegression

    def _fit(self, X: np.ndarray, Y: np.ndarray = None):
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: 2d numpy array of responses, dimensions ``(N, n_responses)``.
        """

        self.scaler = StandardScaler()
        Xc = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=self.n_components)
        Xcr = self.pca.fit_transform(Xc)

        self.regressor = LinearRegression(fit_intercept=True)
        self.regressor.fit(Xcr, Y)

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 2d array."""

        Xc = self.scaler.transform(X)
        Xcr = self.pca.transform(Xc)

        return self.regressor.predict(Xcr)

    @property
    def coeffs(self):

        xcr_coef = self.regressor.coef_    # shape: (n_y, p)

        # coefficients as scaled
        coef_scaled = np.dot(xcr_coef, self.pca.transform(np.identity(self.n_features)).T)    # shape: (n_y, p)

        # restoring scale
        coef = coef_scaled / self.scaler.scale_[np.newaxis, :]

        return coef if len(self._fit_responses_shape) == 2 else coef[0, :]

    @property
    def intercept(self):

        xcr_coef = self.regressor.coef_    # shape: (n_y, p)
        xcr_intercept = self.regressor.intercept_    # shape: (ny,)

        # coefficients as scaled
        coef_scaled = np.dot(xcr_coef, self.pca.transform(np.identity(self.n_features)).T)    # shape: (n_y, p)

        intercept = xcr_intercept - np.dot(coef_scaled, self.scaler.mean_ / self.scaler.scale_)

        return intercept if len(self._fit_responses_shape) == 2 else intercept[0]
