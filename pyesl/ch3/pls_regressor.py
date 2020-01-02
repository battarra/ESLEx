""" PLS regression for ESL."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from .esl_regressor import EslRegressor


class PlsRegressor(EslRegressor):

    def __init__(self, n_components: int):
        """ PLS regression for ESL."""

        super(PlsRegressor).__init__()

        self.n_components = n_components

        self.scaler = None

        self.__Ym = None
        self.norm_coef_ = None

    def _fit(self, X: np.ndarray, Y: np.ndarray = None):
        """ Trains the regressor.

        Args:
            X: numpy matrix of input features, dimensions ``(N, n_features)``.
            Y: 2d numpy array of responses, dimensions ``(N, n_responses)``.
        """

        self.scaler = StandardScaler()
        Xc = self.scaler.fit_transform(X)

        p = Xc.shape[1]

        XtX = np.dot(Xc.T, Xc)
        XtY = np.dot(Xc.T, Y)

        self.__Ym = np.average(Y, axis=0)
        self.norm_coef_ = np.zeros((p, Y.shape[1]))

        for i_resp in range(Y.shape[1]):
            self.norm_coef_[:, i_resp] = self._fit_single_y(XtX, XtY[:, i_resp])

        return self

    def _fit_single_y(self, XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
        """ Fits the PLS regression for a single response variable, returning the fit coefficients."""

        XtY = Xty[:, np.newaxis]

        p = XtX.shape[0]

        # _ are coefficients in terms of X, shape (p, ...)
        # for example, Xm = X Xm_
        Xm_ = np.identity(p)

        yhat_ = np.zeros((p, 1))

        for i_comp in range(self.n_components):
            XmtY = np.dot(Xm_.T, XtY)  # shape: (pm, 1)

            # Zm = Xm Xm^t Y = X Xm_ Xm^t Y
            Zm_ = np.dot(Xm_, XmtY)
            Zm_norm2 = np.dot(Zm_.T, np.dot(XtX, Zm_))

            # regression of Y against Zm
            theta = np.dot(Zm_.T, XtY) / Zm_norm2
            yhat_ += theta * Zm_

            # residualisation coefficients of Xm against Zm
            beta = np.dot(Zm_.T, np.dot(XtX, Xm_)) / Zm_norm2  # shape: (1, pm)
            Xm_ -= np.dot(Zm_, beta)

        return yhat_[:, 0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts, returning a 2d array."""

        Xc = self.scaler.transform(X)

        return self.__Ym[np.newaxis, :] + np.dot(Xc, self.norm_coef_)


    @property
    def coeffs(self):

        coef = self.norm_coef_.T / self.scaler.scale_[np.newaxis, :]

        return coef if len(self._fit_responses_shape) == 2 else coef[0, :]

    @property
    def intercept(self):

        intercept = self.__Ym - np.dot(self.norm_coef_.T, self.scaler.mean_ / self.scaler.scale_)

        return intercept if len(self._fit_responses_shape) == 2 else intercept[0]



