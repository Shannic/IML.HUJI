from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.array([np.mean(X[y == cls], axis=0) for cls in self.classes_])
        mtx = np.zeros(shape=X.shape)
        for i in range(len(self.classes_)):
            # creating the matrix containing x[i]-mu[yi] in each row
            row_idx = y == self.classes_[i]
            mtx[row_idx] = X[row_idx] - self.mu_[i]
        # implementing the outer product using matrix multiplication, afterwards dividing by 'm-k'
        self.cov_ = (mtx.T @ mtx) / (y.size - self.classes_.size)
        self._cov_inv = np.linalg.inv(self.cov_)
        self.pi_ = np.array([np.mean(y == cls) for cls in self.classes_])
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a_vectors = self._cov_inv @ self.mu_.T
        b_vec = np.log(self.pi_) - 0.5 * np.sum(self.mu_.T * a_vectors, axis=0)
        b_vec.shape = (len(self.classes_), 1)
        return self.classes_[np.argmax(a_vectors.T @ X.T + b_vec, axis=0)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        tmp = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(self.cov_))
        likelihoods = []
        for i in range(len(self.classes_)):
            res = np.exp(-0.5 * np.sum((X - self.mu_[i]) @ self._cov_inv * (X - self.mu_[i]), axis=1)) * tmp * \
                  self.pi_[i]
            likelihoods.append(res)
        return np.array(likelihoods).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        predicted_y = self._predict(X)
        return misclassification_error(y, predicted_y)
