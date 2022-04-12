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
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
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
        K = len(self.classes_)

        self.pi_ = np.zeros((K, 1))

        # map between yi to its class
        sample_class_map = dict()

        self.mu_ = np.zeros((K, X.shape[1]))
        for k in self.classes_:
            mu_k = np.zeros(X[0].shape)
            n_k = 0
            for i, xi in enumerate(X):
                if y[i] == k:
                    mu_k += xi
                    n_k += 1
                    sample_class_map[y[i]] = k
            mu_k /= n_k
            self.pi_[k] = n_k / y.size
            self.mu_[k] = mu_k

        cov = np.zeros((X.shape[1], X.shape[1]))
        for i, xi in enumerate(X):
            mu_yi = self.mu_[sample_class_map[y[i]]]
            cov += (xi - mu_yi) @ (xi - mu_yi).T
        self.cov_ = cov

        self._cov_inv = inv(self.cov_)

    def __argmax_k(self, xi: np.ndarray) -> int:
        a_k = (self._cov_inv @ self.mu_[0])
        b_k = np.log(self.pi_) - 0.5 * self.mu_[0] @ self._cov_inv @ \
              self.mu_[0]
        max_val = (a_k.T @ xi) + b_k
        max_k = 0
        for k in range(1, len(self.classes_)):
            a_k = (self._cov_inv @ self.mu_[k])
            b_k = np.log(self.pi_) - 0.5 * self.mu_[k] @ self._cov_inv @ \
                  self.mu_[k]
            new_val = (a_k.T @ xi) + b_k
            if new_val > max_val:
                max_val = new_val
                max_k = k
        return max_k

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
        response = np.zeros((X.shape[0], 1))
        for i, xi in enumerate(X):
            response[i] = self.__argmax_k(xi)
        return response

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

        raise NotImplementedError()

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
        return misclassification_error(y.reshape((-1, 1)), self.predict(X))
