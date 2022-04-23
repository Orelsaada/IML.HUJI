from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        shape = (n_classes, n_features)
        self.pi_, self.mu_, self.vars_ = np.zeros(n_classes), np.zeros(shape), \
                                         np.zeros(shape)
        m = len(X)
        for i, cls in enumerate(self.classes_):
            n_k = (y == cls).sum()
            self.pi_[i] = n_k / m
            self.mu_[i] = X[y == cls, :].sum() / n_k
            self.vars_[i] = np.var(X[y == cls], axis=0)

    def __argmax_k(self, xi: np.ndarray) -> int:
        a_k = (self._cov_inv @ self.mu_[0])
        b_k = np.log(self.pi_[0]) - 0.5 * self.mu_[0] @ self._cov_inv @ \
              self.mu_[0]
        max_val = (a_k.T @ xi) + b_k
        max_k = 0
        for k in range(1, len(self.classes_)):
            a_k = (self._cov_inv @ self.mu_[k])
            b_k = np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ \
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
        raise NotImplementedError()
