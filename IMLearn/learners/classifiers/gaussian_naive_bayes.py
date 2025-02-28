from typing import NoReturn
from ...base import BaseEstimator
from numpy.linalg import det, inv
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
        n_features = 1 if len(X.shape) <= 1 else X.shape[1]
        shape = (n_classes, n_features)
        self.pi_, self.mu_, self.vars_ = np.zeros(n_classes), np.zeros(shape), \
                                         np.zeros(shape)
        m = len(X)
        for i, cls in enumerate(self.classes_):
            n_k = (y == cls).sum()
            self.pi_[i] = n_k / m
            self.mu_[i] = np.mean(X[y == cls], axis=0)
            self.vars_[i] = np.var(X[y == cls], axis=0, ddof=1)

    def _pdf(self, cls_idx, xi):
        mu = self.mu_[cls_idx]
        var = self.vars_[cls_idx]
        numerator = np.exp(-(xi - mu)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

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
        # pred = []
        #
        # for xi in X:
        #     posteriors = []
        #     for idx, cls in enumerate(self.classes_):
        #         prior = np.log(self.pi_[idx])
        #         class_conditional = np.sum(np.log(self._pdf(idx, xi)))
        #         posterior = prior + class_conditional
        #         posteriors.append(posterior)
        #     pred.append(self.classes_[np.argmax(posteriors)])
        #
        # return np.array(pred)

        likelihood = self.likelihood(X)
        return np.argmax(likelihood, axis=0)


    def _class_likelihood(self, X, cls):
        cov = np.diag(self.vars_[cls])
        dimension = 1 if len(X.shape) <= 1 else X.shape[1]
        diff = X - self.mu_[cls]
        expo = np.sum(diff @ inv(cov) * diff, axis=1) * -0.5
        factor = 1 / np.sqrt(((2 * np.pi) ** dimension) * det(cov))
        return factor * np.exp(expo) * self.pi_[cls]

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

        res = []
        for i in range(len(self.classes_)):
            res.append(self._class_likelihood(X, i))
        return np.array(res)

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
