from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .25) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    train_rows_number = int(np.ceil(y.size * train_proportion))
    # test_rows_number = np.floor(y.size * (1 - train_proportion))
    # mask = np.random.choice([False, True], y.size,
    #                         p=[test_rows_number, train_rows_number])
    # not_mask = np.logical_not(mask)
    # train_X = X[mask, :]
    # train_Y = y[mask]
    # test_X = X[not_mask, :]
    # test_Y = y[not_mask]
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:train_rows_number+1],\
                             indices[train_rows_number+1:]
    axis = 0
    train_X = np.take(X, training_idx, axis)
    train_Y = np.take(y, training_idx, axis)
    test_X = np.take(X, test_idx, axis)
    test_Y = np.take(y, test_idx, axis)
    return train_X, train_Y, test_X, test_Y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
