from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    validation_scores = np.zeros(cv)
    train_sores = np.zeros(cv)
    index_partition = np.remainder(np.arange(y.size), cv)
    for i in range(cv):
        train_x, train_y = X[index_partition != i], y[index_partition != i]
        test_x, test_y = X[index_partition == i], y[index_partition == i]
        model = estimator.fit(train_x, train_y)
        validation_scores[i] = np.mean((model.predict(test_x) - test_y) ** 2)
        train_sores[i] = np.mean((model.predict(train_x) - train_y) ** 2)
    return float(np.mean(train_sores)), float(np.mean(validation_scores))
