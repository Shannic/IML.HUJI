from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    f = lambda X: (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = f(x) + eps
    org_x = np.linspace(-1.2, 2, 200)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), train_proportion=(2 / 3))
    fig1 = go.Figure([go.Scatter(x=np.array(test_x)[:, 0], y=np.array(test_y), mode='markers',
                                 line=dict(width=3, color="red"), name="test set"),
                      go.Scatter(x=np.array(train_x)[:, 0], y=np.array(train_y), mode='markers',
                                 line=dict(width=3, color="purple"), name="train set"),
                      go.Scatter(x=org_x, y=f(org_x), mode='lines', line=dict(width=3, color="black"),
                                 name="true model")])
    fig1.update_layout(title=f"True Model vs Train & Test Sets, noise={noise}",
                       xaxis_title=r"$\text{x}$", font=dict(family="Times new roman", size=18),
                       yaxis_title=r"$\text{y}$").write_image(f".\\q1_noise{noise}.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_avgs = np.zeros(11)
    train_avgs = np.zeros(11)
    for k in range(11):
        train_avgs[k], validation_avgs[k] = cross_validate(PolynomialFitting(k=k), np.array(train_x)[:, 0],
                                                           np.array(train_y), mean_square_error, cv=5)
    fig2 = go.Figure([go.Scatter(x=np.arange(11), y=validation_avgs, mode='markers+lines',
                                 line=dict(width=3, color="red"), name="average validation error"),
                      go.Scatter(x=np.arange(11), y=train_avgs, mode='markers+lines',
                                 line=dict(width=3, color="purple"), name="average train error")])
    fig2.update_layout(title=f"Average MSE in 5-folds CV as Function of Polynomial Degree, noise={noise}",
                       xaxis_title=r"$\text{degree}$", font=dict(family="Times new roman", size=16),
                       yaxis_title=r"$\text{average MSE}$").write_image(f".\\q2_noise{noise}.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_avgs)
    model = PolynomialFitting(k=int(k_star)).fit(np.array(train_x)[:, 0], np.array(train_y))
    print(f"noise {noise}, "
          f"MSE over Test for k*={k_star}: ", np.round(model.loss(np.array(test_x)[:, 0], np.array(test_y)), 2))
    print("Best Validation Error was", np.round(validation_avgs[k_star], 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y),
                                                        train_proportion=n_samples / y.shape[0])

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.01, 2, n_evaluations)
    train_lasso = np.zeros(n_evaluations)
    val_lasso = np.zeros(n_evaluations)
    train_ridge = np.zeros(n_evaluations)
    val_ridge = np.zeros(n_evaluations)
    for i, lambda_ in enumerate(lambdas):
        train_ridge[i], val_ridge[i] = cross_validate(RidgeRegression(lambda_), np.array(train_x),
                                                      np.array(train_y), mean_square_error, cv=5)
        train_lasso[i], val_lasso[i] = cross_validate(Lasso(alpha=lambda_), np.array(train_x),
                                                      np.array(train_y), mean_square_error, cv=5)
    fig3 = go.Figure([go.Scatter(x=lambdas, y=train_lasso, mode='lines', name="train lasso"),
                      go.Scatter(x=lambdas, y=val_lasso, mode='lines', name="validation lasso"),
                      go.Scatter(x=lambdas, y=train_ridge, mode='lines', name="train ridge"),
                      go.Scatter(x=lambdas, y=val_ridge, mode='lines', name="validation ridge")])
    fig3.update_layout(title="Train & Validation Errors as Function of Regularization Parameter",
                       xaxis_title=r"$\lambda$", font=dict(family="Times new roman", size=16),
                       yaxis_title="MSE").write_image(f".\\q7.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso = lambdas[np.argmin(val_lasso)]
    best_ridge = lambdas[np.argmin(val_ridge)]
    print("Lasso optimal reg. parameter:", best_lasso, "\nRidge optimal reg. parameter:", best_ridge)
    ridge = RidgeRegression(best_ridge).fit(np.array(train_x), np.array(train_y))
    lasso = Lasso(best_lasso)
    lasso.fit(np.array(train_x), np.array(train_y))
    lin_reg = LinearRegression().fit(np.array(train_x), np.array(train_y))
    print("Ridge MSE:", ridge.loss(np.array(test_x), np.array(test_y)))
    print("Lasso MSE:", mean_square_error(np.array(test_y), lasso.predict(np.array(test_x))))
    print("Linear Regression MSE:", lin_reg.loss(np.array(test_x), np.array(test_y)))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
