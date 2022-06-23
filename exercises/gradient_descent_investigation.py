import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from IMLearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from IMLearn.metrics import misclassification_error
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_lst = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(np.array(val))
        weights_lst.append(np.array(weights))

    return callback, values, weights_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_dat = []
    l2_dat = []
    for et in etas:
        callback1, values1, weights1 = get_gd_state_recorder_callback()
        callback2, values2, weights2 = get_gd_state_recorder_callback()
        val1 = GradientDescent(FixedLR(et), callback=callback1).fit(L1(init), X=None, y=None)
        val2 = GradientDescent(FixedLR(et), callback=callback2).fit(L2(init), X=None, y=None)
        plot_descent_path(L1, np.array(weights1), title=f"Descent Path for L1 with eta={et}").write_image(
            f".\\Q1_L1_eta_{et}.png")
        plot_descent_path(L2, np.array(weights2), title=f"Descent Path for L2 with eta={et}").write_image(
            f".\\Q1_L2_eta_{et}.png")
        l1_dat.append(go.Scatter(x=np.arange(len(values1)), y=values1, mode='markers+lines', name=fr"$\eta={et}$",
                                 line=dict(width=1.5)))
        l2_dat.append(go.Scatter(x=np.arange(len(values2)), y=values2, mode='markers+lines', name=fr"$\eta={et}$",
                                 line=dict(width=1.5)))
        print(f"eta: {et}, L1: ", L1(val1).compute_output())
        print(f"eta: {et}, L2: ", L2(val2).compute_output())

    go.Figure(l1_dat).update_layout(
        title=f"Convergence Rate for L1", xaxis_title="iteration", font=dict(family="Times new roman", size=18),
        yaxis_title="L1").write_image(f".\\Q3_L1.png")
    go.Figure(l2_dat).update_layout(
        title=f"Convergence Rate for L2", xaxis_title="iteration", font=dict(family="Times new roman", size=18),
        yaxis_title="L2").write_image(f".\\Q3_L2.png")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    # Plot algorithm's convergence for the different values of gamma
    data = []
    for gam in gammas:
        cllbck = get_gd_state_recorder_callback()
        val = GradientDescent(ExponentialLR(eta, gam), callback=cllbck[0]).fit(L1(init), init, init)
        data.append(go.Scatter(x=np.arange(len(cllbck[1])), y=cllbck[1], mode='lines+markers',
                               name=fr"$\gamma = {gam}$", line=dict(width=1.5)))
        print(f"last val for gam {gam}, L1", L1(val).compute_output())

    go.Figure(data).update_layout(title=r"$\text{Convergence Rate for Different } \gamma, \eta=0.1$",
                                  xaxis_title="iteration",
                                  font=dict(family="Times new roman", size=18),
                                  yaxis_title=r"value of L1").write_image(f".\\Q5_gammas.png")

    # Plot descent path for gamma=0.95
    cllbck = get_gd_state_recorder_callback()
    val = GradientDescent(ExponentialLR(eta, 0.95), callback=cllbck[0]).fit(L1(init), init, init)
    plot_descent_path(L1, np.array(cllbck[2]), title=r"$\text{Convergence Rate for } \gamma=0.95, "
                                                     r"\eta=0.1$").write_image(f".\\Q7_gamma_0.95.png")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)))
    model.fit(np.array(X_train), np.array(y_train))

    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(np.array(X_train)))

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker=dict(size=5, color="rgb(49,54,149)"),
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).write_image(f".\\Q8_ROC.png")

    alpha_star = thresholds[np.argmax(tpr - fpr)]
    print("alpha* =", alpha_star)
    loss = misclassification_error(model.predict_proba(np.array(X_test)) >= alpha_star, np.array(y_test))
    print("loss over test is:", loss)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    val_l1 = np.zeros(7)
    val_l2 = np.zeros(7)
    for i, lambda_ in enumerate(lambdas):
        val_l1[i] = cross_validate(
            LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l1",
                               lam=lambda_), np.array(X_train), np.array(y_train), misclassification_error)[1]
        val_l2[i] = cross_validate(
            LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l2",
                               lam=lambda_), np.array(X_train), np.array(y_train), misclassification_error)[1]

    lam1 = lambdas[int(np.argmin(val_l1))]
    lam2 = lambdas[int(np.argmin(val_l2))]
    model_l1 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l1",
                                  lam=lam1)
    model_l2 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l2",
                                  lam=lam2)
    model_l1.fit(np.array(X_train), np.array(y_train))
    model_l2.fit(np.array(X_train), np.array(y_train))
    print("L1:", lam1, model_l1.loss(np.array(X_test), np.array(y_test)))
    print("L2:", lam2, model_l2.loss(np.array(X_test), np.array(y_test)))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
