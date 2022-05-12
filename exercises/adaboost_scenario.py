import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_learner = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_loss = []
    test_loss = []
    for T in range(1, n_learners + 1):
        train_loss.append(ada_learner.partial_loss(train_X, train_y, T))
        test_loss.append(ada_learner.partial_loss(test_X, test_y, T))
    data = [go.Scatter(x=list(range(1, n_learners + 1)), y=train_loss, mode='lines', name="train error",
                       showlegend=True),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_loss, mode='lines',
                       name="test error", showlegend=True)]
    fig1 = go.Figure(data)
    fig1.update_layout(title=f"Test and Train Error as Function of Amount of Learners, noise = {noise}",
                       xaxis_title="number of learners", yaxis_title="error").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"Number of Models: {t}" for t in T],
                         horizontal_spacing=0.01, vertical_spacing=.05)
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: ada_learner.partial_predict(x, t), lims[0], lims[1],
                                          showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, symbol="circle",
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig2.update_layout(title=f"Decision Boundaries of Models Including Test Set, noise = {noise}", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_loss) + 1
    fig3 = go.Figure(data=[decision_surface(lambda x: ada_learner.partial_predict(x, best_t), lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=test_y, symbol="circle",
                                                  colorscale=[custom[0], custom[-1]],
                                                  line=dict(color="black", width=1)))])

    accuracy_ = accuracy(test_y, ada_learner.partial_predict(test_X, best_t))
    fig3.update_layout(title=f"Decision Boundary with T={best_t}, Accuracy {accuracy_}", margin=dict(
        t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples
    D = 5 * ada_learner.D_ / np.max(ada_learner.D_)
    fig4 = go.Figure(data=[decision_surface(lambda x: ada_learner.predict(x), lims[0], lims[1], showscale=False),
                           go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=train_y, symbol="circle",
                                                  colorscale=[custom[0], custom[-1]],
                                                  size=D, line=dict(color="black", width=1)))])
    fig4.update_layout(title=f"Decision Boundary of Adaboost including Train Points Sized According to D[250]",
                       margin=dict(
                           t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
