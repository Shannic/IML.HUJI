from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data_X, data_y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perecptron = Perceptron(callback=lambda percpt, X, y: losses.append(percpt.loss(data_X, data_y)))
        perecptron.fit(data_X, data_y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure([go.Scatter(x=np.array(range(len(losses))) + 1, y=losses,
                                    mode='markers+lines',
                                    line=dict(width=3, color="gray"))])
        fig.update_layout(barmode='overlay',
                          title=f"Loss over {n} Data Set In each Iteration",
                          xaxis_title="Loss of the data",
                          yaxis_title="Iteration number",
                          height=500)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data_X, data_y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda_estimator = LDA()
        gaussian_naive_estimator = GaussianNaiveBayes()
        lda_estimator.fit(data_X, data_y)
        gaussian_naive_estimator.fit(data_X, data_y)
        lda_pred = lda_estimator.predict(data_X)
        gn_pred = gaussian_naive_estimator.predict(data_X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Naive Gaussian, Accuracy: {accuracy(gn_pred, data_y)}",
                                                            f"LDA, Accuracy: {accuracy(lda_pred, data_y)}"])
        lda_data = []
        gn_data = []
        colors = ["pink", "skyblue", "lightgreen"]
        shapes = ["square", "cross", "8"]

        for i, cls in enumerate(lda_estimator.classes_):
            for j, c_pred in enumerate(lda_estimator.classes_):
                relevant_data_lda = data_X[(data_y == cls) & (lda_pred == c_pred)]
                if relevant_data_lda.size != 0:
                    lda_data.append(go.Scatter(x=relevant_data_lda[:, 0],
                                               y=relevant_data_lda[:, 1],
                                               mode='markers', showlegend=False,
                                               marker=dict(symbol=shapes[i], color=colors[j])))
                relevant_data_gn = data_X[(data_y == cls) & (gn_pred == c_pred)]
                if relevant_data_gn.size != 0:
                    gn_data.append(go.Scatter(x=relevant_data_gn[:, 0],
                                              y=relevant_data_gn[:, 1],
                                              mode='markers', showlegend=False,
                                              marker=dict(symbol=shapes[i], color=colors[j])))

        # creating legend #
        fig.add_traces([go.Scatter(x=[lda_estimator.mu_[0][0]], y=[lda_estimator.mu_[0][1]], mode='markers',
                                   showlegend=True, name=f"true label {int(cls)}",
                                   marker=dict(symbol=shapes[i], color="black")) for i, cls in
                        enumerate(lda_estimator.classes_)])
        fig.add_traces([go.Scatter(x=[lda_estimator.mu_[0][0]], y=[lda_estimator.mu_[0][1]], mode='markers',
                                   showlegend=True, name=f"predicted label {int(cls)}",
                                   marker=dict(symbol="0", color=colors[i])) for i, cls in
                        enumerate(lda_estimator.classes_)])
        ###################

        fig.add_traces(gn_data, rows=1, cols=1)
        fig.add_traces(lda_data, rows=1, cols=2)
        fig.update_layout(title=f"Predicted Labels vs True Labels in Data Set: {f}\n",
                          xaxis_title="x", yaxis_title="y",
                          font=dict(family="Times new roman", size=16), margin=dict(t=80))

        # Add traces for data-points setting symbols and colors - included above)

        # Add `X` dots specifying fitted Gaussians' means
        for i in range(len(lda_estimator.classes_)):
            fig.add_trace(go.Scatter(x=[gaussian_naive_estimator.mu_[i][0]], y=[gaussian_naive_estimator.mu_[i][1]],
                                     mode='markers', showlegend=False,
                                     marker=dict(symbol="x", color="black", size=10)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[lda_estimator.mu_[i][0]], y=[lda_estimator.mu_[i][1]],
                                     mode='markers', showlegend=False,
                                     marker=dict(symbol="x", color="black", size=10)), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda_estimator.classes_)):
            gn_cov = np.diag(gaussian_naive_estimator.vars_[i])
            fig.add_trace(get_ellipse(gaussian_naive_estimator.mu_[i], gn_cov), row=1, col=1)
            fig.add_trace(get_ellipse(lda_estimator.mu_[i], lda_estimator.cov_), row=1, col=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
