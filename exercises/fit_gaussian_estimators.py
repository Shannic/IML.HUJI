from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    X = np.random.normal(mu, 1, 1000)
    u = UnivariateGaussian()
    u.fit(X)
    print("(", u.mu_, u.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    Y = [abs(u.fit(X[:i]).mu_ - mu) for i in ms]

    fig2 = go.Figure([go.Scatter(x=ms, y=Y, mode='markers+lines')])
    fig2.update_layout(
        title=r"$\text{Distance between Estimation of Expectation "
              r"and Expictation As Function "
              r"Of Number Of Samples}$",
        xaxis_title="$\\text{ number of samples}$",
        yaxis_title="r$|\hat\mu-\mu|$",
        height=300).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x_axis = np.linspace(6, 14, 1000)
    pdf_val = u.pdf(x_axis)

    fig3 = go.Figure([go.Scatter(x=x_axis, y=pdf_val, mode='lines',
                                 line=dict(width=3))])
    fig3.update_layout(barmode='overlay',
                       title=r"$\text{PDF of Samples}$",
                       xaxis_title=r"$\text{value}$",
                       yaxis_title=r"$\text{density function}$",
                       height=300).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov_matrix = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov_matrix, 1000)
    multi_var = MultivariateGaussian()
    multi_var.fit(X)
    print(multi_var.mu_)
    print(multi_var.cov_)

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    likelihoods = []
    for f11 in f1:
        likelihoods.append(
            [multi_var.log_likelihood(np.array([f11, 0, f33, 0]),
                                      cov_matrix, X) for f33 in f3])

    fig = go.Figure(data=
                    go.Contour(z=np.array(likelihoods),
                               x=f3,
                               y=f1)
                    )
    fig.update_layout(xaxis_title=r"$\text{f3}$",
                      yaxis_title=r"$\text{f1}$",
                      title=r"$\text{Heatmap of Log-Likelihood on "
                            r"Samples, With } \mu = [f1,0,f3,0]$",
                      height=500, width=500).show()

    # Question 6 - Maximum likelihood
    max_f1, max_f3 = np.unravel_index(np.argmax(likelihoods),
                                      np.array(likelihoods).shape)
    print("log likelihood argmax are: f1=", round(f1[max_f1], 3), "f3=",
          round(f3[max_f3], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
