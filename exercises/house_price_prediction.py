import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

TRAIN_SET_PORTION = 0.75
REDUNDANT_FEAT = ["id", "date"]
FEATURES = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
            "sqft_lot", "floors", "waterfront", "view", "condition",
            "grade", "sqft_above", "sqft_basement", "yr_built",
            "yr_renovated", "zipcode", "lat", "long", "sqft_living15",
            "sqft_lot15"]
NEEDS_TO_BE_POS = ["price", "yr_built", "sqft_living",
                   "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15",
                   "sqft_lot15", "zipcode"]
NEEDS_TO_BE_NON_NEG = ["bedrooms", "bathrooms",
                       "yr_renovated", "floors"]

LEGAL_VALUES = {"waterfront": [0, 1], "view": [0, 1, 2, 3, 4, 5],
                "condition": [1, 2, 3, 4, 5],
                "grade": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna()

    for col in NEEDS_TO_BE_POS:
        df = df[df[col] > 0]

    for col in NEEDS_TO_BE_NON_NEG:
        df = df[df[col] >= 0]

    df = df.drop(REDUNDANT_FEAT, axis=1)

    for label in LEGAL_VALUES:
        df = df[df[label].isin(LEGAL_VALUES[label])]

    df["zipcode"] = df["zipcode"].astype(int)
    dummies = pd.get_dummies(df["zipcode"], prefix="zip")
    df = pd.concat([df, dummies], axis=1)
    df = df.drop("zipcode", axis=1)

    # RecentRenovation will represent the the relative recentness of the
    # renovation done to the house (including the time it was built,
    # if it hasn't been renovated)
    df["yr_renovated"] = np.max(df[["yr_renovated", "yr_built"]], axis=1)
    df["RecentRenovation"] = df["yr_renovated"] >= np.percentile(
        df["yr_renovated"], 75)
    df["RecentRenovation"] = df["RecentRenovation"].astype(int)

    df = df.drop(["yr_renovated"], axis=1)
    y = df.pop("price")
    return df, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    std_err_y = np.std(y)
    for feat in X:
        std_err_mul = np.std(X[feat]) * std_err_y
        cor = (np.cov(X[feat], y)[0][1]) / std_err_mul
        fig = go.Figure([go.Scatter(x=X[feat], y=y,
                                    mode='markers',
                                    line=dict(width=3))])
        fig.update_layout(barmode='overlay',
                          title=f"Pearson correlation between {feat} & "
                                f"response: {cor}",
                          xaxis_title=f"{feat}",
                          yaxis_title="response",
                          height=500)
        fig.write_image(f"{output_path}{os.sep}feat_{feat}.png")


def calculate_loss_per_percentage(train_x, tr_y, test_x, tst_y):
    """
    implementation of question 3
    :param train_x: train over the data
    :param tr_y: responses of train over the data
    :param test_x: test of the data
    :param tst_y: responses of test of the data
    :return: None
    """
    losses = []
    std_loss_plus = []
    std_loss_minus = []
    lr = LinearRegression(include_intercept=True)
    for p in range(10, 101):
        p_loss = []
        for i in range(10):
            curr_train_x = train_x.sample(frac=p / 100, axis=0)
            cur_y = tr_y.reindex(curr_train_x.index)
            lr.fit(curr_train_x, cur_y)
            p_loss.append(lr.loss(test_x, tst_y))
        std_loss, mean_loss = np.std(p_loss), np.mean(p_loss)
        losses.append(mean_loss)
        std_loss_plus.append(mean_loss + 2 * std_loss)
        std_loss_minus.append(mean_loss - 2 * std_loss)

    x = np.linspace(10, 100, 91).astype(int)
    data = [go.Scatter(x=x, y=losses, mode='lines', name="mean over MSE",
                       showlegend=True),
            go.Scatter(x=x, y=std_loss_plus, mode='lines',
                       name="mean(loss) + 2*std(loss)",
                       line=dict(color='rgb(111, 231, 219)'),
                       showlegend=True),
            go.Scatter(x=x, y=std_loss_minus, mode='lines', fill='tonexty',
                       line=dict(color='rgb(111, 231, 219)'),
                       name="mean(loss) - 2*std(loss)",
                       showlegend=True)]
    fig = go.Figure(data)
    fig.update_layout(
        title="MSE of Test Set as function of percentage of Training set",
        xaxis_title="Percentage").show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, y = load_data("../datasets/house_prices.csv")
    df = pd.DataFrame(df)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, y,
                                                        TRAIN_SET_PORTION)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data For every percentage p in 10%, 11%, ..., 100%, repeat the
    # following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    calculate_loss_per_percentage(train_X, train_y, test_X, test_y)
