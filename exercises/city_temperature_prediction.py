import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

import plotly.graph_objects as go


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna()
    df = df[df["Day"].isin(range(1, 32))]
    df = df[df["Month"].isin(range(1, 13))]
    df = df[df["Year"] > 0]
    df = df[df["Temp"] > -60]
    df["DayOfYear"] = df["Date"].apply(lambda x: x.timetuple().tm_yday)
    return df


def explore_israel(df):
    """
    Implementation of question 3 - exploring the data for Israel
    :param df: the data frame that includes our total data
    :return: None
    """
    isr_df = df[df["Country"] == "Israel"]
    # part 1
    years = set(isr_df["Year"])
    data = []
    for yr in years:
        same_year_df = isr_df[isr_df["Year"] == yr]
        data.append(go.Scatter(x=same_year_df["DayOfYear"],
                               y=same_year_df["Temp"],
                               mode='markers', name=f"{yr}",
                               showlegend=True))
    fig = go.Figure(data)
    fig.update_layout(
        title="Temp of the Day of Year",
        xaxis_title="Day of Year",
        yaxis_title="Temp").show()

    # part 2
    std = isr_df.groupby(["Month"])["Temp"].agg('std')
    months = range(1, 13)
    px.bar(pd.DataFrame({"Std": std, "Month": months}), x="Month", y="Std",
           text=months,
           title="Standard deviation of Temp per month").show()


def question_3(df):
    """
    Implementation of question 3
    :param df: the data frame that includes our total data
    :return: None
    """
    months = list(range(1, 13))
    std = df.groupby(["Country", "Month"])["Temp"].agg('std')
    mean = df.groupby(["Country", "Month"])["Temp"].agg('mean')
    countries = set(df["Country"])
    data = []
    for ct in countries:
        data.append(go.Scatter(x=months,
                               y=mean[ct],
                               mode='markers+lines', name=f"{ct}",
                               showlegend=True, error_y=dict(type="data",
                                                             array=std[ct],
                                                             visible=True)))
    fig = go.Figure(data)
    fig.update_layout(
        title="Average Monthly Temperature and Standard Deviation for Each "
              "Country",
        xaxis_title="Month",
        yaxis_title="mean").show()


def question_4(df):
    """
    Implementation of question 4
    :param df: the data frame that includes our total data
    :return: None
    """
    isr_df = df[df["Country"] == "Israel"]
    train_x, train_y, test_x, test_y = split_train_test(isr_df["DayOfYear"],
                                                        isr_df["Temp"])
    losses = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(np.array(train_x), np.array(train_y))
        losses.append(round(poly_fit.loss(np.array(test_x), np.array(
            test_y)), 2))
    print(losses)
    px.bar(pd.DataFrame({"MSE": losses, "Degree": range(1, 11)}),
           x="Degree",
           y="MSE",
           title="Losses Over Test Set According to Degree of Polynomial "
                 "Fit").show()


def question_5(df):
    """
    Implementation of question 4
    :param df: the data frame that includes our total data
    :return: None
    """
    isr_df = df[df["Country"] == "Israel"]
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(np.array(isr_df["DayOfYear"]), np.array(isr_df["Temp"]))
    countries = ['South Africa', 'The Netherlands', 'Jordan']
    losses = []
    for ct in countries:
        ct_df = df[df["Country"] == ct]
        losses.append(poly_fit.loss(np.array(ct_df["DayOfYear"]),
                                    np.array(
                                        ct_df["Temp"])))

    px.bar(pd.DataFrame({"Countries": countries, "MSE": losses}),
           x="Countries", y="MSE",
           title="Loss Over The Model Fit for Israel").show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    explore_israel(df)

    # Question 3 - Exploring differences between countries
    question_3(df)

    # Question 4 - Fitting model for different values of `k`
    question_4(df)

    # Question 5 - Evaluating fitted model on different countries
    question_5(df)
