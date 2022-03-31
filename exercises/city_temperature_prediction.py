import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


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
    df = pd.read_csv(filename, parse_dates=[2]).dropna().drop_duplicates()
    df.insert(0, "DayOfYear", [date.dayofyear for date in df['Date']])
    for column in ['Temp']:
        df = df[df[column] > -10]
    df['Year'] = df['Year'].astype(str)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    dataset = '..\\datasets\\City_Temperature.csv'
    df = load_data(dataset)

    # Question 2 - Exploring data for specific country
    israel_info = df[df['Country'] == 'Israel']
    israel_temp_day = israel_info[['DayOfYear', 'Temp', 'Year']]
    fig1 = px.scatter(israel_temp_day, x='DayOfYear', y='Temp', color="Year")
    # fig1.show()

    a = israel_info.groupby('Month').agg(np.std)
    fig2 = px.bar(a['Temp'])
    # fig2.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()