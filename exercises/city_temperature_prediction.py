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

    il_std_temp = israel_info.groupby('Month').agg(np.std)
    fig2 = px.bar(il_std_temp['Temp'])
    # fig2.show()

    # Question 3 - Exploring differences between countries
    mean_month_temp = df.groupby(['Month', 'Country']).Temp.agg(
        np.mean).reset_index()
    std_month_temp = df.groupby(['Month', 'Country']).Temp.agg(
        np.std).reset_index()
    fig3 = px.line(mean_month_temp, x='Month', y='Temp',
                   color='Country',
                   error_y=std_month_temp['Temp'])
    # fig3.show()

    # Question 4 - Fitting model for different values of `k`
    y_israel = israel_info['Temp']
    X_israel = israel_info.loc[:, israel_info.columns != 'Temp']
    train_X, train_y, test_X, test_y = split_train_test(X_israel, y_israel)
    poly_model = PolynomialFitting(3).fit(train_X.to_numpy(),
                                          train_y.to_numpy())
    for k in range(1,11):
        poly_model = PolynomialFitting(k).fit(train_X.to_numpy(),
                                              train_y.to_numpy())
        loss_val = poly_model.loss(test_X.to_numpy(), test_y.to_numpy())
        print(loss_val)

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()