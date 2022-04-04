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
    fig1.show()

    il_std_temp = israel_info.groupby('Month').agg(np.std)
    fig2 = px.bar(il_std_temp['Temp'])
    fig2.update_layout(yaxis={"title": "Daily Temp STD"})
    fig2.show()

    # Question 3 - Exploring differences between countries
    mean_month_temp = df.groupby(['Month', 'Country']).Temp.agg(
        np.mean).reset_index()
    std_month_temp = df.groupby(['Month', 'Country']).Temp.agg(
        np.std).reset_index()
    fig3 = px.line(mean_month_temp, x='Month', y='Temp',
                   color='Country',
                   error_y=std_month_temp['Temp'])
    fig3.update_layout(yaxis={"title": "Avg monthly temp"})
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    y_israel = israel_info['Temp']
    X_israel = israel_info['DayOfYear']
    train_X, train_y, test_X, test_y = split_train_test(X_israel, y_israel)
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()
    pd_loss = {'K degree': [], 'Loss': []}
    for k in range(1, 11):
        poly_model = PolynomialFitting(k).fit(train_X,
                                              train_y)
        loss_val = round(poly_model.loss(test_X, test_y), 2)
        pd_loss['K degree'].append(k)
        pd_loss['Loss'].append(loss_val)
        print(loss_val)

    fig4 = px.bar(pd_loss, x='K degree', y='Loss')
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(6).fit(X_israel, y_israel)
    countries = ['Jordan', 'South Africa', 'The Netherlands']
    country_loss = {'Country': [], 'Loss': []}

    for country in countries:
        country_info = df[df['Country'] == country]
        country_loss['Country'].append(country)
        loss = poly_model.loss(country_info['DayOfYear'], country_info['Temp'])
        country_loss['Loss'].append(loss)

    fig5 = px.bar(country_loss, x='Country', y='Loss')
    fig5.update_layout(yaxis={"title": "Model's error"})
    fig5.show()
