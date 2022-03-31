from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from plotly.subplots import make_subplots

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
    dataframe = pd.read_csv(filename).dropna().drop_duplicates()


    exclude_columns = ['id', 'date', 'zipcode', 'lat', 'long']
    dataframe = dataframe.loc[:, ~dataframe.columns.isin(exclude_columns)]

    positive_columns = ['price', 'sqft_living', 'sqft_lot', 'floors',
                        'sqft_above']
    for pos_col in positive_columns:
        dataframe = dataframe[dataframe[pos_col] > 0]

    non_neg_columns = ['bedrooms', 'bathrooms', 'floors', 'waterfront',
                       'view', 'yr_renovated']
    for non_neg_col in non_neg_columns:
        dataframe = dataframe[dataframe[non_neg_col] >= 0]

    prices = dataframe['price']
    dataframe = dataframe.drop('price', 1)

    return dataframe, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
    correlation = lambda X, Y: np.cov(X, Y) / (np.std(X) * np.std(Y))

    y_title = "Response values"
    for ind in range(X.shape[1]):
        relevant_column = X.iloc[:, ind]
        cor_value = correlation(relevant_column, y)[0][1]
        col_name = X.columns[ind]
        x_title = f'{col_name} values'
        df = pd.DataFrame({x_title: relevant_column, y_title: y})
        title = f'Correlation between {col_name} and response vector.\n' \
                f'Correlation: {cor_value}'
        fig = px.scatter(df, x=x_title, y=y_title, title=title)
        fig.write_image(f'{output_path}\\{col_name}_correlation.png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    dataset = '..\\datasets\\house_prices.csv'
    X, y = load_data(dataset)

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, './graphs')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()

    model = LinearRegression()
    res = []
    for p in range(10, 101):
        fraction = p / 100
        loss_arr = []
        for _ in range(10):
            x_sample = train_X.sample(frac=fraction)
            y_sample = train_y.reindex(x_sample.index)
            model.fit(x_sample.to_numpy(), y_sample.to_numpy())
            loss_arr.append(model.loss(test_X, test_y))

        res.append(loss_arr)

    res = np.array(res)
    p_values = np.array(range(10, 101))

    mean_pred, std_pred = np.mean(res, axis=1), np.std(res, axis=1)
    fig = go.Figure([go.Scatter(x=p_values, y=mean_pred,
                                    mode="markers+lines",
                                    name="Mean Prediction",
                                    line=dict(dash="dash"),
                                    marker=dict(color="green",
                                                opacity=.7)),
                         go.Scatter(x=p_values, y=mean_pred - 2 * std_pred,
                                    fill=None, mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False),
                         go.Scatter(x=p_values, y=mean_pred + 2 * std_pred,
                                    fill='tonexty', mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False), ])


    fig.show()