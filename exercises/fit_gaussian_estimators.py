from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import math
pio.templates.default = "simple_white"

NDIGITS = 4


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma_squared = 1
    samples_num = 1000
    samples = np.random.normal(mu, sigma_squared, samples_num)
    fitted = UnivariateGaussian().fit(samples)
    print(f'({fitted.mu_}, {fitted.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    samples_size = [x for x in range(10, 1001, 10)]
    mu_values = []
    for size in samples_size:
        fitted.fit(samples[:size])
        mu_values.append(abs(fitted.mu_ - mu))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples_size, y=mu_values))
    fig.update_layout(title="Expectation error as a function of samples "
                            "number",
                      xaxis_title="Samples number",
                      yaxis_title="Expectation error")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = fitted.pdf(samples)
    x_title = "Sample values"
    y_title = "PDF values"
    df = pd.DataFrame({x_title: samples, y_title: pdf_values})
    title = "PDF graph - Gaussian bell curve"
    fig = px.scatter(df, x=x_title, y=y_title, title=title)
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_vec = np.array([0, 0, 4, 0])
    sigma_mat = np.array([[1, 0.2, 0, 0.5],
                          [0.2, 2, 0, 0],
                          [0, 0, 1, 0],
                          [0.5, 0, 0, 1]])
    samples_number = 1000
    samples = np.random.multivariate_normal(mu_vec, sigma_mat, samples_number)
    fitted = MultivariateGaussian().fit(samples)
    print(f'Estimated expectation: {fitted.mu_}')
    print(f'Estimated covariance matrix: {fitted.cov_}')

    # Question 5 - Likelihood evaluation
    likelihood_matrix = []

    # max_values = [likelihood, f1, f3]
    max_values = [-math.inf, 0, 0]

    space = np.linspace(-10, 10, 200)
    expec_mu = np.array([space[0], 0, space[0], 0])

    for value1 in space:
        row = []
        for value3 in space:
            expec_mu[0], expec_mu[2] = value1, value3
            likelihood = fitted.log_likelihood(expec_mu, sigma_mat, samples)
            if likelihood > max_values[0]:
                max_values = [likelihood, value1, value3]
            row.append(likelihood)
        likelihood_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(z=likelihood_matrix, x=space,
                                    y=space))
    fig.update_layout(title="Likelihood heatmap of (f3,f1) values",
                      xaxis_title="f3 values",
                      yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood.
    print(f'({round(max_values[2], NDIGITS)},'
          f' {round(max_values[1], NDIGITS)})')
    print(f'Maximum log-likelihood: {round(max_values[0], NDIGITS)}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
