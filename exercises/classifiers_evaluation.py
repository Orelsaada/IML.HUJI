from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.utils import split_train_test
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
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        callback = lambda perceptron, _1, _2:\
            losses.append(perceptron._loss(X, y))
        model = Perceptron(callback=callback).fit(X, y)
        df = pd.DataFrame({"losses": losses,
                           "iteration": range(1, len(losses)+1)})

        # Plot figure of loss as function of fitting iteration
        fig = px.line(df, x="iteration", y="losses")
        title = f'Model\'s loss as function of the iterations where data is ' \
            f'{n}'
        fig.update_layout(title=title)
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

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def predictions_plot(gnb_predict, lda_predict, X, y):
    lda_predict = lda_predict.reshape(y.shape).astype(str)
    y = y.astype(str)
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        px.scatter(x=X[:, 0], y=X[:, 1], color=lda_predict),
        row=1, col=1
    )

    fig.add_trace(
        px.scatter(x=X[:, 0], y=X[:, 1]),
        row=1, col=2
    )

    fig.update_layout(height=600, width=1500,
                      title_text="Side By Side Subplots")
    # fig = px.scatter(X, x=X[:, 0], y=X[:, 1], color=lda_predict)
    fig.show()

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)
        X_pd = pd.DataFrame(X)
        y = pd.Series(y)

        # Fit models and predict over training set
        train_X, train_y, test_X, test_y = split_train_test(X_pd, y)
        train_X, train_y, test_X, test_y = train_X.to_numpy(), \
                                           train_y.to_numpy(), \
                                           test_X.to_numpy(), test_y.to_numpy()
        lda = LDA().fit(train_X, train_y)
        gnb = GaussianNaiveBayes().fit(train_X, train_y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        predictions_plot(lda.predict(test_X), lda.predict(test_X), test_X,
                         test_y)

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
