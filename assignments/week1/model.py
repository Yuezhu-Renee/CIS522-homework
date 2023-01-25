import numpy as np


class LinearRegression:
    """
    A linear regression model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.ndarray
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The outcome variable.

        Returns:
            Nothing.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return np.matmul(X, self.w)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        # Set a minimal value so that the gradient will not explode
        # np.clip(gradient,-1,1)
        # raise NotImplementedError()
        """
        Fit the linear regression model with gradient descent.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The outcome variable.
            lr: Learning Rate.
            epochs: The number of iterations.

        Returns:
            Nothing.
        """
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(epochs):
            y_pred = np.dot(X, self.w) + self.b
            dw, db = (1 / X.shape[0]) * np.dot(X.T, y_pred - y), (
                1 / X.shape[0]
            ) * np.sum(y_pred - y)
            self.w, self.b = self.w - lr * dw, self.b - lr * db
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return np.dot(X, self.w) + self.b
