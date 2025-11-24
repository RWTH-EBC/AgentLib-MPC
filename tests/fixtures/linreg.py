from sklearn.linear_model import LinearRegression


class LinRegTrainer:
    """
    Trains GPR with scikit-learn.
    """

    def __init__(self):
        self.test_linreg = self.build_test_linreg()

    def build_test_linreg(self):
        """
        Builds GPR and returns it.
        """
        linear_model = LinearRegression()
        return linear_model

    def fit_test_linreg(self, data: dict):
        self.test_linreg.fit(
            X=data.get("x"),
            y=data.get("y"),
        )
