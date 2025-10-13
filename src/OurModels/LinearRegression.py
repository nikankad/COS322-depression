from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import pandas as pd
class LinearRegression:

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Linear Regression model.

        Parameters
        ----------
        **kwargs : dict
            Any valid sklearn.linear_model.LinearRegression parameter.
            Example: fit_intercept=True, copy_X=True, n_jobs=None
        """
        self.df = df.copy()
        self.model = SklearnLinearRegression()
    
    def dede(self):
        print(self.df.head())