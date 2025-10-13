import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils.helpers import printoutput
from sklearn.linear_model import LinearRegression

class LinReg:

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
        self.model = LinearRegression()
        
        #remove nulls
        # self.df.dropna()

    def output(self):
    
            # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])

        # Define target variable
        target_column = 'depression'  # replace with your numeric target

        # Ensure target exists
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' not found or not numeric")

        # Features and target
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]

        # Handle any NaNs in features by imputing with the mean
        X = X.fillna(X.mean())

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")

        # Optional: display coefficients
        coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
        print(coef_df)
                
