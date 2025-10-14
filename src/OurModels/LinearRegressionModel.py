from sklearn.model_selection import train_test_split
from utils.helpers import regression_metrics, regression_graph
from sklearn.linear_model import LinearRegression

class LinearRegressionModel:

    def __init__(self):
        """
        Initialize the Linear Regression model.

        Parameters
        ----------
        df: Dataframe
        """
        self.model = LinearRegression()        
        # remove nulls

    def train(self, df):
        # Select numeric columns only
        df.dropna()
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        # Define target variable
        target_column = 'depression'  # replace with your numeric target

        # Ensure target exists
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' not found or not numeric")

        # Features and target
        X = numeric_df.drop(columns=[target_column, "id"])
        y = numeric_df[target_column]

        # Handle any NaNs in features by imputing with the mean
        X = X.fillna(X.mean())

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Linear Regression model
        self.model.fit(X_train, y_train)
        # Make predictions
        y_pred = self.model.predict(X_test)
        return y_pred, y_test
    def charts(self, y_pred, y_test):
        #generate charts 
        regression_graph(y_pred, y_test)

    def test(self):
        #use test csv to test 
        print("test")


    def output(self, y_pred, y_test):
        regression_metrics("linear", y_pred, y_test)
       
