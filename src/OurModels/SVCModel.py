from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SVCModel:

    def __init__(self):
        """
        Initialize the SVC model.
        """
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)        
        
        # Store test data for charts
        self.X_test = None
        self.y_test = None
        

    def train(self, df):
        df.dropna(inplace=True)
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        # Define target variable
        target_column = 'depression'  # replace with your numeric target

        # Ensure target exists
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' not found or not numeric")

        # Features and target
        X = numeric_df.drop(columns=[target_column, "id"])
        y = numeric_df[target_column]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the SVC model
        self.model.fit(X_train, y_train)


        # Store test data for charts
        self.X_test = X_test
        self.y_test = y_test

        # Make predictions
        y_pred = self.model.predict(X_test)
      
        return y_pred, y_test
    
