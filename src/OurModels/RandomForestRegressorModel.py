from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class RandomForestRegressorModel:

    def __init__(self, n_estimators: int = 500, max_depth: int = None, random_state: int = 42):
        """        RandomForest wrapper with the same API as your SVCModel.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int or None
            Maximum depth of each tree.
        random_state : int
            RNG seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            oob_score=True,
            n_jobs=-1,
            random_state=random_state,
            class_weight=None  # consider 'balanced' if classes are imbalanced
        )

        # Stored test set for charts
        self.X_test = None
        self.y_test = None

    def _prepare_xy(self, df: pd.DataFrame, target_column: str = "depression"):
        """Prepare X, y from df: drop NA, select numeric cols, handle id if present."""
        df = df.copy()
        # df.dropna(inplace=True)

        numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' not found or not numeric")

        drop_cols = [target_column]
        if "id" in numeric_df.columns:
            drop_cols.append("id")

        X = numeric_df.drop(columns=drop_cols)
        y = numeric_df[target_column]
        return X, y

    def train(self, df: pd.DataFrame):
        """
        Train the RandomForest on numeric features of df and return (y_pred, y_test).
        """

        X, y = self._prepare_xy(df, target_column="depression")
        X.drop(columns=['id'], inplace=True, errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.model.random_state, stratify=y if len(np.unique(y))>1 else None
        )

        # Fit the model
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test

        # Predict on test set
        y_pred = self.model.predict(X_test)

        return y_pred, y_test


    def output(df):
        print("test")
    def predict(self):
        return 1