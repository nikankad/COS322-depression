from sklearn.model_selection import train_test_split
from utils.helpers import generate_classification_charts, classification_metric
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class RandomForestRegressorModel:

    def __init__(self, n_estimators: int = 200, max_depth: int = None, random_state: int = 42):
        """
        RandomForest wrapper with the same API as your SVCModel.

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
        df.dropna(inplace=True)

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

    def charts(self, y_pred, y_test):
        """
        Generate classification charts. Uses predict_proba for probability-based charts.
        """
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("No test data available. Run train() first.")

        # Probabilities for positive class
        if hasattr(self.model, "predict_proba"):
            y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        else:
            # Fallback: use decision_function if available and map to 0-1 (not ideal)
            try:
                scores = self.model.decision_function(self.X_test)
                y_pred_prob = (scores - scores.min()) / (scores.max() - scores.min())
            except Exception:
                # final fallback: binary predictions as 0/1 probabilities
                y_pred_prob = self.model.predict(self.X_test)

        # Delegate to your helper plotting function
        generate_classification_charts(y_pred, y_test, y_pred_prob)

    def output(self, y_pred, y_test):
        """
        Produce metrics and additional info like OOB score and feature importances.
        """
        # Call your existing metric function (keeps same signature)
        classification_metric("random_forest", y_pred, y_test)

        # Print useful RF-specific diagnostics
        try:
            oob = getattr(self.model, "oob_score_", None)
            if oob is not None:
                print(f"OOB score: {oob:.4f}")
        except Exception:
            pass

        # Feature importances (if features exist)
        try:
            importances = self.model.feature_importances_
            feature_names = list(self.X_test.columns) if self.X_test is not None else None
            if feature_names is not None:
                fi_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                print("Top feature importances:")
                print(fi_df.head(15))
        except Exception:
            pass

    def test(self):
        # Placeholder: keep same API
        print("test")
