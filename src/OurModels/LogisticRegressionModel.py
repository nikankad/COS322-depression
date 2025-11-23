from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd


class LogisticRegressionModel:
    def __init__(self):
        """
        LogisticRegression
        """
        self.model = LogisticRegression(
            C=0.20, penalty="l1", solver="liblinear", max_iter=2000
        )
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def _prepare_xy(self, df: pd.DataFrame):
        # """ Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

        X = numeric_df.drop(columns=["depression", "id"])
        y = numeric_df["depression"]
        return X, y

    def train(self, df):
        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test
        # Predict on test set
        # add 2 numbers together

        y_pred = self.model.predict(X_test)

        return y_pred, y_test

    def report(self):
        y_pred = self.model.predict(self.X_test)
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(fpr, tpr, color="blue", label=f"ROC (area = {roc_auc:.2f})")
        ax[0].plot([0, 1], [0, 1], "r--")
        ax[0].set(
            xlim=(0, 1),
            ylim=(0, 1.05),
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC Curve",
        )
        ax[0].legend(loc="lower right")

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[1],
            xticklabels=["No Depression", "Depression"],
            yticklabels=["No Depression", "Depression"],
        )
        ax[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")

        plt.tight_layout()
        plt.show()

        print(classification_report(self.y_test, y_pred))
        print(f"Model Accuracy: {self.model.score(self.X_test, self.y_test):.2f}")

    def predict(self, newdf):
        X_new = newdf.select_dtypes(include=["number", "float64"]).drop(columns=["id"])

        y_pred = self.model.predict(X_new)

        result = newdf[["id"]].copy()
        result["y_pred"] = y_pred
        return result.drop_duplicates()

    def optimize(self, df):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        X, y = self._prepare_xy(df)

        param_grid = [
            {
                "solver": ["liblinear"],
                "penalty": ["l1", "l2"],
                "C": np.logspace(-4, 4, 20),
            },
            {
                "solver": ["saga"],
                "penalty": ["l1", "l2", "elasticnet", "none"],
                "C": np.logspace(-4, 4, 20),
                "l1_ratio": [0, 0.5, 1],
            },
        ]

        grid = GridSearchCV(
            LogisticRegression(max_iter=10000), param_grid, cv=3, n_jobs=-1
        )
        grid.fit(X, y)

        self.model = grid.best_estimator_
        self.best_params_ = grid.best_params_
        return self.best_params_
