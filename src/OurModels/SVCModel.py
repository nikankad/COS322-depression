from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.svm import SVC


class SVCModel:

    def __init__(self):
        """
        LogisticRegression
        """
        self.model = SVC(probability=True)
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def _prepare_xy(self, df: pd.DataFrame):
        """Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

        X = numeric_df.drop(columns=["depression", "id"])
        y = numeric_df["depression"]
        return X, y

    def train(self, df):
        import numpy as np

        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test
        # Predict on test set

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
