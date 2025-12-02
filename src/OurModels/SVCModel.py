from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.helpers import prepare_xy


class SVCModel:

    def __init__(self):
        self.model = SVC(probability=True)
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def train(self, df):
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,stratify=y
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

        ax[0].plot(fpr, tpr, color="blue", label=f"ROC (area = {roc_auc:.4})")
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
        print(f"Model Accuracy: {self.model.score(self.X_test, self.y_test):.4}")
