from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split

from utils.helpers import find_best_threshold, prepare_xy, report_metrics


class LogisticRegressionModel:
    def __init__(self, threshold =0.5):
        self.model = LogisticRegression(
            C=0.20, penalty="l1", solver="liblinear", max_iter=2000, 
        )
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.threshold = threshold

    def train(self, df):
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        
        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test
        
        # Store PROBABILITIES, not binary predictions
        self.y_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1
        
        return self.y_proba, y_test

    
    def predict(self, newdf):
        X_new = newdf.select_dtypes(include=["number", "float64"]).drop(columns=["id"])

        y_pred = self.model.predict(X_new)

        result = newdf[["id"]].copy()
        result["y_pred"] = y_pred
        return result.drop_duplicates()

    def optimize(self, df):

        X, y = prepare_xy(df)

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

    def best_threshold(self):
        results = find_best_threshold(self.model, self.X_test, self.y_test)  # Pass X_test, not y_proba
        self.threshold = results[0]
        return results
    def report(self):
        report_metrics(self.model, self.threshold,self.X_test, self.y_test)
        

