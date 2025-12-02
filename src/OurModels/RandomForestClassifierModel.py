from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from utils.helpers import find_best_threshold, prepare_xy, report_metrics

class RandomForestClassifierModel:

    def __init__(self, threshold = 0.5):
        self.model = RandomForestClassifier(
            n_estimators=50,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features='sqrt',
            max_depth=None,
            class_weight='balanced',
            ccp_alpha=0.0,
            random_state=42
        )

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.threshold = threshold
    


    def train(self, df):
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test
        # Predict on test set using custom threshold
        y_scores = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores >= self.threshold).astype(int)
        self.y_pred = y_pred
        return y_pred, y_test
    def predict(self, newdf):
        X_new = newdf.select_dtypes(include=['number'])
        probs = self.model.predict_proba(X_new)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        result = newdf[['id']].copy()
        result['y_pred'] = preds
        return result

    def optimize(self, df):
        # MASSIVELY reduced search space
        param_dist = {
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt"],
            "ccp_alpha": [0.0, 0.01],
            "n_estimators": [50],
            "class_weight": ["balanced"]
        }

        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=10,
            scoring="f1",
            cv=2,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        print("Best parameters:", search.best_params_)
        print(classification_report(y_test, y_pred))

    def best_threshold(self):
        results = find_best_threshold(self.model, self.X_test, self.y_test)  # Pass X_test, not y_proba
        self.threshold = results[0]
        return results
    def report(self):
        report_metrics(self.model, self.threshold, self.X_test, self.y_test)
        
