from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CatBoost:

    def __init__(self, threshold=0.5):

        # Default base model
        self.model = CatBoostClassifier(
            iterations=300,
            depth=10,
            learning_rate=0.05,
            l2_leaf_reg=5,
            class_weights=[1.0, 3.0],
            border_count=128,
            bagging_temperature=1,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=42,
            verbose=False
        )


        self.X_test = None
        self.y_test = None
        self.threshold = threshold

    # ---------------------------------------------------------
    def _prepare_xy(self, df):
        numeric_df = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])
        X = numeric_df.drop(columns=["depression", "id"])
        y = numeric_df['depression']
        return X, y

    # ---------------------------------------------------------
    def train(self, df):
        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        self.X_test = X_test
        self.y_test = y_test

    # ---------------------------------------------------------
    def tune(self, df):
        X, y = self._prepare_xy(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Base model for search
        base = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='F1',
            verbose=False,
            random_seed=42
        )

        # Hyperparameter search space
        param_dist = {
            'iterations': [200, 300, 500, 700],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7, 10],
            'border_count': [64, 128, 254],
            'bagging_temperature': [0, 1, 2, 3],
            'class_weights': [
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
                [1.0, 6.0]
            ]
        }

        tuner = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=20,
            scoring='recall',
            cv=3,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        tuner.fit(X_train, y_train)

        print("\nBest Parameters Found:")
        print(tuner.best_params_)

        # Update the model inside this class
        self.model = tuner.best_estimator_

        # Evaluate tuned model
        y_pred = self.model.predict(X_test)
        print("\nClassification Report (Tuned Model):")
        print(classification_report(y_test, y_pred))

        # Save test split inside class
        self.X_test = X_test
        self.y_test = y_test

    # ---------------------------------------------------------
    def predict(self, newdf):
        X_new = newdf.select_dtypes(include=['number'])
        probs = self.model.predict_proba(X_new)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        result = newdf[['id']].copy()
        result['y_pred'] = preds
        return result

    # ---------------------------------------------------------
    def report(self):
        probs = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (probs >= self.threshold).astype(int)

        print(classification_report(self.y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        ax[0].set_title("ROC Curve")
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].legend()

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
        ax[1].set_title(f"Confusion Matrix (threshold={self.threshold:.3f})")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    def find_best_threshold(self):
        probs = self.model.predict_proba(self.X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(self.y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)

        best_idx = np.argmax(f1_scores)

        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        self.threshold = 0.2  # store internally

        return best_threshold, best_f1, best_precision, best_recall
