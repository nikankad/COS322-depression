from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, precision_recall_curve, f1_score
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

class RandomForestClassifierModel:

    def __init__(self):
        """
        LogisticRegression
        """
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
        self.threshold = 0.5403329501279615
    
    def _prepare_xy(self, df: pd.DataFrame):
        """Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        X = numeric_df.drop(columns=['depression'])
        y = numeric_df['depression']
        return X, y


    def train(self, df):
        X, y = self._prepare_xy(df)
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

    def report(self, y_pred=None, y_test=None):
        # Use stored predictions/tests if not provided
        y_pred = y_pred if y_pred is not None else self.y_pred
        y_test = y_test if y_test is not None else self.y_test

        if y_pred is None or y_test is None or self.X_test is None:
            raise ValueError("Call train() first or provide y_pred and y_test.")
        #roc 
        y_scores = self.model.predict_proba(self.X_test)[:, 1]  # Get the probabilities for the positive class
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # ROC curve
        ax[0].plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        ax[0].plot([0, 1], [0, 1], color='red', linestyle='--')
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('Receiver Operating Characteristic')
        ax[0].legend(loc='lower right')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1], xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
        ax[1].set_ylabel('Actual')
        ax[1].set_xlabel('Predicted')
        ax[1].set_title('Confusion Matrix')

        plt.tight_layout()
        plt.show()

        reportMetrics = classification_report(y_test, y_pred)
        print(reportMetrics)
       
        accuracy = np.mean(y_pred == y_test)
        print(f'Model Accuracy: {accuracy:.2f}')
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
            "n_estimators": [50],  # <- the most important speedup
            "class_weight": ["balanced"]
        }

        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=10,     # <- cut to 10 random tests
            scoring="f1",
            cv=2,          # <- fastest reasonable CV
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        print("Best parameters:", search.best_params_)
        print(classification_report(y_test, y_pred))

    def find_best_threshold(self, X_test, y_test):
        from sklearn.metrics import precision_recall_curve
        import numpy as np

        probs = self.model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        return best_threshold, best_f1, best_precision, best_recall, precisions, recalls, thresholds
