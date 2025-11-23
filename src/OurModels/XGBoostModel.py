from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd

class XGBoostModel:

    def __init__(self):
        """
        LogisticRegression
        """
        self.model = xgb.XGBClassifier(
        objective='multi:softmax',  # multiclass classification
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False
        )
        self.X_test = None
        self.y_test = None
        self.y_pred = None
    
    def _prepare_xy(self, df: pd.DataFrame):
        """Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        X = numeric_df.drop(columns=['depression'])
        y = numeric_df['depression']
        return X, y


    def train(self, df):
        X, y = self._prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test
        self.y_test = y_test
        # Predict on test set
        y_pred = self.model.predict(X_test)
        return y_pred, y_test
    def report(self, y_pred, y_test):
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
       
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f'Model Accuracy: {accuracy:.2f}')
    def predict(self, newdf):
        """
        Predicts class labels on a new DataFrame using a pre-trained model.
        Returns only the id column and predictions.
        """
        X_new = newdf.select_dtypes(include=['number', 'float64'])
        y_pred = self.model.predict(X_new)

        result = newdf[['id']].copy()
        result['y_pred'] = y_pred
        return result

