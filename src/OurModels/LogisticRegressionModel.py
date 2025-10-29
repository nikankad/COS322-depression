from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
from sklearn import metrics
import seaborn as sns
class LogisticRegressionModel:

    def __init__(self):
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
        self.model = LogisticRegression(max_iter=5000)

        # Stored test set for charts
        self.X_test = None
        self.y_test = None

    def _prepare_xy(self, df: pd.DataFrame):
        df = df.select_dtypes(include=['float64', 'int64'])
        df.dropna(inplace=True)
        X = df.iloc[:, : -1]
        #Target values
        y = df['depression']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y

    def train(self, df: pd.DataFrame):
        """
        Train the RandomForest on numeric features of df and return (y_pred, y_test).
        """

        X, y = self._prepare_xy(df)

        # X.drop(columns=['id'], inplace=True, errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test

        # Predict on test set
        y_pred = self.model.predict(X_test)

        return y_pred, y_test
    def output(self, y_pred, y_test):
        #Plot matrix
        # Confusion Matrix
        confusion_mtx = metrics.confusion_matrix(y_test, self.model.predict(self.X_test))
        sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
        plt.pyplot.title('Confusion Matrix')
        plt.pyplot.xlabel('Predicted')
        plt.pyplot.ylabel('Actual')
        plt.pyplot.show()

        # R^2 Score
        r2_score = metrics.r2_score(y_test, self.model.predict(self.X_test))
        print(f'R^2 Score: {r2_score}')

        # ROC Curve
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.pyplot.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.pyplot.plot([0, 1], [0, 1], 'k--')
        plt.pyplot.xlim([0.0, 1.0])
        plt.pyplot.ylim([0.0, 1.05])
        plt.pyplot.xlabel('False Positive Rate')
        plt.pyplot.ylabel('True Positive Rate')
        plt.pyplot.title('Receiver Operating Characteristic')
        plt.pyplot.legend(loc='lower right')
        plt.pyplot.show()
