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

    def _prepare_xy(self, df: pd.DataFrame, target_column: str = "depression"):
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

        X, y = self._prepare_xy(df, target_column='depression')

        # X.drop(columns=['id'], inplace=True, errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Store test set for charts
        self.X_test = X_test

        # Predict on test set
        y_pred = self.model.predict(X_test)

        return y_pred, y_test
    def report(self):
        #Plot matrix
        cnf_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cnf_matrix, annot=True, fmt ='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        #Classification report
        print(classification_report(*results))

        #AUC ROC
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
        print("AUC ROC ", auc_roc)
