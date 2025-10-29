#use cross validation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.helpers import generate_classification_charts
from sklearn.model_selection import cross_val_score, KFold

class DecisionTreeClassifierModel:
    def __init__(self, max_depth:None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def _prepare_xy(self, df: pd.DataFrame, target_column: str = "depression"):
        df = df.copy()
        # df.drop('id')

        # Require target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in df.")

        # Keep id if present (aligned later)
        ids = df['id'].copy() if 'id' in df.columns else pd.Series(index=df.index, dtype='int64')

        # Only drop rows with missing target; do NOT drop just because a feature is NaN
        df = df[df[target_column].notna()]

        # Select numeric features; coerce non-numerics to NaN and then we'll impute
        numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).copy()

        # Ensure target is numeric
        if target_column not in numeric_df.columns:
            # If target was non-numeric type, bring it in and coerce
            numeric_df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

        # Align ids to numeric_df
        ids = ids.loc[numeric_df.index] if not ids.empty else pd.Series(index=numeric_df.index, dtype='int64')

        # Split X/y and remove id/target from features
        drop_cols = [c for c in ['id', target_column] if c in numeric_df.columns]
        X = numeric_df.drop(columns=drop_cols, errors='ignore')

        # If X is empty after numeric-only selection, fail fast with a clear message
        if X.shape[1] == 0:
            raise ValueError("No numeric feature columns left in X after preprocessing. "
                             "Add feature engineering or one-hot encode categoricals before this step.")

        # Impute numeric NaNs with column medians (simple, fast)
        X = X.fillna(X.median(numeric_only=True))

        y = pd.to_numeric(numeric_df[target_column], errors='coerce')
        # Drop any rows where y is still NaN after coercion
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        ids = ids.loc[valid_idx]

        if X.shape[0] == 0:
            raise ValueError("After filtering invalid/missing targets, there are 0 samples left.")

        return X, y

    def train(self, df: pd.DataFrame):
        """
        Train on numeric features and return:
        y_pred, y_test, id_test, X_test
        """
        X, y = self._prepare_xy(df, target_column="depression")

        # Stratify only if more than one class present
        stratify = y if y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify=stratify)

        self.model.fit(X_train, y_train)

        self.X_test = X_test
        self.y_test = y_test

        y_pred = self.model.predict(X_test)
        return y_pred, y_test
    

    def output(self, y_pred, y_test):  
        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", accuracy_score(y_test, y_pred))
    
    def crossproduct(self):
        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        cross_val_results = cross_val_score(self.model, X, y, cv=kf)
        print("Cross-Validation Results (Accuracy):")
        for i, result in enumerate(cross_val_results, 1):
            print(f"  Fold {i}: {result * 100:.2f}%")
            
        print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')