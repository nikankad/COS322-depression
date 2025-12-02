from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from utils.helpers import find_best_threshold, prepare_xy, report_metrics



class CatBoostModel:

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

    
    def _prepare_xy(self, df):
        return prepare_xy(df)

    
    def train(self, df):
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)
        
        # Store test set and probabilities
        self.X_test = X_test
        self.y_test = y_test
        self.y_proba = self.model.predict(X_test, prediction_type='Probability')[:, 1]  # CatBoost syntax
        
        return self.y_proba, y_test
    
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

    
    def predict(self, newdf):
        X_new = newdf.select_dtypes(include=['number'])
        probs = self.model.predict_proba(X_new)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        result = newdf[['id']].copy()
        result['y_pred'] = preds
        return result

    
    def report(self):
        report_metrics(self.model, self.threshold, self.X_test, self.y_test)  # Pass y_test, not y_pred
        

    
    def best_threshold(self):
        results = find_best_threshold(self.model, self.X_test, self.y_test)
        self.threshold = results[0]
        return results
