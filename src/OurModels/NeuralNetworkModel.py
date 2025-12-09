import os
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from utils.helpers import report_metrics, prepare_xy
import matplotlib.pyplot as plt
class NeuralNetworkModel:

    def __init__(self, threshold=0.5,batch_size=32,epochs=10,validation_split=0.1, early_stop=True,
):
        self.input_dim = 16
        self.num_classes = 2
        self.model = None

        
        self.input_dim = 16
        self.num_classes = 2
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.threshold = threshold

        # new persistent training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stop = early_stop


    def build_model(self, hidden_units=[64, 32]):
        layers_list = [keras.Input(shape=(self.input_dim,))]
        for units in hidden_units:
            layers_list.append(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.001),
                )
            )
            layers_list.append(layers.Dropout(0.2))

        # layers_list.append(layers.Dropout(0.3))

        layers_list.append(layers.Dense(1, activation="sigmoid"))

        self.model = keras.Sequential(layers_list)


        self.model.compile(
            optimizer=keras.optimizers.Adam(3e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def train(self,df):
        callbacks = []
        """Train the model"""
        if self.model is None:
            raise ValueError("Build and compile model first")
        if self.early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                )
            )
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=1,
        )
        

    def evaluate(self, X_test, y_test):
        """Evaluate on test data"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    def report(self):
        """
        Generate comprehensive classification report using stored test data.
        Includes metrics, visualizations, and threshold analysis.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not set. Pass X_test and y_test to load_our_model()")
        
        # Create a wrapper to make the Keras model compatible with report_metrics
        class KerasModelWrapper:
            def __init__(self, nn_model):
                self.nn_model = nn_model
            
            def predict_proba(self, X):
                """
                Return probabilities in sklearn format: (n_samples, 2)
                with columns [P(class=0), P(class=1)]
                """
                probs_class1 = self.nn_model.predict_proba(X)
                probs_class0 = 1 - probs_class1
                return np.column_stack([probs_class0, probs_class1])
        
        # Wrap the model
        wrapped_model = KerasModelWrapper(self)
        
        # Call the comprehensive report_metrics function
        metrics_dict = report_metrics(wrapped_model, self.threshold, self.X_test, self.y_test)
        
        # Store metrics for later access if needed
        self.last_metrics = metrics_dict
        
        return metrics_dict
    
    def predict_proba(self, X):
        """
        Return probability of positive class (class=1) for binary classification.
        """
        probs = self.model.predict(X, verbose=0)
        # Return as 1D array of probabilities for positive class
        return probs.ravel()    
    def save(self, path):
        self.model.save(path)
        print("model saved to ", path)

    def load_our_model(self, path, X_test=None, y_test=None):
        """
        Load a saved Keras model and (optionally) attach test data
        so the user can call `report()` immediately after loading.
        """

        self.model = load_model(path)

        # optionally attach test data
        if X_test is not None and y_test is not None:
            self.X_test = X_test
            self.y_test = y_test

        # ensure model input shape is initialized
        self.model.build((None, self.input_dim))
        self.model.predict(np.zeros((1, self.input_dim)), verbose=0)

        print(f"Model loaded from {path}")
    
  