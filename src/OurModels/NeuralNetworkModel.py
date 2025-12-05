import os
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from utils.helpers import report_metrics, prepare_xy
import matplotlib.pyplot as plt
import shap
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
    
    def compute_shap_values(self, X_background=None, X_explain=None, max_evals=100):
        """
        Compute SHAP values for feature importance analysis.
        
        Parameters
        ----------
        X_background : array-like, optional
            Background data for SHAP (typically a subset of training data).
            If None, uses stored test data.
        X_explain : array-like, optional
            Data to explain. If None, uses stored test data.
        max_evals : int
            Maximum evaluations for DeepExplainer (higher = more accurate but slower).
            
        Returns
        -------
        shap_values : array
            SHAP values for each sample and feature.
        explainer : shap.DeepExplainer
            The SHAP explainer object.
        """
        if self.model is None:
            raise ValueError("Model must be built and trained first")
        
        # Use test data as default
        if X_background is None:
            if self.X_test is None:
                raise ValueError("Provide X_background or set test data via load_our_model()")
            # Use a subset of test data as background (100 samples is usually sufficient)
            X_background = self.X_test[:100]
        
        if X_explain is None:
            if self.X_test is None:
                raise ValueError("Provide X_explain or set test data via load_our_model()")
            X_explain = self.X_test
        
        # Convert to numpy arrays if they're DataFrames
        if hasattr(X_background, 'values'):
            X_background = X_background.values
        if hasattr(X_explain, 'values'):
            X_explain = X_explain.values
        
        # Ensure arrays are float32 for TensorFlow compatibility
        X_background = np.array(X_background, dtype=np.float32)
        X_explain = np.array(X_explain, dtype=np.float32)
        
        # Create SHAP explainer for deep learning models
        explainer = shap.DeepExplainer(self.model, X_background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_explain)
        
        # For binary classification with sigmoid output, handle different return formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Ensure shap_values is 2D: (n_samples, n_features)
        if len(shap_values.shape) > 2:
            shap_values = shap_values.squeeze()
        
        # Final check: should be (n_samples, n_features)
        if len(shap_values.shape) != 2:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}. Expected 2D array.")
        
        self.shap_values = shap_values
        self.shap_explainer = explainer
        self.shap_X_explain = X_explain
        
        print(f"SHAP values computed with shape: {shap_values.shape}")
        
        return shap_values, explainer
    
    def plot_shap_summary(self, X_explain=None, feature_names=None, max_display=16):
        """
        Create SHAP summary plot showing feature importance.
        
        Parameters
        ----------
        X_explain : array-like, optional
            Data to explain. If None, uses data from compute_shap_values().
        feature_names : list of str, optional
            Names of features for the plot.
        max_display : int
            Maximum number of features to display.
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("Run compute_shap_values() first")
        
        if X_explain is None:
            X_explain = self.shap_X_explain
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X_explain, 
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def plot_shap_bar(self, feature_names=None, max_display=16):
        """
        Create SHAP bar plot showing mean absolute feature importance.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names of features for the plot.
        max_display : int
            Maximum number of features to display.
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("Run compute_shap_values() first")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance_ranking(self, feature_names=None):
        """
        Get features ranked by importance based on mean absolute SHAP values.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names of features.
            
        Returns
        -------
        importance_df : pandas.DataFrame
            Features ranked by importance with SHAP values.
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("Run compute_shap_values() first")
        
        import pandas as pd
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]
        
        # Handle potential extra dimensions in shap_values
        shap_vals = self.shap_values
        if len(shap_vals.shape) > 2:
            # Squeeze out extra dimensions
            shap_vals = shap_vals.squeeze()
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Ensure mean_abs_shap is 1D
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Importance_Rank': range(1, len(feature_names) + 1)
        })
        
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df['Importance_Rank'] = range(1, len(feature_names) + 1)
        
        return importance_df
    
    def plot_shap_force(self, instance_idx=0, feature_names=None):
        """
        Create SHAP force plot for a single prediction.
        
        Parameters
        ----------
        instance_idx : int
            Index of instance to explain.
        feature_names : list of str, optional
            Names of features.
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("Run compute_shap_values() first")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]
        
        # Get base value (expected value)
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]
        
        # Create force plot
        shap.force_plot(
            base_value,
            self.shap_values[instance_idx],
            self.shap_X_explain[instance_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, instance_idx=0, feature_names=None, max_display=10):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Parameters
        ----------
        instance_idx : int
            Index of instance to explain.
        feature_names : list of str, optional
            Names of features.
        max_display : int
            Maximum number of features to display.
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("Run compute_shap_values() first")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.input_dim)]
        
        # Get base value
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=base_value,
            data=self.shap_X_explain[instance_idx],
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
