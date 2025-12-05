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
    def _get_weights_flat(self):
        """Return model weights as a flat 1D vector and shapes for reconstruction."""
        weights = self.model.get_weights()
        flats = [w.reshape(-1) for w in weights]
        flat_vec = np.concatenate(flats)
        shapes = [w.shape for w in weights]
        sizes = [w.size for w in weights]
        return flat_vec, shapes, sizes

    def _set_weights_flat(self, flat_vec, shapes, sizes):
        """Set model weights from a flat 1D vector using stored shapes/sizes."""
        new_weights = []
        idx = 0
        for shape, size in zip(shapes, sizes):
            w_flat = flat_vec[idx:idx+size]
            new_weights.append(w_flat.reshape(shape))
            idx += size
        self.model.set_weights(new_weights)

    def _sample_directions(self, flat_origin, n_dirs=2, seed=None):
        """Sample random directions in parameter space and L2-normalize them."""
        rng = np.random.RandomState(seed)
        dirs = []
        for _ in range(n_dirs):
            d = rng.normal(size=flat_origin.shape)
            # normalize by norm of origin to have comparable scale
            d = d * (np.linalg.norm(flat_origin) / (np.linalg.norm(d) + 1e-8))
            dirs.append(d)
        return dirs

    def compute_loss_landscape(
        self,
        X_val,
        y_val,
        grid_points=25,
        span=1.0,
        seed=None,
        return_grids=False,
    ):
        """
        Compute a 2D loss landscape slice around current weights.

        Parameters
        ----------
        X_val, y_val : array-like
            Data used to evaluate loss (use a fixed validation subset).
        grid_points : int
            Number of points along each axis (a,b).
        span : float
            How far to move along each direction (in L2-scaled units).
        seed : int or None
            Random seed for reproducible directions.
        return_grids : bool
            If True, also return (A, B) coordinate grids.

        Returns
        -------
        Z : (grid_points, grid_points) array
            Loss values on the grid.
        (optional) A, B : 2D grids of a, b values.
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before computing loss landscape.")

        # cache original weights
        flat_origin, shapes, sizes = self._get_weights_flat()

        # sample two directions
        d0, d1 = self._sample_directions(flat_origin, n_dirs=2, seed=seed)

        # build grid
        a_vals = np.linspace(-span, span, grid_points)
        b_vals = np.linspace(-span, span, grid_points)
        Z = np.zeros((grid_points, grid_points), dtype=np.float32)

        # iterate grid
        for i, a in enumerate(a_vals):
            for j, b in enumerate(b_vals):
                flat_w = flat_origin + a * d0 + b * d1
                self._set_weights_flat(flat_w, shapes, sizes)
                # evaluate loss only (index 0 of evaluate)
                loss = self.model.evaluate(X_val, y_val, verbose=0)[0]
                Z[j, i] = loss

        # restore original weights
        self._set_weights_flat(flat_origin, shapes, sizes)

        if return_grids:
            A, B = np.meshgrid(a_vals, b_vals)
            return Z, A, B
        return Z

    def plot_loss_landscape_contour(
        self,
        X_val,
        y_val,
        grid_points=25,
        span=1.0,
        levels=20,
        seed=None,
        ax=None,
    ):
        """
        Compute and plot a 2D contour loss landscape.

        Returns
        -------
        ax : matplotlib Axes
        """
        Z, A, B = self.compute_loss_landscape(
            X_val, y_val,
            grid_points=grid_points,
            span=span,
            seed=seed,
            return_grids=True,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        cs = ax.contour(A, B, Z, levels=levels, cmap="magma")
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_xlabel("direction 1 (a)")
        ax.set_ylabel("direction 2 (b)")
        ax.set_title("Loss landscape (contour)")

        return ax

    def plot_loss_landscape_surface(
        self,
        X_val,
        y_val,
        grid_points=25,
        span=1.0,
        seed=None,
        elev=30,
        azim=-60,
    ):
        """
        Compute and plot a 3D surface loss landscape.

        Returns
        -------
        ax : 3D matplotlib Axes
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        Z, A, B = self.compute_loss_landscape(
            X_val, y_val,
            grid_points=grid_points,
            span=span,
            seed=seed,
            return_grids=True,
        )

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(A, B, Z, cmap="magma", linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("direction 1 (a)")
        ax.set_ylabel("direction 2 (b)")
        ax.set_zlabel("loss")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("Loss landscape (3D surface)")
        return ax