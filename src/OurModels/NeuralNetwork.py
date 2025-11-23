import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


# !add optimize class
# class LRFinder(keras.callbacks.Callback):
#     def


class NeuralNetwork:

    def __init__(self):
        """
        LogisticRegression
        """
        self.input_dim = 17
        self.num_classes = 2
        self.model = None

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
            optimizer=keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def prepare_data(self, df):
        numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])
        X = numeric_df.drop(columns=["depression, id"])
        y = numeric_df["depression"]

        """Preprocess and split data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        self.input_dim = X_test.shape[1]
        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        early_stop=True,
    ):
        callbacks = []
        """Train the model"""
        if self.model is None:
            raise ValueError("Build and compile model first")
        if early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                )
            )
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
        )

    def evaluate(self, X_test, y_test):
        """Evaluate on test data"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    def report(self, y_pred, y_test, y_scores):
        """
        y_pred   = thresholded predictions (0 or 1)
        y_test   = true labels (0 or 1)
        y_scores = raw probabilities for class 1 (float)
        """

        from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # -------------------------
        # ROC Curve + AUC
        # -------------------------
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax[0].plot([0, 1], [0, 1], '--')
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_title("ROC Curve")
        ax[0].legend()

        # -------------------------
        # Confusion Matrix
        # -------------------------
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[1],
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
        )
        ax[1].set_title("Confusion Matrix")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")

        plt.tight_layout()
        plt.show()

        # -------------------------
        # Classification Report
        # -------------------------
        print(classification_report(y_test, y_pred))

        # -------------------------
        # Accuracy
        # -------------------------
        acc = np.mean(y_pred == y_test)
        print(f"Accuracy: {acc:.4f}")


    def save(self):
        path = os.environ["IMAGE_LOCATION"] + "/model.keras"
        self.model.save(path)
        print("model saved to ", path)

    def load_our_model(self):
        path = os.environ["IMAGE_LOCATION"]
        path = path + "/model.keras"
        self.model = load_model(path)
        print(f"Model loaded from {path}")
    def predict_proba(self, X):
        probs = self.model.predict(X)
        probs = np.array(probs, dtype=float)
        if probs.ndim == 1:
            probs = probs[:, None]
        if probs.shape[1] == 2:
            return probs[:, 1]   # return P(class=1)
        return probs[:, 0]        # sigmoid case

