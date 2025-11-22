import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


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
                layers.Dense(units, activation="relu", 
                            kernel_regularizer=keras.regularizers.l2(0.001))
            )
            layers_list.append(layers.Dropout(0.2))
        
        layers_list.append(layers.Dense(self.num_classes, activation="softmax"))
        
        self.model = keras.Sequential(layers_list)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, early_stop = True):
        callbacks = []
        """Train the model"""
        if self.model is None:
            raise ValueError("Build and compile model first")
        if early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
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

    def predict(self, X):
        """Make predictions"""
        
        return self.model.predict(X).argmax(axis=1)
