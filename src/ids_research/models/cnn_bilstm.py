import numpy as np
import tensorflow as tf
import mlflow
from keras.layers import (
    Input,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    MaxPooling1D,
    Reshape,
)
from keras.models import Sequential


class CNN_BiLSTM:
    def __init__(
        self,
        input_shape=None,
        conv_filters=64,
        lstm_units_1=32,
        lstm_units_2=64,
        dropout_rate=0.6,
        output_units=1,
        output_activation="sigmoid",
    ):
        """
        Initialize CNN_BiLSTM model with configurable parameters.
        Based on: https://doi.org/10.1145/3430199.3430224

        Args:
            input_shape (tuple): Shape of input data (features_dim, 1)
            conv_filters (int): Number of filters in Conv1D layer
            lstm_units_1 (int): Number of units in first LSTM layer
            lstm_units_2 (int): Number of units in second LSTM layer
            dropout_rate (float): Dropout rate before final Dense layer
            output_units (int): Number of output units (1 for binary classification)
            output_activation (str): Activation function for output layer
        """
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.output_units = output_units
        self.output_activation = output_activation
        self.model = None

    def build(self, input_shape=None):
        """
        Build the CNN-BiLSTM model architecture.

        Args:
            input_shape (tuple, optional): Shape of input data (features_dim, 1).
                                          If None, uses the shape provided during initialization.

        Returns:
            keras.models.Sequential: The built model
        """
        if input_shape is not None:
            self.input_shape = input_shape

        if self.input_shape is None:
            raise ValueError(
                "Input shape must be provided either during initialization or when building the model"
            )

        features_dim = self.input_shape[0]

        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Convolutional block
        model.add(
            Conv1D(
                self.conv_filters,
                kernel_size=features_dim,
                padding="same",
                activation="relu",
            )
        )

        # First LSTM block
        model.add(MaxPooling1D(pool_size=8))
        model.add(BatchNormalization())
        model.add(
            Bidirectional(LSTM(self.lstm_units_1, return_sequences=False))
        )
        model.add(Reshape((self.conv_filters, 1)))

        # Second LSTM block
        model.add(MaxPooling1D(pool_size=16))
        model.add(BatchNormalization())
        model.add(
            Bidirectional(LSTM(self.lstm_units_2, return_sequences=False))
        )

        # Output block
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.output_units, activation=self.output_activation))

        self.model = model
        return self.model

    def compile(
        self, loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    ):
        """
        Compile the model with specified loss function, optimizer, and metrics.

        Args:
            loss (str): Loss function to use
            optimizer (str or keras.optimizers.Optimizer): Optimizer to use
            metrics (list): Metrics to track during training

        Returns:
            The compiled model
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self.model

    def fit(self, *args, **kwargs):
        """
        Train the model with the provided data.

        Args:
            *args, **kwargs: Arguments to pass to model.fit()

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")

        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Make predictions with the model.

        Args:
            *args, **kwargs: Arguments to pass to model.predict()

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be built before making predictions")

        return self.model.predict(*args, **kwargs)

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        if self.model is None:
            raise ValueError("Model must be built before showing summary")

        return self.model.summary()

    def save(self, *args, **kwargs):
        """
        Save the model to disk.

        Args:
            *args, **kwargs: Arguments to pass to model.save()
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        return self.model.save(*args, **kwargs)

    def get_model(self):
        """
        Get the underlying Keras model.

        Returns:
            keras.models.Sequential: The model
        """
        return self.model


# Add callback to check for NaN gradients
class TerminateOnNaN(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and np.isnan(loss):
            print("Batch %d: Invalid loss, terminating training" % (batch))
            self.model.stop_training = True

        val_loss = logs.get("val_loss")
        if val_loss is not None and np.isnan(val_loss):
            print(
                "Batch %d: Invalid validation loss, terminating training"
                % (batch)
            )
            self.model.stop_training = True


class EarlyStoppingEpochLogger(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        # The epoch is 0-indexed in history, so add 1
        best_epoch = np.argmin(self.model.history.history["val_loss"]) + 1
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_param(
            "stopped_epoch", len(self.model.history.history["val_loss"])
        )
