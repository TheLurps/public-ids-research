import unittest
from unittest import mock
import numpy as np
import tensorflow as tf
from ids_research.models.cnn_bilstm import CNN_BiLSTM


class TestCNNBiLSTM(unittest.TestCase):
    """Test suite for the CNN_BiLSTM model class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Define a simple input shape for testing
        self.input_shape = (10, 1)  # 10 features, 1 channel

        # Generate small sample data for testing predictions
        self.X_sample = np.random.random(
            (5, 10, 1)
        )  # 5 samples, 10 features, 1 channel
        self.y_sample = np.array([0, 1, 0, 1, 0])  # Binary labels

        # Use a small test batch for fit method testing
        self.test_batch = (self.X_sample, self.y_sample)

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        model = CNN_BiLSTM()
        self.assertIsNone(model.input_shape)
        self.assertEqual(model.conv_filters, 64)
        self.assertEqual(model.lstm_units_1, 32)
        self.assertEqual(model.lstm_units_2, 64)
        self.assertEqual(model.dropout_rate, 0.6)
        self.assertEqual(model.output_units, 1)
        self.assertEqual(model.output_activation, "sigmoid")
        self.assertIsNone(model.model)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = CNN_BiLSTM(
            input_shape=self.input_shape,
            conv_filters=32,
            lstm_units_1=16,
            lstm_units_2=32,
            dropout_rate=0.5,
            output_units=2,
            output_activation="softmax",
        )

        self.assertEqual(model.input_shape, self.input_shape)
        self.assertEqual(model.conv_filters, 32)
        self.assertEqual(model.lstm_units_1, 16)
        self.assertEqual(model.lstm_units_2, 32)
        self.assertEqual(model.dropout_rate, 0.5)
        self.assertEqual(model.output_units, 2)
        self.assertEqual(model.output_activation, "softmax")
        self.assertIsNone(model.model)

    def test_build_model_with_init_input_shape(self):
        """Test building the model with input shape provided at initialization."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        built_model = model.build()

        self.assertIsNotNone(model.model)
        self.assertIsInstance(built_model, tf.keras.models.Sequential)
        self.assertEqual(built_model.input_shape[1:], self.input_shape)

    def test_build_model_with_build_input_shape(self):
        """Test building the model with input shape provided at build time."""
        model = CNN_BiLSTM()
        built_model = model.build(input_shape=self.input_shape)

        self.assertIsNotNone(model.model)
        self.assertIsInstance(built_model, tf.keras.models.Sequential)
        self.assertEqual(built_model.input_shape[1:], self.input_shape)

    def test_build_model_with_no_input_shape(self):
        """Test building model without input shape raises ValueError."""
        model = CNN_BiLSTM()

        with self.assertRaises(ValueError):
            model.build()

    def test_compile_model(self):
        """Test compiling the model."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()
        compiled_model = model.compile()

        self.assertIsNotNone(model.model)
        self.assertEqual(compiled_model, model.model)

    def test_compile_without_build(self):
        """Test compiling before building raises ValueError."""
        model = CNN_BiLSTM()

        with self.assertRaises(ValueError):
            model.compile()

    def test_custom_compilation(self):
        """Test compiling with custom parameters."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy", "AUC"],
        )

        # Verify that the compilation succeeded (can't easily verify parameters)
        self.assertIsNotNone(model.model)

    @mock.patch("keras.models.Sequential.fit")
    def test_fit_method(self, mock_fit):
        """Test the model fit method."""
        mock_fit.return_value = "history_obj"

        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()
        model.compile()

        history = model.fit(
            self.X_sample, self.y_sample, epochs=2, batch_size=2
        )

        # Check that fit was called with the right parameters
        mock_fit.assert_called_once_with(
            self.X_sample, self.y_sample, epochs=2, batch_size=2
        )
        self.assertEqual(history, "history_obj")

    def test_fit_without_build_compile(self):
        """Test fit before building/compiling raises ValueError."""
        model = CNN_BiLSTM(input_shape=self.input_shape)

        with self.assertRaises(ValueError):
            model.fit(self.X_sample, self.y_sample)

    @mock.patch("keras.models.Sequential.predict")
    def test_predict_method(self, mock_predict):
        """Test the model predict method."""
        mock_predict.return_value = np.array(
            [[0.2], [0.7], [0.3], [0.8], [0.1]]
        )

        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()
        model.compile()

        predictions = model.predict(self.X_sample)

        # Check that predict was called with the right parameters
        mock_predict.assert_called_once_with(self.X_sample)
        self.assertEqual(predictions.shape, (5, 1))

    def test_predict_without_build(self):
        """Test predict before building raises ValueError."""
        model = CNN_BiLSTM()

        with self.assertRaises(ValueError):
            model.predict(self.X_sample)

    @mock.patch("keras.models.Sequential.summary")
    def test_summary_method(self, mock_summary):
        """Test the model summary method."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()

        model.summary()

        # Check that summary was called
        mock_summary.assert_called_once()

    def test_summary_without_build(self):
        """Test summary before building raises ValueError."""
        model = CNN_BiLSTM()

        with self.assertRaises(ValueError):
            model.summary()

    @mock.patch("keras.models.Sequential.save")
    def test_save_method(self, mock_save):
        """Test the model save method."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        model.build()

        model.save("test_model.h5")

        # Check that save was called with the right parameters
        mock_save.assert_called_once_with("test_model.h5")

    def test_save_without_build(self):
        """Test save before building raises ValueError."""
        model = CNN_BiLSTM()

        with self.assertRaises(ValueError):
            model.save("test_model.h5")

    def test_get_model(self):
        """Test getting the underlying Keras model."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        self.assertIsNone(model.get_model())

        model.build()
        self.assertIsNotNone(model.get_model())
        self.assertEqual(model.get_model(), model.model)

    def test_model_architecture(self):
        """Test key components of the model architecture."""
        model = CNN_BiLSTM(input_shape=self.input_shape)
        built_model = model.build()

        # Count layers (basic architecture validation)
        self.assertGreaterEqual(
            len(built_model.layers), 10
        )  # Should have at least 10 layers

        # Check output layer
        output_layer = built_model.layers[-1]
        self.assertIsInstance(output_layer, tf.keras.layers.Dense)
        self.assertEqual(output_layer.units, model.output_units)
        self.assertEqual(
            output_layer.activation.__name__, model.output_activation
        )


if __name__ == "__main__":
    unittest.main()
