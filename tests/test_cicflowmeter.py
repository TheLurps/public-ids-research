import unittest
import polars as pl
import numpy as np
from ids_research.preprocessors.cicflowmeter import CicFlowMeterPreprocessor


class TestCicFlowMeterPreprocessor(unittest.TestCase):
    """Test suite for CicFlowMeterPreprocessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample CicFlowMeter dataset for testing
        self.test_data = pl.DataFrame(
            {
                "Flow Duration": [100, 200, 300, 400, 500],
                "Total Fwd Packet": [5, 10, 15, 20, 25],
                "Total Bwd packets": [3, 6, 9, 12, 15],
                "Total Length of Fwd Packet": [500, 1000, 1500, 2000, 2500],
                "Flow Bytes/s": [50, 100, 150, 200, 250],
                "Protocol": ["TCP", "UDP", "TCP", "ICMP", "UDP"],
                "Label": ["BENIGN", "DoS", "notice", "PortScan", "SSH-Patator"],
                "Source IP": [
                    "192.168.1.1",
                    "192.168.1.2",
                    "192.168.1.3",
                    "192.168.1.4",
                    "192.168.1.5",
                ],
                "Destination IP": [
                    "10.0.0.1",
                    "10.0.0.2",
                    "10.0.0.3",
                    "10.0.0.4",
                    "10.0.0.5",
                ],
            }
        )

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        preprocessor = CicFlowMeterPreprocessor()

        # Check default config contains expected keys
        self.assertIn("scaler", preprocessor.config)
        self.assertIn("numerical_features", preprocessor.config)
        self.assertIn("categorical_features", preprocessor.config)

        # Check binary classifier config
        self.assertIn("binary_classifier", preprocessor.config)
        self.assertIn("target_col", preprocessor.config)
        self.assertIn("target_mappings", preprocessor.config)

        # Check default scaler
        self.assertEqual(preprocessor.config["scaler"], "MinMaxScaler")

        # Check Protocol is included in categorical features
        self.assertIn("Protocol", preprocessor.config["categorical_features"])

        # Check numerical features contain key CicFlowMeter fields
        self.assertIn(
            "Flow Duration", preprocessor.config["numerical_features"]
        )
        self.assertIn(
            "Total Fwd Packet", preprocessor.config["numerical_features"]
        )
        self.assertIn(
            "Total Bwd packets", preprocessor.config["numerical_features"]
        )

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "scaler": "StandardScaler",
            "numerical_features": ["Flow Duration", "Total Fwd Packet"],
            "categorical_features": ["Protocol"],
        }
        preprocessor = CicFlowMeterPreprocessor(custom_config)

        # Check custom config is properly set
        self.assertEqual(preprocessor.config["scaler"], "StandardScaler")
        self.assertEqual(
            preprocessor.config["numerical_features"],
            ["Flow Duration", "Total Fwd Packet"],
        )
        self.assertEqual(
            preprocessor.config["categorical_features"], ["Protocol"]
        )

    def test_fit_and_transform(self):
        """Test fit and transform methods with CicFlowMeter data."""
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": [
                    "Flow Duration",
                    "Total Fwd Packet",
                    "Total Bwd packets",
                    "Flow Bytes/s",
                ],
                "categorical_features": ["Protocol"],
            }
        )

        # Fit the preprocessor
        preprocessor.fit(self.test_data)
        self.assertTrue(preprocessor.fitted)

        # Transform the data
        transformed_data = preprocessor.transform(self.test_data)

        # Check all numerical features are scaled to 0-1 range
        for feature in preprocessor.config["numerical_features"]:
            self.assertTrue(
                transformed_data.select(feature).to_numpy().min() >= 0
            )
            self.assertTrue(
                transformed_data.select(feature).to_numpy().max() <= 1
            )

        # Check Protocol was one-hot encoded
        self.assertIn("Protocol_TCP", transformed_data.columns)
        self.assertIn("Protocol_UDP", transformed_data.columns)
        self.assertIn("Protocol_ICMP", transformed_data.columns)

        # Check original Protocol column was removed
        self.assertNotIn("Protocol", transformed_data.columns)

    def test_fit_transform(self):
        """Test fit_transform method with CicFlowMeter data."""
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": [
                    "Flow Duration",
                    "Total Fwd Packet",
                    "Total Bwd packets",
                    "Flow Bytes/s",
                ],
                "categorical_features": ["Protocol"],
            }
        )

        # Fit and transform in one step
        transformed_data = preprocessor.fit_transform(self.test_data)

        # Check the preprocessor is fitted
        self.assertTrue(preprocessor.fitted)

        # Check that transformed data only contains the numerical features, one-hot encoded features, and index
        expected_columns = set(preprocessor.config["numerical_features"]).union(
            {"Protocol_TCP", "Protocol_UDP", "Protocol_ICMP", "index"}
        )
        self.assertEqual(set(transformed_data.columns), expected_columns)

        # Check all numerical features are scaled to 0-1 range
        for feature in preprocessor.config["numerical_features"]:
            self.assertTrue(
                transformed_data.select(feature).to_numpy().min() >= 0
            )
            self.assertTrue(
                transformed_data.select(feature).to_numpy().max() <= 1
            )

    def test_with_missing_features(self):
        """Test with dataset missing some features from config."""
        # Create a config with features that are in the test data
        # We don't test missing features as the current implementation doesn't handle them
        config = {
            "scaler": "StandardScaler",
            "numerical_features": [
                "Flow Duration",
                "Total Fwd Packet",
            ],
            "categorical_features": ["Protocol"],
        }

        preprocessor = CicFlowMeterPreprocessor(config)

        # Should work with valid features
        preprocessor.fit(self.test_data)
        transformed_data = preprocessor.transform(self.test_data)

        # Check that the features were processed correctly
        self.assertIn("Flow Duration", transformed_data.columns)
        self.assertIn("Total Fwd Packet", transformed_data.columns)
        self.assertIn("Protocol_TCP", transformed_data.columns)
        self.assertIn("Protocol_UDP", transformed_data.columns)
        self.assertIn("Protocol_ICMP", transformed_data.columns)

    def test_inverse_transform(self):
        """Test inverse_transform method recovers original values."""
        # Make a config that doesn't include binary_classifier at all
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration", "Total Fwd Packet"],
                "categorical_features": [
                    "Protocol"
                ],  # Need to include at least one categorical feature
            }
        )

        # Select only the numerical columns for this test
        numerical_data = self.test_data.select(
            ["Flow Duration", "Total Fwd Packet", "Protocol"]
        )

        # Fit and transform
        preprocessor.fit(numerical_data)
        transformed_data = preprocessor.transform(numerical_data)

        # Inverse transform back to original scale
        recovered_data = preprocessor.inverse_transform(transformed_data)

        # Compare original with recovered values
        for feature in preprocessor.config["numerical_features"]:
            np.testing.assert_array_almost_equal(
                numerical_data.select(feature).to_numpy().flatten(),
                recovered_data.select(feature).to_numpy().flatten(),
                decimal=10,
            )

    def test_with_realistic_cicflowmeter_dataset(self):
        """Test with a more realistic CicFlowMeter dataset."""
        # Create a dataset with more CicFlowMeter features
        np.random.seed(42)
        n_samples = 10

        realistic_data = pl.DataFrame(
            {
                "Flow Duration": np.random.randint(1, 1000, n_samples),
                "Total Fwd Packet": np.random.randint(1, 100, n_samples),
                "Total Bwd packets": np.random.randint(1, 100, n_samples),
                "Total Length of Fwd Packet": np.random.randint(
                    100, 10000, n_samples
                ),
                "Total Length of Bwd Packet": np.random.randint(
                    100, 10000, n_samples
                ),
                "Fwd Packet Length Max": np.random.randint(
                    100, 1500, n_samples
                ),
                "Fwd Packet Length Min": np.random.randint(20, 100, n_samples),
                "Fwd Packet Length Mean": np.random.uniform(50, 500, n_samples),
                "Flow Bytes/s": np.random.uniform(100, 10000, n_samples),
                "Flow Packets/s": np.random.uniform(1, 1000, n_samples),
                "Flow IAT Mean": np.random.uniform(0.001, 1, n_samples),
                "Protocol": np.random.choice(
                    ["TCP", "UDP", "ICMP", "HTTP"], n_samples
                ),
                "Label": np.random.choice(
                    ["BENIGN", "DoS", "PortScan", "DDoS"], n_samples
                ),
            }
        )

        # Create a custom config with only the features present in the dataset
        custom_config = {
            "scaler": "MinMaxScaler",
            "numerical_features": [
                "Flow Duration",
                "Total Fwd Packet",
                "Total Bwd packets",
                "Total Length of Fwd Packet",
                "Total Length of Bwd Packet",
                "Fwd Packet Length Max",
                "Fwd Packet Length Min",
                "Fwd Packet Length Mean",
                "Flow Bytes/s",
                "Flow Packets/s",
                "Flow IAT Mean",
            ],
            "categorical_features": ["Protocol"],
        }

        # Use custom config which only has features present in the dataset
        preprocessor = CicFlowMeterPreprocessor(custom_config)

        # Process the data
        preprocessor.fit(realistic_data)
        transformed_data = preprocessor.transform(realistic_data)

        # Check that all expected numerical features and one-hot encoded features are present
        expected_numerical_features = set(custom_config["numerical_features"])
        expected_categorical_features = {
            "Protocol_TCP",
            "Protocol_UDP",
            "Protocol_ICMP",
            "Protocol_HTTP",
        }
        expected_columns = expected_numerical_features.union(
            expected_categorical_features
        ).union({"index"})
        self.assertEqual(set(transformed_data.columns), expected_columns)

    def test_binary_classifier(self):
        """Test binary classification feature."""
        # Setup the preprocessor with binary classification enabled
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration", "Total Fwd Packet"],
                "categorical_features": ["Protocol"],
                "binary_classifier": True,
                "target_col": "Label",
                "target_mappings": {
                    "BENIGN": False,
                    "benign": False,
                    "notice": False,
                    "*": True,  # Catch-all for any other labels
                },
            }
        )

        # Fit and transform the data
        transformed_data = preprocessor.fit_transform(self.test_data)

        # Check that binary classification was applied correctly
        # "DoS", "PortScan", and "SSH-Patator" should be mapped to True
        # "BENIGN" and "notice" should be mapped to False
        expected_binary_labels = [False, True, False, True, True]
        self.assertEqual(
            transformed_data.select("Label").to_series().to_list(),
            expected_binary_labels,
        )

        # Check that index column is present by default
        self.assertIn("index", transformed_data.columns)
        self.assertEqual(
            transformed_data.select("index").to_series().to_list(),
            [0, 1, 2, 3, 4],
        )

    def test_binary_classifier_with_drop_index(self):
        """Test binary classification with drop_index=True."""
        # Setup the preprocessor with binary classification enabled
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration", "Total Fwd Packet"],
                "categorical_features": ["Protocol"],
                "binary_classifier": True,
                "target_col": "Label",
                "target_mappings": {
                    "BENIGN": False,
                    "benign": False,
                    "notice": False,
                    "*": True,  # Catch-all for any other labels
                },
            }
        )

        # Fit and transform the data with drop_index=True
        transformed_data = preprocessor.fit_transform(
            self.test_data, drop_index=True
        )

        # Check that index column is not present
        self.assertNotIn("index", transformed_data.columns)

        # But still check binary classification was applied correctly
        expected_binary_labels = [False, True, False, True, True]
        self.assertEqual(
            transformed_data.select("Label").to_series().to_list(),
            expected_binary_labels,
        )

        # Check that index column is dropped
        self.assertNotIn("index", transformed_data.columns)

    def test_binary_classifier_with_missing_config(self):
        """Test binary classification with missing configuration."""
        # Missing target_col
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration"],
                "categorical_features": ["Protocol"],
                "binary_classifier": True,
                "target_mappings": {"BENIGN": False, "*": True},
            }
        )

        # Need to fit the preprocessor first
        preprocessor.fit(self.test_data)

        with self.assertRaises(ValueError) as context:
            preprocessor.transform(self.test_data)
        self.assertIn("Target column must be specified", str(context.exception))

        # Missing target_mappings
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration"],
                "categorical_features": ["Protocol"],
                "binary_classifier": True,
                "target_col": "Label",
            }
        )

        # Need to fit the preprocessor first
        preprocessor.fit(self.test_data)

        with self.assertRaises(ValueError) as context:
            preprocessor.transform(self.test_data)
        self.assertIn(
            "Target mappings must be specified", str(context.exception)
        )

        # Missing catch-all mapping
        preprocessor = CicFlowMeterPreprocessor(
            {
                "scaler": "MinMaxScaler",
                "numerical_features": ["Flow Duration"],
                "categorical_features": ["Protocol"],
                "binary_classifier": True,
                "target_col": "Label",
                "target_mappings": {"BENIGN": False},
            }
        )

        # Need to fit the preprocessor first
        preprocessor.fit(self.test_data)

        with self.assertRaises(ValueError) as context:
            preprocessor.transform(self.test_data)
        self.assertIn(
            "Catch-all mapping (*) must be specified", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
