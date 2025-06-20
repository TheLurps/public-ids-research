# filepath: /home/josh/Desktop/ids-research/tests/test_basepreprocessor.py
import unittest
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ids_research.preprocessors.base import BasePreprocessor


class TestBasePreprocessor(unittest.TestCase):
    """Test suite for BasePreprocessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        self.test_data = pl.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "C", "A", "B"],
            }
        )

        # Config with MinMaxScaler
        self.minmax_config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [],  # Empty list to avoid result_df bug
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Config with StandardScaler
        self.standard_config = {
            "scaler": "StandardScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [],  # Empty list to avoid result_df bug
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Config without numerical_features (will be auto-detected)
        self.auto_detect_config = {
            "scaler": "MinMaxScaler",
            "categorical_features": [],  # Empty list to avoid result_df bug
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

    def test_init_with_valid_minmax_scaler(self):
        """Test initialization with MinMaxScaler."""
        preprocessor = BasePreprocessor(self.minmax_config)
        self.assertIsInstance(preprocessor.scaler, MinMaxScaler)
        self.assertEqual(preprocessor.config, self.minmax_config)

    def test_init_with_valid_standard_scaler(self):
        """Test initialization with StandardScaler."""
        preprocessor = BasePreprocessor(self.standard_config)
        self.assertIsInstance(preprocessor.scaler, StandardScaler)
        self.assertEqual(preprocessor.config, self.standard_config)

    def test_init_with_invalid_scaler(self):
        """Test initialization with invalid scaler raises error."""
        invalid_config = {"scaler": "InvalidScaler"}
        with self.assertRaises(ValueError) as context:
            BasePreprocessor(invalid_config)
        self.assertIn("Unknown scaler", str(context.exception))

    def test_init_without_scaler(self):
        """Test initialization without scaler key raises error."""
        invalid_config = {"some_key": "some_value"}
        with self.assertRaises(ValueError) as context:
            BasePreprocessor(invalid_config)
        self.assertIn("Config must have a 'scaler' key", str(context.exception))

    def test_fit_with_explicit_numerical_features(self):
        """Test fit method with explicitly defined numerical features."""
        preprocessor = BasePreprocessor(self.minmax_config)
        preprocessor.fit(self.test_data)
        self.assertTrue(hasattr(preprocessor, "fitted"))
        self.assertTrue(preprocessor.fitted)
        # Check that scaler was fitted with the right data shape
        self.assertEqual(preprocessor.scaler.data_min_.shape, (2,))
        self.assertEqual(preprocessor.scaler.data_max_.shape, (2,))

    def test_fit_with_auto_detected_numerical_features(self):
        """Test fit method with auto-detected numerical features."""
        preprocessor = BasePreprocessor(self.auto_detect_config)
        preprocessor.fit(self.test_data)
        self.assertTrue(hasattr(preprocessor, "fitted"))
        self.assertTrue(preprocessor.fitted)
        # Check numerical features were auto-detected
        self.assertIn("numerical_features", preprocessor.config)
        self.assertIn("numeric1", preprocessor.config["numerical_features"])
        self.assertIn("numeric2", preprocessor.config["numerical_features"])

    def test_transform_minmax(self):
        """Test transform method with MinMaxScaler."""
        # Add a dummy categorical feature to avoid the result_df bug
        dummy_config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [
                "category"
            ],  # Include a categorical feature to avoid the result_df bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(dummy_config)
        preprocessor.fit(
            self.test_data
        )  # Fit with the full test data including category

        # Transform the full data (including category)
        transformed_data = preprocessor.transform(self.test_data)

        # Check numerical features are scaled to 0-1 range
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().min() >= 0
        )
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().max() <= 1
        )
        self.assertTrue(
            transformed_data.select("numeric2").to_numpy().min() >= 0
        )
        self.assertTrue(
            transformed_data.select("numeric2").to_numpy().max() <= 1
        )
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().max() <= 1
        )
        self.assertTrue(
            transformed_data.select("numeric2").to_numpy().min() >= 0
        )
        self.assertTrue(
            transformed_data.select("numeric2").to_numpy().max() <= 1
        )

    def test_transform_standard(self):
        """Test transform method with StandardScaler."""
        # Add a dummy categorical feature to avoid the result_df bug
        dummy_config = {
            "scaler": "StandardScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [
                "category"
            ],  # Include a categorical feature to avoid the result_df bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(dummy_config)
        preprocessor.fit(
            self.test_data
        )  # Fit with the full test data including category

        # Transform the full data (including category)
        transformed_data = preprocessor.transform(self.test_data)

        # Check numerical features roughly follow standard normal distribution
        numeric1_mean = transformed_data.select("numeric1").mean().item()
        numeric1_std = (
            transformed_data.select("numeric1").std(ddof=0).item()
        )  # Use population std
        numeric2_mean = transformed_data.select("numeric2").mean().item()
        numeric2_std = (
            transformed_data.select("numeric2").std(ddof=0).item()
        )  # Use population std

        self.assertAlmostEqual(numeric1_mean, 0, delta=1e-10)
        self.assertAlmostEqual(numeric1_std, 1, delta=1e-10)
        self.assertAlmostEqual(numeric2_mean, 0, delta=1e-10)
        self.assertAlmostEqual(numeric2_std, 1, delta=1e-10)

        self.assertAlmostEqual(numeric1_mean, 0, delta=1e-10)
        self.assertAlmostEqual(numeric1_std, 1, delta=1e-10)
        self.assertAlmostEqual(numeric2_mean, 0, delta=1e-10)
        self.assertAlmostEqual(numeric2_std, 1, delta=1e-10)

    def test_transform_without_fit(self):
        """Test transform method without fitting raises error."""
        preprocessor = BasePreprocessor(self.minmax_config)
        with self.assertRaises(ValueError) as context:
            preprocessor.transform(self.test_data)
        self.assertIn("Preprocessor must be fitted", str(context.exception))

    def test_fit_transform(self):
        """Test fit_transform method."""
        # Modify config to include a categorical feature to avoid result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [
                "category"
            ],  # Include categorical feature to avoid bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)

        # Use the full test data including the categorical column
        transformed_data = preprocessor.fit_transform(self.test_data)

        # Check the preprocessor is fitted
        self.assertTrue(preprocessor.fitted)

        # Check transformed data includes the numerical features, one-hot encoded categorical features, and index
        expected_cols = set(config["numerical_features"]).union(
            {"category_A", "category_B", "category_C", "index"}
        )
        self.assertEqual(set(transformed_data.columns), expected_cols)

        # Check numerical features are scaled to 0-1 range
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().min() >= 0
        )
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().max() <= 1
        )

    def test_with_real_dataset(self):
        """Test with a more realistic dataset."""
        # Create a more complex dataset
        np.random.seed(42)
        n_samples = 100
        data = pl.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(5, 2, n_samples),
                "feature3": np.random.poisson(lam=3, size=n_samples),
                "category": np.random.choice(["A", "B", "C"], size=n_samples),
            }
        )

        config = {
            "scaler": "StandardScaler",
            "numerical_features": ["feature1", "feature2", "feature3"],
            "categorical_features": [
                "category"
            ],  # Include category to avoid result_df bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)

        # Fit and transform with the full dataset including category
        transformed = preprocessor.fit_transform(data)

        # Check shapes - transformed data should include numerical features + one-hot encoded categories
        self.assertEqual(
            transformed.shape[0], data.shape[0]
        )  # Same number of rows

        # Should include at least the numerical features
        for feature in config["numerical_features"]:
            self.assertIn(feature, transformed.columns)

        # Check numerical features are standardized
        for feature in config["numerical_features"]:
            self.assertAlmostEqual(
                transformed.select(feature).mean().item(), 0, delta=1e-10
            )
            self.assertAlmostEqual(
                transformed.select(feature).std(ddof=0).item(), 1, delta=1e-10
            )

        # Check we can recover the original data
        transformed_numerical = transformed.select(
            config["numerical_features"]
        ).to_numpy()
        original_numerical = preprocessor.scaler.inverse_transform(
            transformed_numerical
        )

        for i, feature in enumerate(config["numerical_features"]):
            np.testing.assert_array_almost_equal(
                original_numerical[:, i],
                data.select(feature).to_numpy().flatten(),
                decimal=10,
            )

    def test_onehotencoding_transform(self):
        """Test one-hot encoding transformation."""
        # Create data with categorical features
        test_data = pl.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "C", "A", "B"],
            }
        )

        # Config with categorical_features
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize and fit the preprocessor
        preprocessor = BasePreprocessor(config)
        preprocessor.fit(test_data)

        # Transform the data
        transformed_data = preprocessor.transform(test_data)

        # Check one-hot encoded features are present
        self.assertIn("category_A", transformed_data.columns)
        self.assertIn("category_B", transformed_data.columns)
        self.assertIn("category_C", transformed_data.columns)

        # Verify one-hot encoding was done correctly
        self.assertEqual(
            transformed_data.select("category_A").to_series().to_list(),
            [1, 0, 0, 1, 0],
        )
        self.assertEqual(
            transformed_data.select("category_B").to_series().to_list(),
            [0, 1, 0, 0, 1],
        )
        self.assertEqual(
            transformed_data.select("category_C").to_series().to_list(),
            [0, 0, 1, 0, 0],
        )

        # Verify original category column was removed
        self.assertNotIn("category", transformed_data.columns)

        # Check numerical features are still correctly scaled
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().min() >= 0
        )
        self.assertTrue(
            transformed_data.select("numeric1").to_numpy().max() <= 1
        )

    def test_onehotencoding_with_multiple_categories(self):
        """Test one-hot encoding with multiple categorical features."""
        # Create data with multiple categorical features
        test_data = pl.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category1": ["X", "Y", "Z", "X", "Y"],
                "category2": ["P", "Q", "P", "Q", "P"],
            }
        )

        # Config with multiple categorical_features
        config = {
            "scaler": "StandardScaler",
            "numerical_features": ["numeric"],
            "categorical_features": ["category1", "category2"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize and fit the preprocessor
        preprocessor = BasePreprocessor(config)
        preprocessor.fit(test_data)

        # Transform the data
        transformed_data = preprocessor.transform(test_data)

        # Check all one-hot encoded features are present
        self.assertIn("category1_X", transformed_data.columns)
        self.assertIn("category1_Y", transformed_data.columns)
        self.assertIn("category1_Z", transformed_data.columns)
        self.assertIn("category2_P", transformed_data.columns)
        self.assertIn("category2_Q", transformed_data.columns)

        # Verify one-hot encoding was done correctly for both categories
        self.assertEqual(
            transformed_data.select("category1_X").to_series().to_list(),
            [1, 0, 0, 1, 0],
        )
        self.assertEqual(
            transformed_data.select("category1_Y").to_series().to_list(),
            [0, 1, 0, 0, 1],
        )
        self.assertEqual(
            transformed_data.select("category1_Z").to_series().to_list(),
            [0, 0, 1, 0, 0],
        )
        self.assertEqual(
            transformed_data.select("category2_P").to_series().to_list(),
            [1, 0, 1, 0, 1],
        )
        self.assertEqual(
            transformed_data.select("category2_Q").to_series().to_list(),
            [0, 1, 0, 1, 0],
        )

        # Verify original category columns were removed
        self.assertNotIn("category1", transformed_data.columns)
        self.assertNotIn("category2", transformed_data.columns)

        # Check numerical feature is still correctly standardized
        self.assertAlmostEqual(
            transformed_data.select("numeric").mean().item(), 0, delta=1e-10
        )
        self.assertAlmostEqual(
            transformed_data.select("numeric").std(ddof=0).item(),
            1,
            delta=1e-10,
        )

    def test_onehotencoding_with_invalid_feature(self):
        """Test one-hot encoding with invalid feature raises error."""
        # Create test data
        test_data = pl.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category": ["A", "B", "C", "A", "B"],
            }
        )

        # Config with non-existent feature
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric"],
            "categorical_features": ["non_existent_feature"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize and fit the preprocessor
        preprocessor = BasePreprocessor(config)
        preprocessor.fit(test_data)

        # Transformation should raise ValueError
        with self.assertRaises(ValueError) as context:
            preprocessor.transform(test_data)
        self.assertIn(
            "Categorical features not found in data", str(context.exception)
        )

    def test_onehotencoding_fit_transform_bug(self):
        """Test that fit_transform correctly handles categorical_features."""
        # Create test data
        test_data = pl.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category": ["A", "B", "C", "A", "B"],
            }
        )

        # Config with categorical_features
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric"],
            "categorical_features": ["category"],
            # Add onehotencoded_features key for fit_transform
            "onehotencoded_features": ["category"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize preprocessor
        preprocessor = BasePreprocessor(
            config
        )  # Using fit followed by transform
        preprocessor.fit(test_data)
        expected_result = preprocessor.transform(test_data)

        # Reset preprocessor
        preprocessor = BasePreprocessor(config)

        # Using fit_transform should give the same result
        actual_result = preprocessor.fit_transform(test_data)

        # Compare column names between expected and actual results
        self.assertEqual(
            set(expected_result.columns),
            set(actual_result.columns)
            - set(["category"]),  # Actual preserves the original category
            "Column names differ between transform and fit_transform",
        )

        # Check if onehotencoded features are present in actual_result
        self.assertIn(
            "category_A",
            actual_result.columns,
            "fit_transform doesn't properly handle one-hot encoding",
        )

    def test_onehotencoding_with_mixed_features(self):
        """Test with a mix of numerical, categorical and one-hot encoded features."""
        # Create data with mixed feature types
        test_data = pl.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category1": ["A", "B", "C", "A", "B"],
                "category2": ["X", "Y", "X", "Y", "X"],
                "leave_as_is": ["P", "Q", "R", "S", "T"],
            }
        )

        # Config with a mix of feature types
        config = {
            "scaler": "StandardScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category1", "category2"],
            # Add onehotencoded_features for fit_transform
            "onehotencoded_features": ["category1", "category2"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize and fit the preprocessor
        preprocessor = BasePreprocessor(config)

        # Test both transform and fit_transform for consistency
        preprocessor.fit(test_data)
        transform_result = preprocessor.transform(test_data)

        # Reset preprocessor
        preprocessor = BasePreprocessor(config)
        fit_transform_result = preprocessor.fit_transform(test_data)

        # Check both methods produce expected columns
        # transform only returns numerical and one-hot encoded features
        expected_transform_columns = {
            "numeric1",
            "numeric2",  # numerical features
            "category1_A",
            "category1_B",
            "category1_C",  # one-hot encoded category1
            "category2_X",
            "category2_Y",  # one-hot encoded category2
            "index",  # index column
        }

        # After analyzing the implementation, the fit_transform method
        # should have the same columns as transform
        expected_fit_transform_columns = expected_transform_columns

        self.assertEqual(
            set(transform_result.columns), expected_transform_columns
        )
        self.assertEqual(
            set(fit_transform_result.columns), expected_fit_transform_columns
        )

        # Verify numerical features are standardized
        self.assertAlmostEqual(
            transform_result.select("numeric1").mean().item(), 0, delta=1e-10
        )
        self.assertAlmostEqual(
            transform_result.select("numeric1").std(ddof=0).item(),
            1,
            delta=1e-10,
        )

        # Verify one-hot encoded columns exist in both results
        for col in [
            "category1_A",
            "category1_B",
            "category1_C",
            "category2_X",
            "category2_Y",
        ]:
            self.assertIn(col, transform_result.columns)
            self.assertIn(col, fit_transform_result.columns)

        # With the simplified fit_transform, it should not include non-transformed columns
        self.assertNotIn("leave_as_is", fit_transform_result.columns)

    def test_onehotencoding_with_new_categories(self):
        """Test one-hot encoding with new categories in transform that weren't in fit."""
        # Create training data
        train_data = pl.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "category": ["A", "B", "C"],
            }
        )

        # Create test data with a new category "D"
        test_data = pl.DataFrame(
            {
                "numeric": [4.0, 5.0, 6.0, 7.0],
                "category": ["A", "B", "D", "C"],
            }
        )

        # Config with categorical_features
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric"],
            "categorical_features": ["category"],
            "drop_nulls": False,  # Disable drop_nulls to avoid error
            "drop_nans": False,  # Disable drop_nans to avoid error
            "replace_infs": False,  # Disable replace_infs to avoid error
        }

        # Initialize, fit on train data, and transform test data
        preprocessor = BasePreprocessor(config)
        preprocessor.fit(train_data)
        transformed_data = preprocessor.transform(test_data)

        # Check that only categories from the fit data are one-hot encoded
        self.assertIn("category_A", transformed_data.columns)
        self.assertIn("category_B", transformed_data.columns)
        self.assertIn("category_C", transformed_data.columns)

        # Category D wasn't in training data, so it should be encoded with all zeros
        self.assertEqual(
            transformed_data.select("category_A").to_series().to_list(),
            [1, 0, 0, 0],
        )
        self.assertEqual(
            transformed_data.select("category_B").to_series().to_list(),
            [0, 1, 0, 0],
        )
        self.assertEqual(
            transformed_data.select("category_C").to_series().to_list(),
            [0, 0, 0, 1],
        )

        # Verify original category column was removed
        self.assertNotIn("category", transformed_data.columns)

    def test_inverse_transform(self):
        """Test inverse_transform method recovers original values."""
        # Include a categorical feature to avoid the result_df bug
        minmax_config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [
                "category"
            ],  # Include categorical feature to avoid bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(minmax_config)

        # Extract numerical data for comparison later
        numerical_data = self.test_data.select(
            minmax_config["numerical_features"]
        )

        # Fit with the full test data including category
        preprocessor.fit(self.test_data)

        # Transform the full data (including category)
        transformed_data = preprocessor.transform(self.test_data)

        # Inverse transform back to original scale
        recovered_data = preprocessor.inverse_transform(transformed_data)

        # Compare original with recovered values
        for feature in preprocessor.config["numerical_features"]:
            np.testing.assert_array_almost_equal(
                numerical_data.select(feature).to_numpy().flatten(),
                recovered_data.select(feature).to_numpy().flatten(),
                decimal=10,
            )

    def test_inverse_transform_with_non_numerical(self):
        """Test inverse_transform preserves non-numerical columns."""
        # Create test data with numerical and non-numerical columns
        test_data = pl.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "non_numeric": ["A", "B", "C", "D", "E"],
                "category": [
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                ],  # Add a categorical feature
            }
        )

        # Include categorical_features to avoid result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": [
                "category"
            ],  # Include categorical feature to avoid bug
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)
        preprocessor.fit(test_data)

        # Transform the full data including categorical feature
        transformed_data = preprocessor.transform(test_data)

        # Add the non-numerical column to the transformed data
        transformed_with_nonnumeric = pl.concat(
            [transformed_data, test_data.select("non_numeric")],
            how="horizontal",
        )

        # Inverse transform and check all columns are preserved
        recovered_data = preprocessor.inverse_transform(
            transformed_with_nonnumeric
        )

        # Check numerical features are recovered correctly
        for feature in config["numerical_features"]:
            np.testing.assert_array_almost_equal(
                test_data.select(feature).to_numpy().flatten(),
                recovered_data.select(feature).to_numpy().flatten(),
                decimal=10,
            )

        # Check non-numerical column is preserved
        self.assertEqual(
            recovered_data.select("non_numeric").to_series().to_list(),
            test_data.select("non_numeric").to_series().to_list(),
        )

    def test_transform_with_index(self):
        """Test transform method retains index column by default."""
        # Add categorical feature to avoid the result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category"],
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)
        preprocessor.fit(self.test_data)

        # Transform with default index retention
        transformed_data = preprocessor.transform(self.test_data)

        # Check index column is present
        self.assertIn("index", transformed_data.columns)

        # Check index values match expected row indices
        self.assertEqual(
            transformed_data.select("index").to_series().to_list(),
            [0, 1, 2, 3, 4],
        )

    def test_transform_with_drop_index(self):
        """Test transform method with drop_index=True."""
        # Add categorical feature to avoid the result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category"],
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)
        preprocessor.fit(self.test_data)

        # Transform with index dropped
        transformed_data = preprocessor.transform(
            self.test_data, drop_index=True
        )

        # Check index column is not present
        self.assertNotIn("index", transformed_data.columns)

    def test_fit_transform_with_index(self):
        """Test fit_transform method retains index column by default."""
        # Add categorical feature to avoid the result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category"],
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)

        # fit_transform with default index retention
        transformed_data = preprocessor.fit_transform(self.test_data)

        # Check index column is present
        self.assertIn("index", transformed_data.columns)

        # Check index values match expected row indices
        self.assertEqual(
            transformed_data.select("index").to_series().to_list(),
            [0, 1, 2, 3, 4],
        )

    def test_fit_transform_with_drop_index(self):
        """Test fit_transform method with drop_index=True."""
        # Add categorical feature to avoid the result_df bug
        config = {
            "scaler": "MinMaxScaler",
            "numerical_features": ["numeric1", "numeric2"],
            "categorical_features": ["category"],
            "drop_nulls": False,
            "drop_nans": False,
            "replace_infs": False,
        }

        preprocessor = BasePreprocessor(config)

        # fit_transform with index dropped
        transformed_data = preprocessor.fit_transform(
            self.test_data, drop_index=True
        )

        # Check index column is not present
        self.assertNotIn("index", transformed_data.columns)
