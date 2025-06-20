from typing import Dict
import logging
import importlib
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """
    Base class for all preprocessors.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.fitted = False

        if "scaler" not in self.config:
            raise ValueError("Config must have a 'scaler' key.")

        SCALER_IMPORTS = {
            "MinMaxScaler": "sklearn.preprocessing.MinMaxScaler",
            "StandardScaler": "sklearn.preprocessing.StandardScaler",
        }

        scaler_path = SCALER_IMPORTS.get(self.config["scaler"])
        if scaler_path is None:
            raise ValueError(f"Unknown scaler: {self.config['scaler']}")

        module_path, class_name = scaler_path.rsplit(".", 1)
        Scaler = getattr(importlib.import_module(module_path), class_name)

        if "drop_nans" not in self.config:
            self.config["drop_nans"] = True
        if "drop_nulls" not in self.config:
            self.config["drop_nulls"] = True
        if "replace_infs" not in self.config:
            self.config["replace_infs"] = True

        self.scaler = Scaler()

    @staticmethod
    def _clean_numerical_data(
        data: pl.DataFrame,
        drop_nans: bool = True,
        drop_nulls: bool = True,
        replace_infs: bool = True,
    ) -> pl.DataFrame:
        """
        Clean numerical data by replacing infinities with NaN and dropping NaN/null values.
        """
        if replace_infs:
            # Replace infinities with NaN
            clean_data = data.with_columns(
                [
                    pl.when(pl.col(col).is_infinite())
                    .then(np.nan)
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in data.columns
                ]
            )
        else:
            clean_data = data

        if drop_nans:
            # Drop rows with NaN values
            clean_data = clean_data.drop_nans()

            # Raise an error if all values are NaN
            if clean_data.is_empty():
                logger.warning("All values are NaN after cleaning.")
                raise ValueError("All values are NaN after cleaning.")

        if drop_nulls:
            # Drop rows with null values
            clean_data = clean_data.drop_nulls()

            # Raise an error if all values are null
            if clean_data.is_empty():
                logger.warning("All values are null after cleaning.")
                raise ValueError("All values are null after cleaning.")

        return clean_data

    def fit(self, data: pl.DataFrame):
        """
        Fit the preprocessor to the data.
        """
        if "numerical_features" not in self.config:
            # Auto-detect numerical features by checking dtype
            numeric_dtypes = [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.Int16,
                pl.Int8,
                pl.UInt64,
                pl.UInt32,
                pl.UInt16,
                pl.UInt8,
            ]
            self.config["numerical_features"] = [
                col
                for col in data.columns
                if data.schema[col] in numeric_dtypes
            ]

        logger.debug("Fitting preprocessor to data.")

        # Get numerical data and clean it
        numerical_data = data.select(pl.col(self.config["numerical_features"]))
        clean_data = self._clean_numerical_data(
            numerical_data,
            replace_infs=self.config["replace_infs"],
            drop_nans=self.config["drop_nans"],
            drop_nulls=self.config["drop_nulls"],
        )
        logger.debug("Cleaned data for fitting.")

        self.scaler.fit(clean_data.to_numpy())
        self.fitted = True

        return self

    def transform(
        self, data: pl.DataFrame, drop_index: bool = False
    ) -> pl.DataFrame:
        """
        Transform the data using the fitted preprocessor.
        Optimized for performance with large datasets.
        """
        if not self.fitted:
            raise ValueError(
                "Preprocessor must be fitted before transforming data."
            )

        # Validate all features exist in the data
        missing_numerical = [
            f
            for f in self.config["numerical_features"]
            if f not in data.columns
        ]
        if missing_numerical:
            raise ValueError(
                f"Numerical features not found in data: {missing_numerical}"
            )

        if "categorical_features" in self.config:
            missing_categorical = [
                f
                for f in self.config["categorical_features"]
                if f not in data.columns
            ]
            if missing_categorical:
                raise ValueError(
                    f"Categorical features not found in data: {missing_categorical}"
                )

        # Add row index only once for the entire transformation process
        data_with_index = data.with_row_index()

        # Process numerical features
        numerical_features = self.config["numerical_features"]
        if numerical_features:
            # Extract numerical data with index
            numerical_data = data_with_index.select(
                ["index"] + [pl.col(col) for col in numerical_features]
            )

            # Clean numerical data
            clean_numerical_data = self._clean_numerical_data(
                numerical_data,
                replace_infs=self.config["replace_infs"],
                drop_nans=self.config["drop_nans"],
                drop_nulls=self.config["drop_nulls"],
            )
            # Transform numerical data
            transformed_ndarray = self.scaler.transform(
                clean_numerical_data.drop("index").to_numpy()
            )

            # Create transformed DataFrame
            numerical_df = pl.concat(
                [
                    clean_numerical_data.select("index").cast(pl.UInt32),
                    pl.DataFrame(
                        {
                            col: transformed_ndarray[:, idx]
                            for idx, col in enumerate(numerical_features)
                        }
                    ),
                ],
                how="horizontal",
            )
        else:
            # If no numerical features, just use the index
            numerical_df = data_with_index.select("index")

        # Process categorical features in batch if possible
        if (
            "categorical_features" in self.config
            and self.config["categorical_features"]
        ):
            # Pre-process all categorical features at once
            cat_features = []
            for feature in self.config["categorical_features"]:
                one_hot = data_with_index.select(["index", feature]).to_dummies(
                    columns=[feature], separator="_"
                )
                # Convert all one-hot columns (except 'index') to boolean
                bool_cols = [col for col in one_hot.columns if col != "index"]
                one_hot = one_hot.with_columns(
                    [pl.col(col).cast(pl.Boolean) for col in bool_cols]
                )
                cat_features.append(one_hot)

            # Join all categorical features to result in a single operation
            categorical_df = cat_features[0]
            for df in cat_features[1:]:
                categorical_df = categorical_df.join(df, on="index", how="left")

            # Join with numerical features
            result_df = numerical_df.join(
                categorical_df, on="index", how="left"
            )

        if self.config["drop_nulls"]:
            result_df = result_df.drop_nulls()
        if self.config["drop_nans"]:
            result_df = result_df.drop_nans()

        if drop_index:
            return result_df.drop("index")
        else:
            return result_df

    def fit_transform(
        self, data: pl.DataFrame, drop_index: bool = False
    ) -> pl.DataFrame:
        """
        Fit the preprocessor to the data and transform it.
        """
        logger.debug("Fitting and transforming data.")

        # First, fit the preprocessor
        self.fit(data)

        # Then transform the data
        return self.transform(data, drop_index=drop_index)

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Inverse transform the data using the fitted preprocessor.
        """
        if not self.fitted:
            raise ValueError(
                "Preprocessor must be fitted before inverse transforming data."
            )

        logger.debug("Inverse transforming data.")

        # Check if all numerical features are in the data
        for feature in self.config["numerical_features"]:
            if feature not in data.columns:
                raise ValueError(
                    f"Feature {feature} not found in data columns for inverse transform."
                )

        # Extract numerical features for inverse transform
        numerical_data = data.select(
            pl.col(self.config["numerical_features"])
        ).to_numpy()

        # Apply inverse transform
        inverse_transformed_data = self.scaler.inverse_transform(numerical_data)
        logger.debug("Data inverse transformed.")

        # Create a new DataFrame with the inverse transformed numerical columns
        inverse_transformed_df = pl.DataFrame(
            {
                col: inverse_transformed_data[:, i]
                for i, col in enumerate(self.config["numerical_features"])
            }
        )

        # Get all non-numerical columns from the input data
        non_numerical_cols = [
            col
            for col in data.columns
            if col not in self.config["numerical_features"]
        ]

        # If there are non-numerical columns, include them in the output
        if non_numerical_cols:
            return pl.concat(
                [data.select(non_numerical_cols), inverse_transformed_df],
                how="horizontal",
            )

        return inverse_transformed_df
