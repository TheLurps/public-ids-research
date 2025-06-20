from ids_research.preprocessors.base import BasePreprocessor
from typing import Dict
import polars as pl
import logging

logger = logging.getLogger(__name__)


class CicFlowMeterPreprocessor(BasePreprocessor):
    """
    Preprocessor for CicFlowMeter data.
    """

    def __init__(
        self,
        config: Dict = {
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
                "Fwd Packet Length Std",
                "Bwd Packet Length Max",
                "Bwd Packet Length Min",
                "Bwd Packet Length Mean",
                "Bwd Packet Length Std",
                "Flow Bytes/s",
                "Flow Packets/s",
                "Flow IAT Mean",
                "Flow IAT Std",
                "Flow IAT Max",
                "Flow IAT Min",
                "Fwd IAT Total",
                "Fwd IAT Mean",
                "Fwd IAT Std",
                "Fwd IAT Max",
                "Fwd IAT Min",
                "Bwd IAT Total",
                "Bwd IAT Mean",
                "Bwd IAT Std",
                "Bwd IAT Max",
                "Bwd IAT Min",
                "Fwd PSH Flags",
                "Bwd PSH Flags",
                "Fwd URG Flags",
                "Bwd URG Flags",
                "Fwd Header Length",
                "Bwd Header Length",
                "Fwd Packets/s",
                "Bwd Packets/s",
                "Packet Length Min",
                "Packet Length Max",
                "Packet Length Mean",
                "Packet Length Std",
                "Packet Length Variance",
                "FIN Flag Count",
                "SYN Flag Count",
                "RST Flag Count",
                "PSH Flag Count",
                "ACK Flag Count",
                "URG Flag Count",
                "CWR Flag Count",
                "ECE Flag Count",
                "Down/Up Ratio",
                "Average Packet Size",
                "Fwd Segment Size Avg",
                "Bwd Segment Size Avg",
                "Fwd Bytes/Bulk Avg",
                "Fwd Packet/Bulk Avg",
                "Fwd Bulk Rate Avg",
                "Bwd Bytes/Bulk Avg",
                "Bwd Packet/Bulk Avg",
                "Bwd Bulk Rate Avg",
                "Subflow Fwd Packets",
                "Subflow Fwd Bytes",
                "Subflow Bwd Packets",
                "Subflow Bwd Bytes",
                "FWD Init Win Bytes",
                "Bwd Init Win Bytes",
                "Fwd Act Data Pkts",
                "Fwd Seg Size Min",
                "Active Mean",
                "Active Std",
                "Active Max",
                "Active Min",
                "Idle Mean",
                "Idle Std",
                "Idle Max",
                "Idle Min",
            ],
            "categorical_features": ["Protocol"],
            "target_col": "Label",
            "binary_classifier": True,
            "target_mappings": {
                "BENIGN": False,
                "benign": False,
                "*": True,  # This is a catch-all for any other labels
            },
        },
    ):
        super().__init__(config)
        self.config = config

    def fit(self, data: pl.DataFrame):
        super().fit(data)

    def transform(
        self, data: pl.DataFrame, drop_index: bool = False
    ) -> pl.DataFrame:
        if "binary_classifier" in self.config:
            transformed_data = super().transform(data, drop_index=False)

            if "target_col" not in self.config:
                raise ValueError(
                    "Target column must be specified in the config."
                )
            if "target_mappings" not in self.config:
                raise ValueError(
                    "Target mappings must be specified in the config."
                )
            if "*" not in self.config["target_mappings"]:
                raise ValueError(
                    "Catch-all mapping (*) must be specified in target mappings."
                )

            if self.config["binary_classifier"]:
                target_col = self.config["target_col"]
                labels = data.select(target_col)

                # Start with a base expression
                mapping_expr = pl.lit(self.config["target_mappings"]["*"])

                # Then override with specific mappings
                for k, v in self.config["target_mappings"].items():
                    mapping_expr = (
                        pl.when(pl.col(target_col) == k)
                        .then(v)
                        .otherwise(mapping_expr)
                    )

                # Apply the mapping expression once
                labels = labels.with_row_index().with_columns(
                    pl.col("index"),
                    mapping_expr.cast(pl.Boolean).alias(target_col),
                )

                transformed_data = transformed_data.join(
                    labels, on="index", how="left"
                )

                # Drop index if requested
                if drop_index:
                    transformed_data = transformed_data.drop("index")
        else:
            transformed_data = super().transform(data, drop_index=drop_index)

        return transformed_data

    def fit_transform(
        self, data: pl.DataFrame, drop_index: bool = False
    ) -> pl.DataFrame:
        """
        Fit the preprocessor to the data and transform it.

        Args:
            data: The data to fit and transform
            drop_index: Whether to drop the index column in the output
        """
        logger.debug("Fitting and transforming data.")

        # First, fit the preprocessor
        self.fit(data)

        # Then transform the data with the drop_index parameter
        return self.transform(data, drop_index=drop_index)

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return super().inverse_transform(data)
