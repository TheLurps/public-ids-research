#!/usr/bin/env python
# coding: utf-8

"""
CNN-BiLSTM Binary Classifier for Network Intrusion Detection

This script is used to test a pre-trained CNN-BiLSTM model for binary classification
on CICFlowMeter datasets. It loads a pre-trained model from MLflow, evaluates it on
test data, and logs the results back to MLflow.

The experiment workflow includes:
1. Loads test data from a parquet file
2. Retrieves a preprocessor from a previous MLflow run and transforms the data
3. Loads a pre-trained model from MLflow
4. Evaluates the model and calculates performance metrics
5. Generates and saves performance visualizations (ROC curve, confusion matrix)
6. Logs all results and metrics to MLflow

Usage example:

```
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --mlflow-run-id "0f4dbe66d69b42bf96477079f2c71d4d" \
    -vv
```
"""

import os
import argparse
import logging
import mlflow
import pickle
import tensorflow as tf
import numpy as np
import polars as pl


from train_cnn_bilstm_cicflowmeter_binary_classifier import (
    setup_logging,
    load_data,
    evaluate_model,
    export_predictions,
    plot_roc_curve,
    plot_confusion_matrix,
)

logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.Namespace:
    """
    Set up and configure the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Testing CNN-BiLSTM on CICFlowMeter datasets as binary classifier"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data file (CICFlowMeter parquet file)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="CNN-BiLSTM-CICFlowMeter",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'CICIDS2017')",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="CNN-BiLSTM",
        help="Type of model to test (default: 'CNN-BiLSTM')",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "--mlflow-run-id",
        type=str,
        required=True,
        help="MLflow run ID for loading model",
    )
    parser.add_argument(
        "--mlflow-model-name",
        type=str,
        default="model",
        help="MLflow model name",
    )

    return parser.parse_args()


def preprocess_data(
    data: pl.DataFrame,
    target_col: str,
    mlflow_run_id: str,
    mlflow_preprocessor_name: str = "preprocessor/preprocessor.pkl",
) -> tuple[
    list[str],  # feature_cols
    tuple[int, int],  # input_shape
    np.ndarray,  # X_test
    np.ndarray,  # y_test
    np.ndarray,  # test_index
]:
    """
    Preprocess the data using a saved preprocessor from a previous MLflow run.

    Args:
        data: Input data in Polars DataFrame format
        target_col: Name of the target column in the dataset
        mlflow_run_id: ID of the MLflow run containing the saved preprocessor
        mlflow_preprocessor_name: Path to the preprocessor artifact in MLflow

    Returns:
        A tuple containing:
        - feature_cols: List of feature column names
        - input_shape: Shape of the input for the model (n_features, 1)
        - X_test: Preprocessed feature data as numpy array
        - y_test: Target values as numpy array
        - test_index: Index of the test data
    """
    preprocessor_uri = f"runs:/{mlflow_run_id}/{mlflow_preprocessor_name}"
    preprocess_path = mlflow.artifacts.download_artifacts(
        artifact_uri=preprocessor_uri,
    )
    logger.info(
        f"Restore preprocessor from run {mlflow_run_id} to {preprocess_path}"
    )
    with open(preprocess_path, "rb") as f:
        preprocessor = pickle.load(f)
        transformed_data = preprocessor.transform(data)

    logger.info("Transformed data shape: %s", transformed_data.shape)
    mlflow.log_param("transformed_data_shape", transformed_data.shape)

    feature_cols = [
        col
        for col in transformed_data.columns
        if col not in ["index", target_col]
    ]
    mlflow.log_param("feature_cols", feature_cols)
    input_shape = (transformed_data[feature_cols].shape[1], 1)
    mlflow.log_param("input_shape", input_shape)

    X_test = transformed_data[["index"] + feature_cols].to_numpy()
    y_test = transformed_data[target_col].to_numpy()

    # remove index added by preprocessor
    test_index, X_test = X_test[:, 0], X_test[:, 1:]

    X_test_stats = {
        "min": X_test.min(),
        "max": X_test.max(),
        "mean": X_test.mean(),
        "std": X_test.std(),
    }
    logger.debug(f"{X_test_stats=}")
    mlflow.log_param("X_test_stats", X_test_stats)

    y_test_distribution = np.unique(y_test, return_counts=True)
    logger.info(f"Test label distribution: {y_test_distribution}")
    mlflow.log_param(
        "test_label_distribution",
        pl.DataFrame(
            y_test_distribution,
            schema=[("Label", pl.UInt32), ("count", pl.UInt32)],
        ).to_dict(as_series=False),
    )

    return (
        feature_cols,
        input_shape,
        X_test,
        y_test,
        test_index,
    )


def main() -> None:
    args = setup_argparser()
    setup_logging(args.verbose)

    # Log system metrics
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    # Suppress TF logging except errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Prevent multiple registrations of CUDA components
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.autolog(log_datasets=False)
    logger.info("Using GPU: %s", tf.config.list_physical_devices("GPU"))

    logger.info(f"Starting experiment: {args.experiment_name}")
    with mlflow.start_run():
        mlflow.set_tag("operation", "test")
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("model_name", args.model_name)

        logger.info("Loading data...")
        data, target_col = load_data(
            data_path=args.data_path, dataset_name=args.dataset_name
        )

        logger.info("Preprocessing data...")
        (
            feature_cols,
            input_shape,
            X_test,
            y_test,
            test_index,
        ) = preprocess_data(
            data=data,
            target_col=target_col,
            mlflow_run_id=args.mlflow_run_id,
        )

        logger.info("Loading model...")
        mlflow.set_tag("model_mlflow_run_id", args.mlflow_run_id)
        model_uri = f"runs:/{args.mlflow_run_id}/{args.mlflow_model_name}"
        mlflow.log_param("model_uri", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("Evaluating model...")
        pred, pred_proba, fpr, tpr, roc_auc = evaluate_model(
            target_col=target_col,
            feature_cols=feature_cols,
            X_test=X_test,
            y_test=y_test,
            model=model,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
        )

        logger.info("Combining predictions with test data...")
        export_predictions(test_index, y_test, pred, pred_proba, data)

        logger.info("Plotting ROC curve...")
        plot_roc_curve(fpr, tpr, roc_auc)

        logger.info("Plotting confusion matrix...")
        plot_confusion_matrix(y_test, pred)

    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
