#!/usr/bin/env python
# coding: utf-8

"""
CNN-BiLSTM Binary Classifier for Network Intrusion Detection

This experiment implements and evaluates a CNN-BiLSTM hybrid deep learning model
for binary classification of network traffic as either benign or anomalous.

The experiment workflow includes:
1. Loading and preprocessing CICFlowMeter feature data
2. Building a hybrid CNN-BiLSTM neural network architecture
3. Training the model with early stopping
4. Evaluating performance using accuracy, ROC curve, and confusion matrix
5. Logging all experiment artifacts and metrics using MLflow

The CNN-BiLSTM architecture combines convolutional layers for spatial feature extraction
with bidirectional LSTM layers for capturing temporal dependencies in network flow data.
This hybrid approach aims to improve detection accuracy for various network attacks
while minimizing false positives.

Usage example:

```
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --epochs 1 \
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
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from matplotlib import pyplot as plt

from ids_research.preprocessors import CicFlowMeterPreprocessor
from ids_research.models import CNN_BiLSTM
from ids_research.models.cnn_bilstm import (
    TerminateOnNaN,
    EarlyStoppingEpochLogger,
)

logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.Namespace:
    """
    Set up and configure the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Training CNN-BiLSTM on CICFlowMeter datasets as binary classifier"
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
        "--test-size", type=float, default=0.2, help="Test size"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2, help="Validation size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--clipnorm", type=float, default=1.0, help="Clipnorm for optimizer"
    )
    parser.add_argument(
        "--benign-labels",
        type=str,
        default="benign,BENIGN",
        help="Comma-separated list of benign labels",
    )

    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbosity: Integer indicating verbosity level (0-3)
                  0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG
    """
    LOGLEVEL = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    logging.basicConfig(level=LOGLEVEL.get(verbosity, logging.DEBUG))


def load_data(
    data_path: str, dataset_name: str
) -> tuple[
    pl.DataFrame,  # data
    str,  # target_col
]:
    """
    Load dataset from the specified path and log metadata.

    Args:
        data_path: Path to the data file
        dataset_name: Name of the dataset

    Returns:
        tuple: (DataFrame containing the loaded data, target column name)

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    mlflow.set_tag("dataset_name", dataset_name)

    logger.info("Reading data from %s", data_path)
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path_obj}")

    data = pl.read_parquet(data_path_obj.expanduser())
    logger.info("Data shape: %s", data.shape)
    mlflow.log_param("data_shape", data.shape)

    target_col = "Label"
    label_distribution = data[target_col].value_counts()
    mlflow.log_param(
        "label_distribution", label_distribution.to_dict(as_series=False)
    )
    logger.info(f"Label distribution: {label_distribution}")
    return data, target_col


def preprocess_data(
    data: pl.DataFrame,
    target_col: str,
    data_path: str,
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    benign_labels: list[str] = ["benign"],
    random_state: int = 42,
    store_preprocessor: bool = True,
) -> tuple[
    list[str],  # feature_cols
    tuple[int, int],  # input_shape
    np.ndarray,  # X_train
    np.ndarray,  # y_train
    np.ndarray,  # train_index
    np.ndarray,  # X_val
    np.ndarray,  # y_val
    np.ndarray,  # val_index
    np.ndarray,  # X_test
    np.ndarray,  # y_test
    np.ndarray,  # test_index
]:
    """
    Preprocess data using CicFlowMeterPreprocessor and split into train/val/test sets.

    Args:
        data: DataFrame containing the raw data
        target_col: Name of the target column
        data_path: Path to the data file
        dataset_name: Name of the dataset
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        benign_labels: Array of benign labels
        random_state: Random seed for reproducibility
        store_preprocessor: Whether to store the preprocessor in MLflow

    Returns:
        tuple containing:
        - feature_cols: List of feature column names
        - input_shape: Shape of the input for the model
        - X_train: Training features
        - y_train: Training labels
        - train_index: Indices of training samples
        - X_val: Validation features
        - y_val: Validation labels
        - val_index: Indices of validation samples
        - X_test: Test features
        - y_test: Test labels
        - test_index: Indices of test samples
    """
    preprocessor = CicFlowMeterPreprocessor()
    preprocessor.config["target_mappings"] = {
        benign_label: False for benign_label in benign_labels
    } | {"*": True}
    transformed_data = preprocessor.fit_transform(data)

    if store_preprocessor:
        preprocessor_path = Path("/tmp/preprocessor.pkl")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

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

    X = transformed_data[["index"] + feature_cols].to_numpy()
    y = transformed_data[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
        )
    else:
        X_val, y_val = np.zeros((1, X_train.shape[1])), np.zeros((1,))

    # remove index added by preprocessor
    train_index, X_train = X_train[:, 0], X_train[:, 1:]
    val_index, X_val = X_val[:, 0], X_val[:, 1:]
    test_index, X_test = X_test[:, 0], X_test[:, 1:]

    X_train_stats = {
        "min": X_train.min(),
        "max": X_train.max(),
        "mean": X_train.mean(),
        "std": X_train.std(),
    }
    X_val_stats = {
        "min": X_val.min(),
        "max": X_val.max(),
        "mean": X_val.mean(),
        "std": X_val.std(),
    }
    X_test_stats = {
        "min": X_test.min(),
        "max": X_test.max(),
        "mean": X_test.mean(),
        "std": X_test.std(),
    }
    logger.debug(f"{X_train_stats=}")
    logger.debug(f"{X_val_stats=}")
    logger.debug(f"{X_test_stats=}")
    mlflow.log_param("X_train_stats", X_train_stats)
    mlflow.log_param("X_val_stats", X_val_stats)
    mlflow.log_param("X_test_stats", X_test_stats)

    y_train_distribution = np.unique(y_train, return_counts=True)
    y_val_distribution = np.unique(y_val, return_counts=True)
    y_test_distribution = np.unique(y_test, return_counts=True)
    logger.info(f"Training label distribution: {y_train_distribution}")
    logger.info(f"Validation label distribution: {y_val_distribution}")
    logger.info(f"Test label distribution: {y_test_distribution}")
    mlflow.log_param(
        "train_label_distribution",
        pl.DataFrame(
            y_train_distribution,
            schema=[("Label", pl.UInt32), ("count", pl.UInt32)],
        ).to_dict(as_series=False),
    )
    mlflow.log_param(
        "val_label_distribution",
        pl.DataFrame(
            y_val_distribution,
            schema=[("Label", pl.UInt32), ("count", pl.UInt32)],
        ).to_dict(as_series=False),
    )
    mlflow.log_param(
        "test_label_distribution",
        pl.DataFrame(
            y_test_distribution,
            schema=[("Label", pl.UInt32), ("count", pl.UInt32)],
        ).to_dict(as_series=False),
    )

    train_dataset = mlflow.data.from_pandas(
        pd.DataFrame(
            np.hstack((X_train, y_train.reshape(-1, 1))),
            columns=feature_cols + [target_col],
        ),
        source=data_path,
        targets=target_col,
        name=dataset_name,
    )
    mlflow.log_input(train_dataset, context="training")

    val_dataset = mlflow.data.from_pandas(
        pd.DataFrame(
            np.hstack((X_val, y_val.reshape(-1, 1))),
            columns=feature_cols + [target_col],
        ),
        source=data_path,
        targets=target_col,
        name=dataset_name,
    )
    mlflow.log_input(val_dataset, context="validation")
    return (
        feature_cols,
        input_shape,
        X_train,
        y_train,
        train_index,
        X_val,
        y_val,
        val_index,
        X_test,
        y_test,
        test_index,
    )


def build_model(
    input_shape: tuple[int, int], learning_rate: float, clipnorm: float
) -> tf.keras.Model:
    """
    Build and compile the CNN-BiLSTM model.

    Args:
        input_shape: Shape of the input data
        learning_rate: Learning rate for the optimizer
        clipnorm: Gradient clipping norm value

    Returns:
        Compiled TensorFlow model
    """
    model = CNN_BiLSTM(input_shape=input_shape)
    _ = model.build()

    opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, clipnorm=clipnorm
    )
    _ = model.compile(optimizer=opt)

    logger.info("Model summary: %s", model.summary())
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: tf.keras.Model,
    epochs: int,
    batch_size: int,
) -> tf.keras.callbacks.History:
    """
    Train the model on the provided data.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model: Compiled TensorFlow model
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Training history object
    """
    early_stopping_monitor = "val_loss"
    early_stopping_patience = 5
    mlflow.log_param("early_stopping_monitor", early_stopping_monitor)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            early_stopping,
            TerminateOnNaN(),
            EarlyStoppingEpochLogger(),
        ],
    )

    return history


def plot_history_metric(
    history: tf.keras.callbacks.History,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot and save training history metrics.

    Args:
        history: Training history object
        metric: Name of the metric to plot
        title: Title for the plot
        ylabel: Label for the y-axis
        filename: Filename to save the plot
    """
    fig = plt.figure()
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("epoch")
    plt.legend()
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def evaluate_model(
    target_col: str,
    feature_cols: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
    data_path: str,
    dataset_name: str,
) -> tuple[
    np.ndarray,  # pred
    np.ndarray,  # fpr
    np.ndarray,  # tpr
    float,  # roc_auc
]:
    """
    Evaluate the model on test data and log metrics.

    Args:
        target_col: Name of the target column
        feature_cols: List of feature column names
        X_test: Test features
        y_test: Test labels
        model: Trained TensorFlow model
        data_path: Path to the data file
        dataset_name: Name of the dataset

    Returns:
        tuple containing:
        - pred: Binary predictions
        - pred_proba: Predicted probabilities
        - fpr: False positive rates for ROC curve
        - tpr: True positive rates for ROC curve
        - roc_auc: Area under the ROC curve

    Raises:
        ValueError: If predicted probabilities contain NaN values
    """
    pred_proba = model.predict(X_test)
    if np.isnan(pred_proba).any():
        raise ValueError("Predicted probabilities contain NaN values.")

    pred = (pred_proba > 0.5).astype(int)

    test_dataset = mlflow.data.from_pandas(
        pd.DataFrame(
            np.hstack((X_test, y_test.reshape(-1, 1), pred.reshape(-1, 1))),
            columns=feature_cols + [target_col, "pred"],
        ),
        source=data_path,
        targets=target_col,
        name=dataset_name,
        predictions="pred",
    )
    mlflow.log_input(test_dataset, context="test")

    acc = accuracy_score(y_test, pred)
    mlflow.log_metric("accuracy", acc)

    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    np.savez_compressed(
        Path("/tmp/roc_curve.npz"),
        fpr=fpr,
        tpr=tpr,
    )
    mlflow.log_artifact(Path("/tmp/roc_curve.npz"))

    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("roc_auc", roc_auc)
    return pred, pred_proba, fpr, tpr, roc_auc


def export_predictions(
    test_index: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    pred_proba: np.ndarray,
    data: pl.DataFrame,
    predictions_path: Path = Path("/tmp/predictions.parquet"),
    artifact_path: str = "predictions",
) -> None:
    """
    Export predictions to a Parquet file and log it to MLflow.

    Args:
        test_index: Indices of test samples
        y_test: True labels for test samples
        pred: Predicted labels for test samples
        pred_proba: Predicted probabilities for test samples
        data: Original data DataFrame
        predictions_path: Path to save the predictions file
        artifact_path: Artifact path in MLflow
    """

    predictions = pl.DataFrame(
        {
            "index": test_index.astype(np.uint32),
            "y": y_test.flatten(),
            "y_pred": pred.flatten().astype(np.bool),
            "y_pred_proba": pred_proba.flatten(),
        }
    )

    logger.debug(f"{predictions.shape=} {data.with_row_index().shape=}")

    predictions.join(
        data.with_row_index(), on="index", how="left"
    ).write_parquet(predictions_path, compression="lz4")
    mlflow.log_artifact(predictions_path, artifact_path=artifact_path)


def plot_confusion_matrix(y_test: np.ndarray, pred: np.ndarray) -> None:
    """
    Plot and save the confusion matrix.
    Args:
        y_test: True labels
        pred: Predicted labels
    """
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["BENIGN", "ANOMALOUS"]
    )
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    """
    Plot and save the ROC curve.
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
    """
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC curve")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)


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
        mlflow.set_tag("operation", "train_test")
        mlflow.set_tag("model_type", "CNN-BiLSTM")
        mlflow.set_tag("benign_labels", args.benign_labels)

        logger.info("Loading data...")
        data, target_col = load_data(
            data_path=args.data_path, dataset_name=args.dataset_name
        )

        logger.info("Preprocessing data...")
        (
            feature_cols,
            input_shape,
            X_train,
            y_train,
            train_index,
            X_val,
            y_val,
            val_index,
            X_test,
            y_test,
            test_index,
        ) = preprocess_data(
            data=data,
            target_col=target_col,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            test_size=args.test_size,
            val_size=args.val_size,
            benign_labels=args.benign_labels.split(","),
        )

        logger.info("Building model...")
        model = build_model(
            input_shape=input_shape,
            learning_rate=args.learning_rate,
            clipnorm=args.clipnorm,
        )

        logger.info("Training model...")
        history = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        for metric in ["loss", "accuracy"]:
            logger.info(f"Plotting {metric}...")
            plot_history_metric(
                history=history,
                metric=metric,
                title=f"model {metric}",
                ylabel=metric,
                filename=f"model_{metric}.png",
            )

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
