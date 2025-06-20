#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import mlflow
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from train_cnn_bilstm_cicflowmeter_binary_classifier import (
    setup_logging,
    load_data,
    preprocess_data,
    export_predictions,
)

logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.Namespace:
    """
    Set up and configure the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Create baseline on CICFlowMeter datasets as binary classifier"
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
        default="Baseline-CICFlowMeter",
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
        required=True,
        help="Model type",
        choices=[
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "LogisticRegression",
            "XGBClassifier",
            "IsolationForest",
        ],
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test size"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.0, help="Validation size"
    )
    parser.add_argument(
        "--benign-labels",
        type=str,
        default="benign,BENIGN",
        help="Comma-separated list of benign labels",
    )

    return parser.parse_args()


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
        mlflow.set_tag("operation", "baseline")
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("benign_labels", args.benign_labels)

        # Load data
        logger.info("Loading data...")
        data, target_col = load_data(
            data_path=args.data_path, dataset_name=args.dataset_name
        )

        # Preprocess data
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
            val_size=0.0,
            benign_labels=args.benign_labels.split(","),
        )

        # Training
        logger.info(f"Building {args.model_type} model...")
        if args.model_type == "IsolationForest":
            from sklearn.ensemble import IsolationForest

            contamination_value = min(0.5, y_train.mean())
            model = IsolationForest(
                n_estimators=100,
                max_samples="auto",
                contamination=contamination_value,
                random_state=42,
            )

            logger.info("Training model...")
            model.fit(X_train[y_train == 0])
        else:
            # Supervised classifiers
            if args.model_type == "RandomForestClassifier":
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=100,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                )
            elif args.model_type == "DecisionTreeClassifier":
                from sklearn.tree import DecisionTreeClassifier

                model = DecisionTreeClassifier(random_state=42)
            elif args.model_type == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(
                    penalty="l2", C=1.0, max_iter=1000, random_state=42
                )
            elif args.model_type == "XGBClassifier":
                from xgboost import XGBClassifier

                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                )

            logger.info("Training model...")
            model.fit(X_train, y_train)

        # Evaluation
        if args.model_type == "IsolationForest":
            # predict: -1 outliers, 1 inliers
            raw_pred = model.predict(X_test)
            pred = (raw_pred == -1).astype(int)
            # use decision_function for scores (higher = inlier)
            scores = -model.decision_function(X_test)  # anomaly scores
            auc = roc_auc_score(y_test, scores)
        else:
            pred = model.predict(X_test)
            scores = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, scores)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        # Log metrics
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        logger.info(
            f"ROC AUC: {auc:.4f}, Acc: {acc:.4f}, "
            f"Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}"
        )

        # Export predictions
        export_predictions(test_index, y_test, pred, scores, data)


if __name__ == "__main__":
    main()
