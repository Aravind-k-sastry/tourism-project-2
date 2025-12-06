#!/usr/bin/env python3
"""
train.py

Train multiple model families with a common preprocessing pipeline,
perform threshold tuning for best F1, log experiments to MLflow,
and upload winning model to Hugging Face Hub.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

# xgboost
import xgboost as xgb

# MLflow
import mlflow

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -------------------------
# Configuration (edit if needed)
# -------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "mlops-training-experiment-2"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face repo settings
HF_REPO_ID = "aravindshaz3/predictor-model-2"  # change if desired
HF_REPO_TYPE = "model"  # "model" or "space" if needed

# Data paths (hf:// URLs you had earlier)
Xtrain_path = "hf://datasets/aravindshaz3/travel-package-sales-2/Xtrain.csv"
Xtest_path = "hf://datasets/aravindshaz3/travel-package-sales-2/Xtest.csv"
ytrain_path = "hf://datasets/aravindshaz3/travel-package-sales-2/ytrain.csv"
ytest_path = "hf://datasets/aravindshaz3/travel-package-sales-2/ytest.csv"

# -------------------------
# Utility functions
# -------------------------
def find_best_threshold(y_true, probs):
    """
    Given true binary labels and probability estimates for the positive class,
    find the threshold that maximizes F1 on the provided data.
    Returns: best_threshold (float), best_f1 (float)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve returns array of length n_thresholds + 1 for precisions/recalls,
    # thresholds length is n_thresholds. We align by ignoring last precision/recall entry.
    best_f1 = -1.0
    best_t = 0.5
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if (p + r) == 0:
            continue
        f1 = 2 * (p * r) / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_with_threshold(estimator, X, y, threshold=0.5):
    """Return dict of metrics when using a given probability threshold for positive class."""
    probs = estimator.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }
    return metrics, probs, preds


# -------------------------
# Main training flow
# -------------------------
def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading datasets...")
    Xtrain = pd.read_csv(Xtrain_path)
    Xtest = pd.read_csv(Xtest_path)
    ytrain = pd.read_csv(ytrain_path).squeeze()
    ytest = pd.read_csv(ytest_path).squeeze()

    # Basic sanity
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}, ytrain: {ytrain.shape}, ytest: {ytest.shape}")

    # Feature lists (as in your prior script)
    numeric_features = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    ]

    categorical_features = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
    ]

    # Compute class weight ratio (for XGBoost scale_pos_weight)
    # Using ytrain.value_counts() expects binary labels 0/1
    class_counts = ytrain.value_counts()
    if 1 in class_counts and 0 in class_counts:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0

    # Preprocessor
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        remainder="drop",
    )

    # Model catalog and grids (use 'model' named step in Pipeline, so param keys use model__*)
    models_and_grids = {
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "Bagging": (
            BaggingClassifier(random_state=42),
            {
                "model__n_estimators": [10, 50],
                "model__max_samples": [0.5, 1.0],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [5, 7, None],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.01, 0.1, 1.0],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.01, 0.1],
            },
        ),
        "XGBoost": (
            xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss", random_state=42),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__colsample_bytree": [0.4, 0.6],
            },
        ),
    }

    # Cross-validation splitter
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Global tracking of best model (by test F1)
    best_global = {
        "name": None,
        "estimator": None,
        "test_f1": -1.0,
        "test_metrics": None,
        "best_threshold": 0.5,
    }

    # Start one top-level MLflow run for this full training job
    with mlflow.start_run(run_name="multi_model_selection") as parent_run:
        mlflow.log_param("models_tried", list(models_and_grids.keys()))

        for model_name, (model_obj, param_grid) in models_and_grids.items():
            print(f"\n========== Training & tuning: {model_name} ==========")
            pipe = Pipeline([("preproc", preprocessor), ("model", model_obj)])

            # Grid search
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )

            # Fit
            grid.fit(Xtrain, ytrain)
            best_est = grid.best_estimator_
            best_cv_score = grid.best_score_
            best_params = grid.best_params_

            print(f"{model_name} CV best F1: {best_cv_score:.4f}")
            print(f"{model_name} best params: {best_params}")

            # Nested MLflow run for this model family
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params({"model_family": model_name, **best_params})
                mlflow.log_metric("cv_best_f1", float(best_cv_score))

                # Threshold tuning: use predict_proba on training data to find best threshold that maximizes F1
                try:
                    _, train_probs, _ = evaluate_with_threshold(best_est, Xtrain, ytrain, threshold=0.5)
                    best_t, best_f1_on_train = find_best_threshold(ytrain, train_probs)
                    mlflow.log_metric("train_best_threshold", float(best_t))
                    mlflow.log_metric("train_best_f1", float(best_f1_on_train))
                except Exception as e:
                    # some estimators may not support predict_proba; fall back to 0.5
                    print(f"Threshold tuning skipped for {model_name}: {e}")
                    best_t = 0.5

                # Evaluate on test set with that threshold
                test_metrics, test_probs, test_preds = evaluate_with_threshold(best_est, Xtest, ytest, threshold=best_t)
                print(f"{model_name} Test metrics at threshold {best_t}: {test_metrics}")

                # Log test metrics
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", float(v))

                mlflow.log_metric("test_f1_at_threshold", float(test_metrics["f1"]))
                mlflow.log_metric("test_accuracy_at_threshold", float(test_metrics["accuracy"]))

                # Save the model artifact for this run (not necessarily global best)
                model_filename = OUTPUT_DIR / f"{model_name}_best.joblib"
                joblib.dump(best_est, model_filename)
                mlflow.log_artifact(str(model_filename), artifact_path=f"models/{model_name}")

            # Compare and update global best (choose by test F1)
            if test_metrics["f1"] > best_global["test_f1"]:
                best_global["name"] = model_name
                best_global["estimator"] = best_est
                best_global["test_f1"] = test_metrics["f1"]
                best_global["test_metrics"] = test_metrics
                best_global["best_threshold"] = best_t

        # After trying all models
        print("\n========== Model selection complete ==========")
        print(f"Best model overall: {best_global['name']} with test F1 = {best_global['test_f1']:.4f}")
        mlflow.log_param("best_model_name", best_global["name"])
        mlflow.log_metric("best_model_test_f1", float(best_global["test_f1"]))

        # Persist best global model locally and as MLflow artifact
        final_model_path = OUTPUT_DIR / f"best_model_{best_global['name']}.joblib"
        joblib.dump(best_global["estimator"], final_model_path)
        mlflow.log_artifact(str(final_model_path), artifact_path="models/best_overall")
        # Persist threshold too
        threshold_file = OUTPUT_DIR / "best_threshold.txt"
        threshold_file.write_text(str(best_global["best_threshold"]))
        mlflow.log_artifact(str(threshold_file), artifact_path="models/best_overall")

        # Attempt to upload to Hugging Face model repo
        api = HfApi()
        try:
            # If repo doesn't exist, create it
            try:
                api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
                print(f"Hugging Face repo {HF_REPO_ID} exists.")
            except RepositoryNotFoundError:
                print(f"Hugging Face repo {HF_REPO_ID} not found. Creating...")
                create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, private=False)
                print("Created HF repo.")

            # Upload files
            print("Uploading model and threshold to Hugging Face repo (this requires HF token in environment)...")
            api.upload_file(
                path_or_fileobj=str(final_model_path),
                path_in_repo=final_model_path.name,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
            )
            api.upload_file(
                path_or_fileobj=str(threshold_file),
                path_in_repo=threshold_file.name,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
            )
            print("Upload to Hugging Face completed.")
        except HfHubHTTPError as e:
            print(f"Hugging Face HTTP error: {e}. Continue without HF upload.")
        except Exception as e:
            print(f"Unexpected error uploading to Hugging Face: {e}. Continue without HF upload.")

    # Done
    print("Training finished. Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
