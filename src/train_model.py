
import os
from pathlib import Path

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from huggingface_hub import hf_hub_download, HfApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV


DATASET_REPO = "Lasya679/engine-predictive-maintenance"
MODEL_REPO = "Lasya679/engine-predictive-maintenance-model"

MODEL_FILENAME = "engine_predictive_maintenance_model.joblib"


def load_data():
    # Download the registered train and test datasets
    train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="train_data.csv",
        repo_type="dataset",
    )

    test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="test_data.csv",
        repo_type="dataset",
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def train_model(train_df):
    # Separate predictors and target
    X_train = train_df.drop("Engine Condition", axis=1)
    y_train = train_df["Engine Condition"]

    # Random Forest classifier
    rf_classifier = RandomForestClassifier(
        random_state=42
    )

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    # Tune using F1-score
    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, test_df):
    X_test = test_df.drop("Engine Condition", axis=1)
    y_test = test_df["Engine Condition"]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("Test Performance:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics


def save_and_register_model(model):
    model_dir = Path("artifacts")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )

    api = HfApi(token=hf_token)

    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    print("Model uploaded successfully to Hugging Face.")
    
def save_metadata(best_params, metrics, train_shape, test_shape):
    metadata_dir = Path("artifacts")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "RandomForestClassifier",
        "dataset": DATASET_REPO,
        "model_repository": MODEL_REPO,
        "training_rows": train_shape[0],
        "testing_rows": test_shape[0],
        "training_features": train_shape[1] - 1,
        "testing_features": test_shape[1] - 1,
        "best_parameters": best_params,
        "test_metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = metadata_dir / "model_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")

def main():
    print("Loading train and test data...")
    train_df, test_df = load_data()

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    print("Training and tuning Random Forest...")
    model, best_params = train_model(train_df)

    metrics = evaluate_model(model, test_df)

    save_and_register_model(model)

    save_metadata(
        best_params,
        metrics,
        train_df.shape,
        test_df.shape,
    )
        
    


if __name__ == "__main__":
    main()
