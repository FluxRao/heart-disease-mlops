
from pathlib import Path
import sys
import json

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

from data_processing import build_preprocessor, FEATURE_COLUMNS


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_raw.csv"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heart_Disease_Classification")

RANDOM_STATE = 42


# Load dataset
df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS].copy()
y = (df["num"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# Model configurations
models = {
    "Logistic_Regression": {
        "classifier": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "params": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
    "Random_Forest": {
        "classifier": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        },
    },
}


# Train and track each model
for model_name, config in models.items():

    with mlflow.start_run(run_name=model_name):

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", config["classifier"]),
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Log experiment parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params(search.best_params_)

        # Log training CV metric
        mlflow.log_metric(
            "cv_roc_auc",
            float(search.best_score_),
        )

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
        }

        mlflow.log_metrics(metrics)

        print("Best parameters:", search.best_params_)
        print("CV ROC-AUC:", round(search.best_score_, 4))
        print("Test metrics:", metrics)

        # Save confusion matrix artifact
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No Disease", "Disease"],
            cmap="Blues",
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        plot_path = PROJECT_ROOT / f"{model_name}_confusion.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        mlflow.log_artifact(str(plot_path), artifact_path="plots")
        plot_path.unlink()

        # Save trained model as MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        # Log dataset metadata
        mlflow.log_dict(
            {
                "dataset": "UCI Heart Disease",
                "rows": len(df),
                "features": FEATURE_COLUMNS,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            },
            "dataset_info.json",
        )

        print(f"MLflow run completed: {model_name}")

print("\nAll MLflow experiments completed!")
