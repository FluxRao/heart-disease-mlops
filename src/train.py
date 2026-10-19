
from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_processing import build_preprocessor, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_raw.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "screenshots"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# Convert target to binary
y = (df["num"] > 0).astype(int)
X = df[FEATURE_COLUMNS].copy()

print("Dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Define two classification models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.1, 1.0, 10.0]
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        },
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

results = []
trained_models = {}


# 4. Cross-validation and hyperparameter tuning
for model_name, (classifier, param_grid) in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_

    # Cross-validation scores on training data
    cv_scores = cross_validate(
        clone(best_pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
    )

    result = {
        "model": model_name,
        "cv_accuracy": float(np.mean(cv_scores["test_accuracy"])),
        "cv_precision": float(np.mean(cv_scores["test_precision"])),
        "cv_recall": float(np.mean(cv_scores["test_recall"])),
        "cv_roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
        "tuning_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
    }

    results.append(result)
    trained_models[model_name] = best_pipeline

    print("Best parameters:", search.best_params_)
    print("CV ROC-AUC:", round(result["cv_roc_auc"], 4))


# 5. Compare models using CV results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="tuning_roc_auc",
    ascending=False,
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False,
)

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# 6. Final evaluation on untouched test data
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

print("\nFINAL TEST METRICS:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))


# 7. Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Disease", "Disease"],
    cmap="Blues",
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200,
)
plt.close()


# 8. Save model and evaluation results
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

with open(MODEL_DIR / "test_metrics.json", "w") as file:
    json.dump(test_metrics, file, indent=4)

print("\nModel saved to models/best_model.joblib")
print("Training completed successfully!")
