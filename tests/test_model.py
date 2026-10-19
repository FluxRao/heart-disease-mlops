from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_processing import build_preprocessor, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


SAMPLE_PATIENT = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 2,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_feature_count():
    assert len(FEATURE_COLUMNS) == 13
    assert len(set(FEATURE_COLUMNS)) == 13


def test_preprocessor_handles_missing_values():
    df = pd.DataFrame([
        SAMPLE_PATIENT,
        {
            **SAMPLE_PATIENT,
            "age": np.nan,
            "chol": np.nan,
            "ca": np.nan,
        },
    ])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


def test_saved_model_exists():
    assert MODEL_PATH.is_file(), "Trained model file is missing"


def test_saved_model_predicts():
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([SAMPLE_PATIENT])

    prediction = model.predict(df)

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)


def test_saved_model_probabilities():
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([SAMPLE_PATIENT])

    probabilities = model.predict_proba(df)

    assert probabilities.shape == (1, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)
