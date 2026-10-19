from fastapi.testclient import TestClient

from src.api import app


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


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_PATIENT)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "Heart Disease",
        "No Heart Disease",
    ]
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["heart_disease_probability"] <= 1


def test_invalid_patient_data():
    invalid_patient = SAMPLE_PATIENT.copy()
    invalid_patient["age"] = -10

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422


def test_missing_required_feature():
    incomplete_patient = SAMPLE_PATIENT.copy()
    del incomplete_patient["age"]

    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete_patient)

    assert response.status_code == 422
