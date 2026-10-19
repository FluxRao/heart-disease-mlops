import logging
import time

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("heart_disease_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        logger.error("Model file not found")
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    logger.info("Heart disease model loaded successfully")

    yield

    model = None
    logger.info("Heart disease API shutting down")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Heart disease classification using a trained ML model",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "Request failed | method=%s | path=%s | duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Request completed | method=%s | path=%s | status=%d | duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response

class PatientData(BaseModel):
    age: Annotated[float, Field(ge=0, le=120)]
    sex: Annotated[int, Field(ge=0, le=1)]
    cp: Annotated[int, Field(ge=1, le=4)]
    trestbps: Annotated[float, Field(gt=0)]
    chol: Annotated[float, Field(ge=0)]
    fbs: Annotated[int, Field(ge=0, le=1)]
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[float, Field(ge=0)]
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: float
    slope: Annotated[int, Field(ge=1, le=3)]
    ca: Annotated[int | None, Field(ge=0, le=3)] = None
    thal: Annotated[int | None, Field(ge=3, le=7)] = None


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        input_df = pd.DataFrame([patient.model_dump()])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        # Confidence is the probability assigned to the predicted class.
        confidence = float(probabilities[prediction])

        return {
            "prediction": prediction,
            "prediction_label": (
                "Heart Disease" if prediction == 1 else "No Heart Disease"
            ),
            "confidence": round(confidence, 4),
            "heart_disease_probability": round(
                float(probabilities[1]), 4
            ),
        }

    except Exception:
        logger.exception("Prediction processing failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        ) from None
