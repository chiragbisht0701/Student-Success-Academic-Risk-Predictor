from pathlib import Path
import joblib
import pandas as pd
from .config import FEATURES

def load_model(path: str | Path):
    return joblib.load(path)

def predict_student(pipeline, student: dict) -> dict:
    row = pd.DataFrame([student])[FEATURES]
    prediction = int(pipeline.predict(row)[0])

    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0][1])

    return {
        "prediction": prediction,
        "label": "At Risk" if prediction == 1 else "Not At Risk",
        "risk_probability": probability,
    }
