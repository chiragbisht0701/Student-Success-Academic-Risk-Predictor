from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "student_data.csv"
MODEL_PATH = MODEL_DIR / "student_risk_pipeline.joblib"

FEATURES = [
    "attendance_pct",
    "study_hours_per_day",
    "sleep_hours",
    "assignment_score",
    "previous_score",
    "extracurricular_hours",
    "internet_access",
]

NUMERIC_FEATURES = [
    "attendance_pct",
    "study_hours_per_day",
    "sleep_hours",
    "assignment_score",
    "previous_score",
    "extracurricular_hours",
]

CATEGORICAL_FEATURES = ["internet_access"]
TARGET = "at_risk"

RANDOM_STATE = 42
