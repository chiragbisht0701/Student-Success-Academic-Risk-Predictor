import numpy as np
import pandas as pd

def generate_student_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    attendance = np.clip(rng.normal(78, 12, n_samples), 35, 100)
    study = np.clip(rng.normal(2.8, 1.4, n_samples), 0, 8)
    sleep = np.clip(rng.normal(7.0, 1.1, n_samples), 4, 10)
    assignment = np.clip(rng.normal(72, 15, n_samples), 20, 100)
    previous = np.clip(rng.normal(70, 16, n_samples), 20, 100)
    extra = np.clip(rng.normal(1.8, 1.2, n_samples), 0, 6)
    internet = rng.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])

    risk_score = (
        0.055 * (75 - attendance)
        + 0.55 * (2.5 - study)
        + 0.35 * np.abs(sleep - 7.2)
        + 0.035 * (65 - assignment)
        + 0.04 * (65 - previous)
        + 0.18 * np.maximum(extra - 3.0, 0)
        + np.where(internet == "No", 0.7, 0.0)
        + rng.normal(0, 0.75, n_samples)
    )

    at_risk = (risk_score > 0.9).astype(int)

    return pd.DataFrame({
        "attendance_pct": attendance.round(1),
        "study_hours_per_day": study.round(1),
        "sleep_hours": sleep.round(1),
        "assignment_score": assignment.round(1),
        "previous_score": previous.round(1),
        "extracurricular_hours": extra.round(1),
        "internet_access": internet,
        "at_risk": at_risk,
    })
