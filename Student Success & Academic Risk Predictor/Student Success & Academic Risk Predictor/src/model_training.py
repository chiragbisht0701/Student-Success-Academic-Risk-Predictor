from dataclasses import dataclass
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import FEATURES, TARGET, RANDOM_STATE
from .preprocessing import build_preprocessor
from .evaluation import classification_metrics

@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict
    all_results: dict

def _candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

def train_best_model(df: pd.DataFrame) -> TrainingResult:
    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("Training data must contain both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    all_results = {}
    best_name = None
    best_pipeline = None
    best_metrics = None
    best_f1 = -1.0

    for name, model in _candidate_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = classification_metrics(y_test, pred)
        all_results[name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    return TrainingResult(
        model_name=best_name,
        pipeline=best_pipeline,
        metrics=best_metrics,
        all_results=all_results,
    )

def save_model(pipeline: Pipeline, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
