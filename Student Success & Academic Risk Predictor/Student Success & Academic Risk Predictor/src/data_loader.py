from pathlib import Path
import pandas as pd
from .config import FEATURES, TARGET

def load_csv(path: str | Path, training: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_dataframe(df, training=training)
    return df

def validate_dataframe(df: pd.DataFrame, training: bool = True) -> None:
    required = FEATURES + ([TARGET] if training else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if training and not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column 'at_risk' must contain only 0 and 1.")
