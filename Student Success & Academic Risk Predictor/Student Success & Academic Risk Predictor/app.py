import pandas as pd
import streamlit as st

from src.config import DATA_PATH, MODEL_PATH, FEATURES
from src.data_generator import generate_student_data
from src.data_loader import validate_dataframe
from src.model_training import train_best_model, save_model
from src.predictor import load_model, predict_student

st.set_page_config(page_title="Student Success AI", page_icon="🎓", layout="wide")
st.title("🎓 AI-Powered Student Success & Academic Risk Predictor")
st.caption("Educational ML project: train, evaluate, analyze, and predict.")

tab1, tab2, tab3 = st.tabs(["Dataset & Training", "Prediction", "Analytics"])

with tab1:
    st.subheader("1. Dataset")
    uploaded = st.file_uploader("Upload training CSV (optional)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
        else:
            df = generate_student_data()
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)

    try:
        validate_dataframe(df, training=True)
        st.success(f"Dataset valid: {len(df)} rows")
        st.dataframe(df.head(20), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if st.button("Train & Compare Models", type="primary"):
        with st.spinner("Training models..."):
            result = train_best_model(df)
            save_model(result.pipeline, MODEL_PATH)

        st.success(f"Best model: {result.model_name}")
        st.write("Best model metrics")
        st.json(result.metrics)
        st.write("Comparison")
        comparison = pd.DataFrame(result.all_results).T[["accuracy", "precision", "recall", "f1"]]
        st.dataframe(comparison, use_container_width=True)

with tab2:
    st.subheader("2. Predict Academic Risk")

    if not MODEL_PATH.exists():
        st.warning("Train a model first in the Dataset & Training tab.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            study = st.slider("Study hours/day", 0.0, 10.0, 3.0, 0.5)
            sleep = st.slider("Sleep hours/night", 3.0, 12.0, 7.0, 0.5)
            assignment = st.slider("Assignment score", 0, 100, 75)
        with c2:
            previous = st.slider("Previous score", 0, 100, 70)
            extra = st.slider("Extracurricular hours/day", 0.0, 8.0, 1.5, 0.5)
            internet = st.selectbox("Internet access", ["Yes", "No"])

        if st.button("Predict Risk"):
            pipeline = load_model(MODEL_PATH)
            student = {
                "attendance_pct": attendance,
                "study_hours_per_day": study,
                "sleep_hours": sleep,
                "assignment_score": assignment,
                "previous_score": previous,
                "extracurricular_hours": extra,
                "internet_access": internet,
            }
            result = predict_student(pipeline, student)
            st.metric("Prediction", result["label"])
            if result["risk_probability"] is not None:
                st.metric("Estimated risk probability", f'{result["risk_probability"]:.1%}')
            st.info("Use this output as decision support, not as a final academic judgment.")

with tab3:
    st.subheader("3. Analytics")
    if "df" not in locals():
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
        else:
            df = generate_student_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Students", len(df))
    col2.metric("At-risk rate", f'{df["at_risk"].mean():.1%}')
    col3.metric("Average attendance", f'{df["attendance_pct"].mean():.1f}%')

    st.write("Average values by risk group")
    numeric = [c for c in FEATURES if c != "internet_access"]
    summary = df.groupby("at_risk")[numeric].mean().round(2)
    st.dataframe(summary, use_container_width=True)

    st.write("Risk count")
    counts = df["at_risk"].map({0: "Not At Risk", 1: "At Risk"}).value_counts()
    st.bar_chart(counts)
