from src.data_generator import generate_student_data
from src.model_training import train_best_model
from src.predictor import predict_student

def test_training_and_prediction():
    df = generate_student_data(n_samples=300, random_state=7)
    result = train_best_model(df)

    assert result.model_name in {"Logistic Regression", "Random Forest"}
    assert 0 <= result.metrics["f1"] <= 1

    sample = {
        "attendance_pct": 55,
        "study_hours_per_day": 1.0,
        "sleep_hours": 5.5,
        "assignment_score": 50,
        "previous_score": 48,
        "extracurricular_hours": 4.0,
        "internet_access": "Yes",
    }
    pred = predict_student(result.pipeline, sample)
    assert pred["prediction"] in {0, 1}
