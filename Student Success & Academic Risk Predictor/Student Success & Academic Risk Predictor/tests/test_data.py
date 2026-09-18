from src.data_generator import generate_student_data
from src.data_loader import validate_dataframe

def test_generated_data_is_valid():
    df = generate_student_data(n_samples=100, random_state=1)
    validate_dataframe(df, training=True)
    assert len(df) == 100
    assert set(df["at_risk"].unique()).issubset({0, 1})
