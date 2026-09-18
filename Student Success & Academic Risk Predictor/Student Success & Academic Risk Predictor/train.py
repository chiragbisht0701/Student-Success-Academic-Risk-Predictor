from src.config import DATA_DIR, DATA_PATH, MODEL_PATH
from src.data_generator import generate_student_data
from src.data_loader import load_csv
from src.model_training import train_best_model, save_model

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        df = generate_student_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"Generated demo dataset: {DATA_PATH}")

    df = load_csv(DATA_PATH, training=True)
    result = train_best_model(df)
    save_model(result.pipeline, MODEL_PATH)

    print(f"Selected model: {result.model_name}")
    print("Selected-model metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
