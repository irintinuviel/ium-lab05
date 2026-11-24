"""Etap 3: ewaluacja zapisanego modelu, bez ponownego treningu."""
import json
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
RESULTS_PATH = os.getenv("RESULTS_PATH", "results/wyniki.txt")
METRICS_PATH = os.getenv("METRICS_PATH", "results/metrics.json")


def main() -> dict:
    model = joblib.load(MODEL_PATH)
    df = load_data(DATA_PATH)
    _, X_test, _, y_test = split_data(df, TEST_SIZE, RANDOM_SEED)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Liczba próbek testowych: {len(y_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # metryki w formacie JSON - odczytywane przez usługę visualize
    metrics = {"model_type": MODEL_TYPE, "accuracy": round(acc, 4),
               "n_test": len(y_test)}
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[evaluate] Accuracy: {acc:.4f}")
    print(report)
    print(f"[evaluate] Wyniki zapisano do {RESULTS_PATH} i {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
