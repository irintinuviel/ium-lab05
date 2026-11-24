"""Etap 2: trening modelu i zapis do models/."""
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, split_data

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression")
MAX_ITER = int(os.getenv("MAX_ITER", 1000))
C = float(os.getenv("C", 1.0))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
API_KEY = os.getenv("API_KEY")  # sekret - nigdy nie wypisujemy jego wartości


def build_model() -> Pipeline:
    """Tworzy Pipeline: skalowanie + klasyfikator wybrany w MODEL_TYPE."""
    if MODEL_TYPE == "logistic_regression":
        clf = LogisticRegression(C=C, max_iter=MAX_ITER)
    elif MODEL_TYPE == "random_forest":
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Nieznany MODEL_TYPE: {MODEL_TYPE}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main() -> None:
    df = load_data(DATA_PATH)
    X_train, _, y_train, _ = split_data(df, TEST_SIZE, RANDOM_SEED)

    model = build_model()
    model.fit(X_train, y_train)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if API_KEY:
        # symulacja: tu model mógłby zostać wysłany do rejestru modeli
        print(f"[train] API_KEY otrzymany ({len(API_KEY)} znaków) - "
              "uwierzytelnienie w rejestrze modeli (symulacja)")
    else:
        print("[train] Brak API_KEY - pomijam wysyłkę do rejestru modeli")
    print(f"[train] Wytrenowano {MODEL_TYPE} na {len(X_train)} próbkach, zapisano: {MODEL_PATH}")


if __name__ == "__main__":
    main()
