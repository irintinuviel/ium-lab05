"""Etap 1: wczytanie i przygotowanie danych."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")


def load_data(path: str) -> pd.DataFrame:
    """Wczytuje dane z pliku CSV i zwraca DataFrame."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Brak pliku z danymi: {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float, seed: int,
               target: str = TARGET_COLUMN):
    """Dzieli dane na (X_train, X_test, y_train, y_test) z zachowaniem proporcji klas."""
    if df.empty:
        raise ValueError("Zbiór danych jest pusty")
    if target not in df.columns:
        raise ValueError(f"Brak kolumny docelowej '{target}'")
    if not 0 < test_size < 1:
        raise ValueError("test_size musi być z przedziału (0, 1)")

    X = df.drop(columns=target)
    y = df[target]
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


def export_wine(path: str) -> None:
    """Zapisuje zbiór Wine ze scikit-learn do pliku CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    load_wine(as_frame=True).frame.to_csv(path, index=False)


def main() -> None:
    export_wine(DATA_PATH)
    print(f"[data_prep] Zapisano dane do {DATA_PATH}: {load_data(DATA_PATH).shape}")


if __name__ == "__main__":
    main()
