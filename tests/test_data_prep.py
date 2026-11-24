"""Testy jednostkowe modułu data_prep."""
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from data_prep import load_data, split_data


@pytest.fixture
def df():
    """Zbiór Wine w pamięci - testy nie zależą od pliku na dysku."""
    return load_wine(as_frame=True).frame


def test_split_test_size_proportion(df):
    _, X_test, _, _ = split_data(df, test_size=0.2, seed=42)
    assert len(X_test) == pytest.approx(len(df) * 0.2, abs=1)


def test_split_sizes_sum_to_total(df):
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=42)
    assert len(X_train) + len(X_test) == len(df) +1
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_removes_target_from_features(df):
    X_train, X_test, _, _ = split_data(df, test_size=0.2, seed=42)
    assert "target" not in X_train.columns
    assert X_train.shape[1] == df.shape[1] - 1


def test_split_is_reproducible(df):
    first = split_data(df, test_size=0.2, seed=42)
    second = split_data(df, test_size=0.2, seed=42)
    assert first[1].index.equals(second[1].index)


def test_split_empty_dataframe_raises():
    with pytest.raises(ValueError):
        split_data(pd.DataFrame(), test_size=0.2, seed=42)


def test_split_missing_target_raises(df):
    with pytest.raises(ValueError):
        split_data(df.drop(columns="target"), test_size=0.2, seed=42)


@pytest.mark.parametrize("bad_size", [0, 1, -0.1, 1.5])
def test_split_invalid_test_size_raises(df, bad_size):
    with pytest.raises(ValueError):
        split_data(df, test_size=bad_size, seed=42)


def test_load_data_reads_csv(tmp_path, df):
    path = tmp_path / "dane.csv"
    df.to_csv(path, index=False)
    loaded = load_data(str(path))
    assert loaded.shape == df.shape


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "nie_istnieje.csv"))
