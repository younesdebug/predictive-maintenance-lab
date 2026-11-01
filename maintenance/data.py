"""Strict readers for the upstream simulated turbofan benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

COLUMNS = ["engine_id", "cycle", *[f"setting{i}" for i in range(1, 4)],
           *[f"sensor{i}" for i in range(1, 22)]]
FEATURES = COLUMNS[1:]  # No engine identifier, remaining life or target.


def read_engines(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS) or frame.empty:
        raise ValueError("Expected a non-empty file with 26 numeric columns.")
    frame.columns = COLUMNS
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All sensor values must be numeric.") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or infinite sensor values are not supported.")
    for column in ["engine_id", "cycle"]:
        if ((frame[column] <= 0) | (frame[column] % 1 != 0)).any():
            raise ValueError(f"{column} must contain positive integers.")
        frame[column] = frame[column].astype(int)
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle observations.")
    return frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def add_training_target(frame: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("The horizon must be a positive integer number of cycles.")
    result = frame.copy()
    result["rul"] = result.groupby("engine_id")["cycle"].transform("max") - result["cycle"]
    result["failure_within_horizon"] = (result["rul"] <= horizon).astype(int)
    return result


def split_by_engine(frame: pd.DataFrame, seed: int = 42):
    if frame.engine_id.nunique() < 4:
        raise ValueError("At least four engines are required for grouped validation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, validation = next(splitter.split(frame, groups=frame.engine_id))
    return frame.iloc[train].copy(), frame.iloc[validation].copy()


def last_test_observations(frame: pd.DataFrame, truth_path: str | Path,
                           horizon: int = 30) -> pd.DataFrame:
    """Truth row n refers to test engine n+1, at its last recorded cycle."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("The horizon must be a positive integer number of cycles.")
    truth = pd.read_csv(truth_path, sep=r"\s+", header=None)
    last = frame.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).copy()
    last = last.sort_values("engine_id").reset_index(drop=True)
    if truth.shape != (len(last), 1):
        raise ValueError("One remaining-life value is required per test engine.")
    if last.engine_id.tolist() != list(range(1, len(last) + 1)):
        raise ValueError("Test engine IDs must match the ordered truth file: 1..N.")
    values = pd.to_numeric(truth.iloc[:, 0], errors="raise").to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Remaining life must contain finite non-negative integer cycles.")
    last["rul"] = values.astype(int)
    last["failure_within_horizon"] = (last.rul <= horizon).astype(int)
    return last
