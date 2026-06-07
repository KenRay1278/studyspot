from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def regression_metrics(
    actual: np.ndarray | pd.Series, predicted: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(np.asarray(predicted, dtype=float), 1.0, 5.0)
    actual_array = np.asarray(actual, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual_array, clipped)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, clipped))),
        "r2": float(r2_score(actual_array, clipped)),
    }


def evaluate_pipeline_grouped(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, float]] = []
    predictions = np.zeros(len(target), dtype=float)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(features.iloc[train_index], target.iloc[train_index])
        fold_predictions = fold_pipeline.predict(features.iloc[test_index])
        predictions[test_index] = fold_predictions
        fold_metrics = regression_metrics(
            target.iloc[test_index], fold_predictions
        )
        fold_rows.append({"fold": fold, **fold_metrics})

    fold_frame = pd.DataFrame(fold_rows)
    summary = regression_metrics(target, predictions)
    summary.update(
        {
            "mae_std": float(fold_frame["mae"].std(ddof=0)),
            "rmse_std": float(fold_frame["rmse"].std(ddof=0)),
            "r2_std": float(fold_frame["r2"].std(ddof=0)),
        }
    )
    return summary, fold_frame


def evaluate_mean_baseline_grouped(
    target: pd.Series, groups: pd.Series, n_splits: int = 5
) -> tuple[dict[str, float], pd.DataFrame]:
    dummy_features = np.zeros((len(target), 1))
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.zeros(len(target), dtype=float)
    fold_rows: list[dict[str, float]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(dummy_features, target, groups), start=1
    ):
        predictions[test_index] = target.iloc[train_index].mean()
        fold_metrics = regression_metrics(
            target.iloc[test_index], predictions[test_index]
        )
        fold_rows.append({"fold": fold, **fold_metrics})

    fold_frame = pd.DataFrame(fold_rows)
    summary = regression_metrics(target, predictions)
    summary.update(
        {
            "mae_std": float(fold_frame["mae"].std(ddof=0)),
            "rmse_std": float(fold_frame["rmse"].std(ddof=0)),
            "r2_std": float(fold_frame["r2"].std(ddof=0)),
        }
    )
    return summary, fold_frame


def rule_based_predictions(feature_frame: pd.DataFrame) -> np.ndarray:
    noise_fit = np.clip(1 - np.maximum(feature_frame["noise_gap"], 0) / 4, 0, 1)
    crowd_fit = np.clip(1 - np.maximum(feature_frame["crowd_gap"], 0) / 4, 0, 1)
    wifi_fit = np.clip(1 + np.minimum(feature_frame["wifi_gap"], 0) / 4, 0, 1)
    outlet_fit = np.clip(
        1 + np.minimum(feature_frame["outlet_gap"], 0) / 4, 0, 1
    )
    table_fit = np.clip(
        1 + np.minimum(feature_frame["table_capacity_gap"], 0) / 6, 0, 1
    )
    price_scale = np.maximum(
        feature_frame["avg_session_spending_numeric"].to_numpy(), 25000
    )
    price_fit = np.clip(
        1 - np.maximum(feature_frame["price_gap"], 0) / price_scale, 0, 1
    )
    distance_scale = np.maximum(
        feature_frame["distance_tolerance"].to_numpy(), 1
    )
    distance_fit = np.clip(
        1 - np.maximum(feature_frame["distance_gap"], 0) / distance_scale, 0, 1
    )

    weighted_fit = (
        0.12 * noise_fit
        + 0.10 * crowd_fit
        + 0.17 * wifi_fit
        + 0.14 * outlet_fit
        + 0.12 * table_fit
        + 0.13 * price_fit
        + 0.12 * distance_fit
        + 0.05 * feature_frame["sunday_match"].to_numpy()
        + 0.05 * feature_frame["preferred_time_match"].to_numpy()
    )
    return 1 + 4 * weighted_fit


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

