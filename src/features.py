from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "study_frequency",
    "usual_group_size",
    "noise_tolerance",
    "crowd_tolerance",
    "wifi_importance",
    "outlet_importance",
    "table_capacity_importance",
    "avg_session_spending_numeric",
    "avg_session_spending_open_ended",
    "distance_tolerance",
    "distance_tolerance_unlimited",
    "study_on_sundays_numeric",
    "noise_level",
    "crowd_level",
    "table_capacity",
    "wifi_quality",
    "power_outlet_availability",
    "average_price",
    "distance_to_binus_km",
    "open_on_sundays",
    "is_24_hours",
    "closes_next_day",
    "opening_duration_hours",
    "noise_gap",
    "crowd_gap",
    "wifi_gap",
    "outlet_gap",
    "table_capacity_gap",
    "price_gap",
    "distance_gap",
    "sunday_match",
    "preferred_time_match",
]

CATEGORICAL_FEATURES = [
    "main_study_purpose",
    "preferred_study_time",
    "area",
    "place_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _is_open_at(row: pd.Series, target_minutes: int) -> float:
    if row["is_24_hours"] == 1:
        return 1.0

    opening = int(row["open_minutes"])
    closing = int(row["close_minutes"])
    if row["closes_next_day"] == 1:
        return float(target_minutes >= opening or target_minutes <= closing)
    return float(opening <= target_minutes <= closing)


def add_compatibility_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = frame.copy()
    featured["noise_gap"] = (
        featured["noise_level"] - featured["noise_tolerance"]
    )
    featured["crowd_gap"] = (
        featured["crowd_level"] - featured["crowd_tolerance"]
    )
    featured["wifi_gap"] = (
        featured["wifi_quality"] - featured["wifi_importance"]
    )
    featured["outlet_gap"] = (
        featured["power_outlet_availability"]
        - featured["outlet_importance"]
    )
    featured["table_capacity_gap"] = (
        featured["table_capacity"] - featured["usual_group_size"]
    )
    featured["price_gap"] = (
        featured["average_price"]
        - featured["avg_session_spending_numeric"]
    )
    featured["distance_gap"] = (
        featured["distance_to_binus_km"] - featured["distance_tolerance"]
    )
    featured["sunday_match"] = 1.0 - (
        featured["open_on_sundays"]
        - featured["study_on_sundays_numeric"]
    ).abs()

    preferred_minutes = {
        "Morning": 9 * 60,
        "Afternoon": 14 * 60,
        "Evening": 19 * 60,
    }
    featured["preferred_time_match"] = featured.apply(
        lambda row: (
            1.0
            if row["preferred_study_time"] == "No Preference"
            else _is_open_at(
                row, preferred_minutes.get(row["preferred_study_time"], 14 * 60)
            )
        ),
        axis=1,
    )
    return featured


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Model input is missing feature columns: {missing}")
    return frame[FEATURE_COLUMNS].copy()


def build_candidate_rows(
    user_preferences: dict[str, object], items: pd.DataFrame
) -> pd.DataFrame:
    candidates = items.copy()
    for column, value in user_preferences.items():
        candidates[column] = value
    return add_compatibility_features(candidates)

