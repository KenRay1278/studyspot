from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PERSONA_PATH = RAW_DATA_DIR / "ML_User_Persona_Dataset.csv"
RATING_PATH = RAW_DATA_DIR / "ML_User_Rating_Dataset.csv"
ITEM_PATH = RAW_DATA_DIR / "ML_Item_Dataset.csv"

PERSONA_REQUIRED_COLUMNS = {
    "user_code",
    "study_frequency",
    "main_study_purpose",
    "usual_group_size",
    "noise_tolerance",
    "crowd_tolerance",
    "wifi_importance",
    "outlet_importance",
    "table_capacity_importance",
    "avg_session_spending",
    "travel_distance_tolerance",
    "preferred_study_time",
    "studies_on_sunday",
}

ITEM_REQUIRED_COLUMNS = {
    "place_id",
    "place_name",
    "area",
    "place_type",
    "maps_url",
    "noise_level",
    "crowd_level",
    "table_capacity",
    "wifi_quality",
    "power_outlet_availability",
    "average_price",
    "distance_to_binus_km",
    "open_time",
    "close_time",
    "open_on_sundays",
}

RATING_REQUIRED_COLUMNS = {
    "user_code",
    "place_id",
    "place_name",
    "rating",
    "form_version",
}


def validate_columns(
    frame: pd.DataFrame, required_columns: set[str], dataset_name: str
) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _normalize_purpose(value: object) -> str:
    text = str(value).strip().lower().replace("/", " / ")
    text = " ".join(text.split())
    if "coding" in text or "laptop" in text:
        return "Coding / laptop work"
    if "group" in text:
        return "Group assignment / discussion"
    if "exam" in text:
        return "Exam preparation"
    if "casual" in text:
        return "Casual study"
    return "Solo study"


def parse_group_size(value: object) -> int:
    text = str(value).strip()
    if "-" in text:
        return int(text.split("-")[-1])
    return int(float(text))


def parse_spending(value: object) -> tuple[float, int]:
    text = str(value).strip().replace(",", "")
    is_open_ended = int(text.startswith(">"))
    numeric_text = text.lstrip(">").strip()
    numeric = float(numeric_text)
    if is_open_ended:
        numeric += 25000.0
    return numeric, is_open_ended


def parse_distance_tolerance(value: object) -> tuple[float, int]:
    text = str(value).strip()
    if text.lower() == "distance does not matter":
        return 20.0, 1
    return float(text), 0


def _time_to_minutes(value: object) -> int:
    hours, minutes = str(value).strip().split(":")
    return int(hours) * 60 + int(minutes)


def clean_personas(frame: pd.DataFrame) -> pd.DataFrame:
    validate_columns(frame, PERSONA_REQUIRED_COLUMNS, "User persona dataset")
    cleaned = frame.copy()
    cleaned = cleaned.rename(
        columns={
            "user_code": "user_id",
            "travel_distance_tolerance": "distance_tolerance",
            "studies_on_sunday": "study_on_sundays",
        }
    )

    cleaned["user_id"] = _clean_text(cleaned["user_id"])
    cleaned["main_study_purpose"] = cleaned["main_study_purpose"].map(
        _normalize_purpose
    )
    cleaned["usual_group_size_raw"] = _clean_text(cleaned["usual_group_size"])
    cleaned["usual_group_size"] = cleaned["usual_group_size"].map(parse_group_size)

    cleaned["avg_session_spending_raw"] = _clean_text(
        cleaned["avg_session_spending"]
    )
    spending = cleaned["avg_session_spending"].map(parse_spending)
    cleaned["avg_session_spending_numeric"] = spending.map(lambda pair: pair[0])
    cleaned["avg_session_spending_open_ended"] = spending.map(lambda pair: pair[1])
    cleaned = cleaned.drop(columns=["avg_session_spending"])

    cleaned["distance_tolerance_raw"] = _clean_text(cleaned["distance_tolerance"])
    distance = cleaned["distance_tolerance"].map(parse_distance_tolerance)
    cleaned["distance_tolerance"] = distance.map(lambda pair: pair[0])
    cleaned["distance_tolerance_unlimited"] = distance.map(lambda pair: pair[1])

    cleaned["preferred_study_time"] = _clean_text(
        cleaned["preferred_study_time"]
    ).str.title()
    cleaned["study_on_sundays"] = _clean_text(
        cleaned["study_on_sundays"]
    ).str.title()
    cleaned["study_on_sundays_numeric"] = cleaned["study_on_sundays"].map(
        {"No": 0.0, "Sometimes": 0.5, "Yes": 1.0}
    )

    if cleaned["user_id"].duplicated().any():
        duplicates = cleaned.loc[
            cleaned["user_id"].duplicated(keep=False), "user_id"
        ].tolist()
        raise ValueError(f"Duplicate user IDs found: {duplicates}")

    expected_users = {f"u{index}" for index in range(1, 16)}
    actual_users = set(cleaned["user_id"])
    if actual_users != expected_users:
        raise ValueError(
            "Persona dataset must contain anonymized users u1 to u15. "
            f"Missing: {sorted(expected_users - actual_users)}; "
            f"unexpected: {sorted(actual_users - expected_users)}"
        )

    return cleaned


def clean_items(frame: pd.DataFrame) -> pd.DataFrame:
    validate_columns(frame, ITEM_REQUIRED_COLUMNS, "Item dataset")
    cleaned = frame.copy()
    for column in [
        "place_name",
        "area",
        "place_type",
        "maps_url",
        "open_time",
        "close_time",
    ]:
        cleaned[column] = _clean_text(cleaned[column])

    cleaned["open_minutes"] = cleaned["open_time"].map(_time_to_minutes)
    cleaned["close_minutes"] = cleaned["close_time"].map(_time_to_minutes)
    cleaned["is_24_hours"] = (
        (cleaned["open_time"] == "00:00") & (cleaned["close_time"] == "23:59")
    ).astype(int)
    cleaned["closes_next_day"] = (
        (cleaned["close_minutes"] < cleaned["open_minutes"])
        & (cleaned["is_24_hours"] == 0)
    ).astype(int)

    effective_close = cleaned["close_minutes"] + (
        cleaned["closes_next_day"] * 24 * 60
    )
    cleaned["opening_duration_hours"] = (
        effective_close - cleaned["open_minutes"]
    ) / 60
    cleaned.loc[cleaned["is_24_hours"] == 1, "opening_duration_hours"] = 24.0

    if cleaned["place_id"].duplicated().any():
        duplicates = cleaned.loc[
            cleaned["place_id"].duplicated(keep=False), "place_id"
        ].tolist()
        raise ValueError(f"Duplicate place IDs found: {duplicates}")

    numeric_ranges = {
        "noise_level": (1, 5),
        "crowd_level": (1, 5),
        "wifi_quality": (1, 5),
        "power_outlet_availability": (1, 5),
        "open_on_sundays": (0, 1),
    }
    for column, (minimum, maximum) in numeric_ranges.items():
        invalid = ~cleaned[column].between(minimum, maximum)
        if invalid.any():
            raise ValueError(
                f"Item column {column} contains values outside "
                f"{minimum}-{maximum}."
            )

    return cleaned


def clean_ratings(
    frame: pd.DataFrame, valid_users: set[str], valid_places: set[int]
) -> pd.DataFrame:
    validate_columns(frame, RATING_REQUIRED_COLUMNS, "Rating dataset")
    cleaned = frame.copy().rename(columns={"user_code": "user_id"})
    cleaned["user_id"] = _clean_text(cleaned["user_id"])
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned = cleaned.dropna(subset=["rating"]).copy()
    cleaned["rating"] = cleaned["rating"].astype(float)

    invalid_ratings = ~cleaned["rating"].between(1, 5)
    if invalid_ratings.any():
        raise ValueError("Ratings must be between 1 and 5.")

    unknown_users = set(cleaned["user_id"]) - valid_users
    unknown_places = set(cleaned["place_id"]) - valid_places
    if unknown_users or unknown_places:
        raise ValueError(
            f"Unknown rating references. Users: {sorted(unknown_users)}; "
            f"places: {sorted(unknown_places)}"
        )

    if cleaned.duplicated(["user_id", "place_id"]).any():
        raise ValueError("Duplicate user-place rating interactions found.")

    return cleaned


def load_clean_datasets(
    persona_path: Path = PERSONA_PATH,
    rating_path: Path = RATING_PATH,
    item_path: Path = ITEM_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    personas = clean_personas(pd.read_csv(persona_path))
    items = clean_items(pd.read_csv(item_path))
    ratings = clean_ratings(
        pd.read_csv(rating_path),
        valid_users=set(personas["user_id"]),
        valid_places=set(items["place_id"]),
    )
    return personas, ratings, items


def build_training_table(
    personas: pd.DataFrame, ratings: pd.DataFrame, items: pd.DataFrame
) -> pd.DataFrame:
    rating_columns = ["user_id", "place_id", "rating", "form_version"]
    joined = ratings[rating_columns].merge(
        personas, on="user_id", how="left", validate="many_to_one"
    )
    joined = joined.merge(
        items, on="place_id", how="left", validate="many_to_one"
    )

    if joined.isna().any().any():
        missing = joined.columns[joined.isna().any()].tolist()
        raise ValueError(f"Joined training data contains missing values: {missing}")

    return joined


def save_clean_datasets(
    personas: pd.DataFrame,
    items: pd.DataFrame,
    training_table: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or (PROJECT_ROOT / "data")
    output_dir.mkdir(parents=True, exist_ok=True)
    personas.to_csv(
        output_dir / "StudySpot_User_Persona_Completed_Cleaned.csv", index=False
    )
    items.to_csv(
        output_dir / "StudySpot_Item_Dataset_With_Derived_Time_Features.csv",
        index=False,
    )
    training_table.to_csv(
        output_dir / "StudySpot_Training_Interactions_Joined.csv", index=False
    )
