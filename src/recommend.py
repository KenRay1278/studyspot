from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explain import explanation_text
from src.features import build_candidate_rows, select_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.joblib"


def load_model_bundle(path: Path = DEFAULT_BUNDLE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_with_bundle(bundle: dict, feature_frame: pd.DataFrame) -> np.ndarray:
    model_input = select_model_features(feature_frame)
    transformed = bundle["preprocessor"].transform(model_input)
    return bundle["model"].predict(transformed)


def recommend_places(
    user_preferences: dict[str, object],
    items: pd.DataFrame,
    bundle: dict,
    top_n: int = 5,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    candidates = build_candidate_rows(user_preferences, items)
    predictions = predict_with_bundle(bundle, candidates)
    candidates["predicted_rating_raw"] = predictions
    candidates["predicted_rating"] = np.clip(predictions, 1.0, 5.0)
    candidates["explanation"] = candidates.apply(explanation_text, axis=1)

    ranked = candidates.sort_values(
        ["predicted_rating", "distance_to_binus_km", "average_price"],
        ascending=[False, True, True],
    ).head(min(top_n, len(candidates)))
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked

