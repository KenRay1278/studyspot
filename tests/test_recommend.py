import pandas as pd

from src.data_validation import clean_items
from src.features import FEATURE_COLUMNS
from src.recommend import recommend_places


class IdentityPreprocessor:
    def transform(self, frame):
        return frame


class PriceAwareModel:
    def predict(self, frame):
        return 5 - frame["average_price"].to_numpy() / 100000


def test_recommendations_are_ranked_and_clipped():
    items = pd.DataFrame(
        [
            {
                "place_id": place_id,
                "place_name": name,
                "area": "Alam Sutera",
                "place_type": "Cafe",
                "maps_url": "https://example.com",
                "noise_level": 2,
                "crowd_level": 2,
                "table_capacity": 4,
                "wifi_quality": 4,
                "power_outlet_availability": 4,
                "average_price": price,
                "distance_to_binus_km": float(place_id),
                "open_time": "08:00",
                "close_time": "22:00",
                "open_on_sundays": 1,
            }
            for place_id, name, price in [
                (1, "Affordable", 20000),
                (2, "Midrange", 50000),
                (3, "Premium", 90000),
            ]
        ]
    )
    items = clean_items(items)
    preferences = {
        "study_frequency": 3,
        "main_study_purpose": "Solo study",
        "usual_group_size": 2,
        "noise_tolerance": 2,
        "crowd_tolerance": 2,
        "wifi_importance": 4,
        "outlet_importance": 4,
        "table_capacity_importance": 3,
        "avg_session_spending_numeric": 50000.0,
        "avg_session_spending_open_ended": 0,
        "distance_tolerance": 5.0,
        "distance_tolerance_unlimited": 0,
        "preferred_study_time": "Afternoon",
        "study_on_sundays": "Yes",
        "study_on_sundays_numeric": 1.0,
    }
    bundle = {
        "preprocessor": IdentityPreprocessor(),
        "model": PriceAwareModel(),
        "feature_columns": FEATURE_COLUMNS,
    }

    recommendations = recommend_places(
        preferences, items, bundle, top_n=2
    )
    assert recommendations["place_name"].tolist() == ["Affordable", "Midrange"]
    assert recommendations["rank"].tolist() == [1, 2]
    assert recommendations["predicted_rating"].between(1, 5).all()
    assert recommendations["explanation"].str.len().gt(0).all()
