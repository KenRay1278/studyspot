import pandas as pd

from src.data_validation import clean_items, clean_personas
from src.features import add_compatibility_features


def test_persona_numeric_parsing():
    frame = pd.DataFrame(
        [
            {
                "user_code": f"u{index}",
                "study_frequency": 3,
                "main_study_purpose": "Coding/Laptop Work",
                "usual_group_size": "3-4",
                "noise_tolerance": 2,
                "crowd_tolerance": 2,
                "wifi_importance": 5,
                "outlet_importance": 4,
                "table_capacity_importance": 3,
                "avg_session_spending": ">100000",
                "travel_distance_tolerance": "Distance does not matter",
                "preferred_study_time": "evening",
                "studies_on_sunday": "Sometimes ",
            }
            for index in range(1, 16)
        ]
    )

    cleaned = clean_personas(frame)
    first = cleaned.iloc[0]
    assert first["usual_group_size"] == 4
    assert first["avg_session_spending_numeric"] == 125000
    assert first["avg_session_spending_open_ended"] == 1
    assert first["distance_tolerance"] == 20
    assert first["main_study_purpose"] == "Coding / laptop work"
    assert first["study_on_sundays_numeric"] == 0.5


def test_item_time_features_and_compatibility_gaps():
    item = pd.DataFrame(
        [
            {
                "place_id": 1,
                "place_name": "Late Cafe",
                "area": "Alam Sutera",
                "place_type": "Cafe",
                "maps_url": "https://example.com",
                "noise_level": 3,
                "crowd_level": 2,
                "table_capacity": 4,
                "wifi_quality": 5,
                "power_outlet_availability": 4,
                "average_price": 40000,
                "distance_to_binus_km": 2.0,
                "open_time": "10:00",
                "close_time": "03:00",
                "open_on_sundays": 1,
            }
        ]
    )
    cleaned_item = clean_items(item)
    assert cleaned_item.loc[0, "closes_next_day"] == 1
    assert cleaned_item.loc[0, "is_24_hours"] == 0

    row = cleaned_item.assign(
        study_frequency=3,
        main_study_purpose="Solo study",
        usual_group_size=3,
        noise_tolerance=2,
        crowd_tolerance=2,
        wifi_importance=4,
        outlet_importance=5,
        table_capacity_importance=3,
        avg_session_spending_numeric=50000,
        avg_session_spending_open_ended=0,
        distance_tolerance=3.0,
        distance_tolerance_unlimited=0,
        preferred_study_time="Evening",
        study_on_sundays="Yes",
        study_on_sundays_numeric=1.0,
    )
    featured = add_compatibility_features(row)
    assert featured.loc[0, "price_gap"] == -10000
    assert featured.loc[0, "noise_gap"] == 1
    assert featured.loc[0, "table_capacity_gap"] == 1
    assert featured.loc[0, "preferred_time_match"] == 1

