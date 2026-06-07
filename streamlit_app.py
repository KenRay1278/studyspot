from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_validation import ITEM_PATH, clean_items
from src.recommend import load_model_bundle, recommend_places


st.set_page_config(
    page_title="StudySpot",
    page_icon=":round_pushpin:",
    layout="wide",
)


@st.cache_resource
def get_model_bundle() -> dict:
    return load_model_bundle()


@st.cache_data
def get_items() -> pd.DataFrame:
    return clean_items(pd.read_csv(ITEM_PATH))


def rupiah(value: float) -> str:
    return f"Rp{value:,.0f}".replace(",", ".")


st.title("StudySpot")
st.caption(
    "Personalized study-space recommendations around BINUS Alam Sutera, "
    "powered by a classical machine learning model."
)

try:
    bundle = get_model_bundle()
    items = get_items()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("Train the project first with `python -m src.train`.")
    st.stop()

with st.sidebar:
    st.header("Your Study Preferences")
    study_frequency = st.slider(
        "How often do you study outside home?", 1, 5, 3,
        help="1 means rarely and 5 means very often.",
    )
    main_study_purpose = st.selectbox(
        "What is your main study purpose?",
        [
            "Solo study",
            "Group assignment / discussion",
            "Coding / laptop work",
            "Exam preparation",
            "Casual study",
        ],
    )
    usual_group_size = st.slider(
        "How many people usually study with you?", 1, 8, 2
    )
    noise_tolerance = st.slider(
        "How much noise can you tolerate?", 1, 5, 2,
        help="1 means you need quiet; 5 means noise is acceptable.",
    )
    crowd_tolerance = st.slider(
        "How crowded can the place be?", 1, 5, 2
    )
    wifi_importance = st.slider(
        "How important is Wi-Fi quality?", 1, 5, 4
    )
    outlet_importance = st.slider(
        "How important are power outlets?", 1, 5, 4
    )
    table_capacity_importance = st.slider(
        "How important is table capacity?", 1, 5, 3
    )
    spending = st.number_input(
        "Usual spending per study session (Rp)",
        min_value=0,
        max_value=300000,
        value=50000,
        step=5000,
    )
    distance_tolerance = st.slider(
        "Comfortable distance from BINUS Alam Sutera (km)",
        0.5,
        20.0,
        5.0,
        0.5,
    )
    preferred_study_time = st.selectbox(
        "When do you usually study?",
        ["Morning", "Afternoon", "Evening", "No Preference"],
    )
    sunday_label = st.selectbox(
        "Do you usually study on Sundays?", ["No", "Sometimes", "Yes"]
    )
    top_n = st.slider(
        "How many recommendations do you want?",
        3,
        min(10, len(items)),
        5,
    )
    recommend_clicked = st.button(
        "Find My Study Spots", type="primary", use_container_width=True
    )

if not recommend_clicked:
    st.subheader("How it works")
    st.write(
        "StudySpot combines your preferences with the attributes of every "
        "available study place. Its trained regression model predicts a "
        "suitability rating, ranks the places, and explains practical matches "
        "and tradeoffs."
    )
    model_name = bundle["metadata"]["model_type"]
    st.info(f"Current trained model: {model_name}")
    st.stop()

sunday_numeric = {"No": 0.0, "Sometimes": 0.5, "Yes": 1.0}[sunday_label]
preferences = {
    "study_frequency": study_frequency,
    "main_study_purpose": main_study_purpose,
    "usual_group_size": usual_group_size,
    "noise_tolerance": noise_tolerance,
    "crowd_tolerance": crowd_tolerance,
    "wifi_importance": wifi_importance,
    "outlet_importance": outlet_importance,
    "table_capacity_importance": table_capacity_importance,
    "avg_session_spending_numeric": float(spending),
    "avg_session_spending_open_ended": 0,
    "distance_tolerance": float(distance_tolerance),
    "distance_tolerance_unlimited": 0,
    "preferred_study_time": preferred_study_time,
    "study_on_sundays": sunday_label,
    "study_on_sundays_numeric": sunday_numeric,
}

recommendations = recommend_places(
    preferences, items, bundle, top_n=top_n
)

st.subheader("Your Recommended Study Spots")
st.caption(
    "Predictions are estimates based on the available ratings and place "
    "attributes. Consider the explanations as practical guidance, not certainty."
)

for _, place in recommendations.iterrows():
    with st.container(border=True):
        heading, score = st.columns([4, 1])
        heading.markdown(
            f"### #{int(place['rank'])} {place['place_name']}"
        )
        heading.caption(f"{place['area']} | {place['place_type']}")
        score.metric(
            "Predicted suitability", f"{place['predicted_rating']:.1f} / 5"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**Average price:** {rupiah(place['average_price'])}")
        col2.write(
            f"**Distance:** {place['distance_to_binus_km']:.1f} km"
        )
        col3.write(
            f"**Hours:** {place['open_time']}-{place['close_time']}"
        )
        col4.write(
            f"**Sunday:** {'Open' if place['open_on_sundays'] else 'Closed'}"
        )

        st.write(
            f"Noise **{int(place['noise_level'])}/5** | "
            f"Crowd **{int(place['crowd_level'])}/5** | "
            f"Wi-Fi **{int(place['wifi_quality'])}/5** | "
            f"Outlets **{int(place['power_outlet_availability'])}/5** | "
            f"Table capacity **{int(place['table_capacity'])} people**"
        )
        st.write(place["explanation"])
        st.link_button("Open in Google Maps", place["maps_url"])

with st.expander("About this prediction"):
    st.write(
        "This place is predicted to match your preferences based on anonymized "
        "user ratings and study-space attributes. StudySpot uses "
        f"{bundle['metadata']['model_type']}, a classical Scikit-learn model."
    )
