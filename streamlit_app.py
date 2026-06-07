from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_validation import ITEM_PATH, clean_items
from src.recommend import load_model_bundle, recommend_places


FREQUENCY_OPTIONS = {
    "1 - Rarely": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5 - Very often": 5,
}
NOISE_TOLERANCE_OPTIONS = {
    "1 - Need quiet": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5 - Noise is fine": 5,
}
CROWD_TOLERANCE_OPTIONS = {
    "1 - Prefer empty": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5 - Crowds are fine": 5,
}
IMPORTANCE_OPTIONS = {
    "1 - Not important": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5 - Very important": 5,
}

NOISE_LABELS = {
    1: "Very quiet",
    2: "Quiet",
    3: "Moderate",
    4: "Noisy",
    5: "Very noisy",
}
CROWD_LABELS = {
    1: "Almost empty",
    2: "Uncrowded",
    3: "Moderately crowded",
    4: "Crowded",
    5: "Very crowded",
}
WIFI_LABELS = {1: "Very poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
OUTLET_LABELS = {
    1: "Very limited",
    2: "Limited",
    3: "Moderate",
    4: "Available",
    5: "Widely available",
}


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


def semantic_input(
    label: str, options: dict[str, int], default: str, help_text: str | None = None
) -> int:
    selected = st.select_slider(
        label,
        options=list(options),
        value=default,
        help=help_text,
    )
    return options[selected]


def apply_responsive_theme(dark_mode: bool) -> None:
    if dark_mode:
        background = "#0E1715"
        surface = "#162522"
        sidebar = "#12201D"
        text = "#F1F7F5"
        muted = "#B5C8C3"
        border = "#2B4640"
    else:
        background = "#F7FAF9"
        surface = "#FFFFFF"
        sidebar = "#EAF3F1"
        text = "#18332E"
        muted = "#506C65"
        border = "#C8DCD7"

    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"] {{
            background: {background};
            color: {text};
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: {sidebar};
        }}
        [data-testid="stHeader"] {{
            background: color-mix(in srgb, {background} 88%, transparent);
        }}
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stMetric"] {{
            background: {surface};
            border-color: {border};
        }}
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp [data-testid="stCaptionContainer"] {{
            color: {text};
        }}
        .stApp [data-testid="stCaptionContainer"] p {{
            color: {muted};
        }}
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem 0.8rem 4rem;
            }}
            [data-testid="stHorizontalBlock"] {{
                flex-direction: column;
                gap: 0.5rem;
            }}
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
            }}
            [data-testid="stMetric"] {{
                padding: 0.75rem;
            }}
            h1 {{
                font-size: 2rem !important;
            }}
            .stButton button, .stLinkButton a {{
                width: 100%;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


dark_mode = st.sidebar.toggle(
    "Dark mode",
    value=False,
    help="Use a darker color scheme that is easier on the eyes at night.",
)
apply_responsive_theme(dark_mode)

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
    st.caption("Describe your ideal study session. There are no wrong answers.")
    study_frequency = semantic_input(
        "How often do you study outside home?",
        FREQUENCY_OPTIONS,
        "3",
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
    noise_tolerance = semantic_input(
        "What noise level are you comfortable with?",
        NOISE_TOLERANCE_OPTIONS,
        "2",
    )
    crowd_tolerance = semantic_input(
        "How crowded can the place be?",
        CROWD_TOLERANCE_OPTIONS,
        "2",
    )
    st.divider()
    st.subheader("Facilities")
    wifi_importance = semantic_input(
        "How important is Wi-Fi quality?",
        IMPORTANCE_OPTIONS,
        "4",
    )
    outlet_importance = semantic_input(
        "How important are power outlets?",
        IMPORTANCE_OPTIONS,
        "4",
    )
    table_capacity_importance = semantic_input(
        "How important is table capacity?",
        IMPORTANCE_OPTIONS,
        "3",
    )
    st.divider()
    st.subheader("Budget and Schedule")
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
            f"Noise **{NOISE_LABELS[int(place['noise_level'])]}** | "
            f"Crowd **{CROWD_LABELS[int(place['crowd_level'])]}** | "
            f"Wi-Fi **{WIFI_LABELS[int(place['wifi_quality'])]}** | "
            f"Outlets "
            f"**{OUTLET_LABELS[int(place['power_outlet_availability'])]}** | "
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
