# AGENTS.md

## Project Identity

Project name: **StudySpot — A Classical Machine Learning Recommender for Personalized Study Spaces Around BINUS Alam Sutera**

This repository implements a Machine Learning final project for BINUS University. The system recommends study spaces around BINUS Alam Sutera, Alam Sutera, Gading Serpong, and BSD areas.

The application must be a deployed, usable classical machine learning system. It must accept user input, execute a trained model, and produce clear, interpretable recommendations.

## Hard Constraints

Follow these constraints strictly:

1. Use **classical machine learning only**.
2. Do **not** use deep learning.
3. Do **not** use neural networks, embeddings, transformers, LLM-based recommendation, or deep collaborative filtering.
4. Main deployment target is **Streamlit**.
5. The system must not look like naive filtering only.
6. The system must execute a trained Scikit-learn model.
7. The app must accept user preference input and recommend ranked study places.
8. Outputs must be interpretable for non-technical users.
9. The codebase must be clean enough for GitHub submission, demo video, and PPT explanation.

Allowed model families:

* Mean rating baseline
* Rule-based preference scoring baseline
* Linear Regression
* Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor as a similarity-style comparison model

Preferred final model:

* Use Random Forest Regressor if it performs reasonably.
* Use Ridge Regression or Linear Regression as interpretable baselines.
* Use KNN Regressor only as a comparison model, not as the main project framing.

## ML Framing

Frame the system as a **hybrid supervised recommender system**.

Main formulation:

```text
user profile features + item/study-space features -> predicted suitability rating
```

Target variable:

```text
rating, from 1 to 5
```

Rating meaning:

```text
1 = very unsuitable
2 = unsuitable
3 = neutral / okay
4 = suitable
5 = very suitable
```

The app should:

1. collect a new user's study preferences,
2. combine those preferences with every item/study-space row,
3. predict a suitability rating for every place,
4. rank all places,
5. recommend the top-N places,
6. explain why each place was recommended.

## Dataset Files

Use these datasets:

```text
data/raw/ML_User_Persona_Dataset.csv
data/raw/ML_User_Rating_Dataset.csv
data/raw/ML_Item_Dataset.csv
```

## Important Dataset Semantics

Recommended derived feature:

```text
price_gap = average_price - avg_session_spending_numeric
```

Interpretation:

* Negative price gap: place is cheaper than the user's usual spending comfort.
* Near-zero price gap: place aligns with the user's usual spending comfort.
* Positive price gap: place is more expensive than the user's usual spending comfort.

## Dataset Structure

### Item Dataset

One row per study place.

Expected item columns include:

```text
id
place_name
area
place_type
maps_url
noise_level
crowd_level
table_capacity
wifi_quality
power_outlet_availability
average_price
distance_to_binus_km
open_time
close_time
open_on_sundays
is_24_hours
closes_next_day
```

If `is_24_hours` and `closes_next_day` are missing, derive them.

Rules:

* If `open_time = 00:00` and `close_time = 23:59`, set `is_24_hours = 1`.
* If `close_time` is earlier than `open_time`, set `closes_next_day = 1`.
* Do not remove items unless there is a clear duplicate or invalid row.

### User Persona Dataset

One row per user/respondent.

Expected features include:

```text
user_id
study_frequency
main_study_purpose
usual_group_size
noise_tolerance
crowd_tolerance
wifi_importance
outlet_importance
table_capacity_importance
avg_session_spending_raw
avg_session_spending_numeric
avg_session_spending_open_ended
distance_tolerance
preferred_study_time
study_on_sundays
```

The completed dataset should include users:

```text
u1 to u15
```

### Rating Dataset / Joined Training Dataset

The final training table should contain valid user-item interactions only.

Exclude blank or N/A ratings.

The joined training table should include:

```text
user_id
place_id or id
place_name
rating
user persona features
item features
derived compatibility features
```

There should be approximately 167 valid training interactions after excluding blank ratings.

## Feature Engineering Requirements

Build a feature engineering pipeline that is reusable for both training and app inference.

Recommended derived features:

```text
noise_gap = noise_level - noise_tolerance
crowd_gap = crowd_level - crowd_tolerance
wifi_gap = wifi_quality - wifi_importance
outlet_gap = power_outlet_availability - outlet_importance
table_capacity_gap = table_capacity - usual_group_size
price_gap = average_price - avg_session_spending_numeric
distance_gap = distance_to_binus_km - distance_tolerance
sunday_match = whether item open_on_sundays matches user's study_on_sundays need
```

These compatibility features are important because the model should learn preference-item fit, not memorize place IDs.

Do not use `place_id` or `id` as a strong memorization feature in the main model.

Safe use:

* Keep `id` for joining and display.
* Do not one-hot encode `id` unless creating a clearly labeled memorization/ablation experiment.
* Prefer item attributes and user-item compatibility features.

## Preprocessing Requirements

Use Scikit-learn Pipelines and ColumnTransformers where practical.

Handle numeric and categorical features separately.

Recommended preprocessing:

* Numeric features: median imputation + StandardScaler for linear/KNN models.
* Categorical features: most-frequent imputation + OneHotEncoder with `handle_unknown="ignore"`.
* Tree models do not require scaling, but using a consistent preprocessing pipeline is acceptable.
* Keep all transformations reproducible.

Set a fixed random seed:

```python
RANDOM_STATE = 42
```

## Model Training Requirements

Create a training script or notebook that:

1. loads cleaned datasets,
2. validates schema,
3. excludes blank ratings,
4. builds the training table if needed,
5. creates derived compatibility features,
6. trains multiple models,
7. evaluates them,
8. saves the best model and preprocessing pipeline,
9. saves metrics and feature list.

Recommended models:

```text
MeanRatingBaseline
RuleBasedPreferenceBaseline
LinearRegression
Ridge
DecisionTreeRegressor
RandomForestRegressor
KNeighborsRegressor
```

Main evaluation metrics:

```text
MAE
RMSE
R2
```

Also include recommender-oriented evaluation where possible:

```text
Top-N qualitative inspection
Predicted vs actual rating comparison
Per-user leave-one-user-out or grouped validation if feasible
```

Because the dataset is small, avoid overclaiming performance. Prefer honest wording:

```text
The model is evaluated as an MVP recommender using limited real user-item interactions.
```

## Validation Strategy

Do not use a random row split only as the sole validation method if avoidable, because rows from the same user may leak preference patterns.

Preferred evaluation:

1. Group-aware split by user if feasible.
2. Leave-One-User-Out cross-validation if feasible.
3. Also report a simple train/test split only as a secondary result.

If the grouped split becomes too unstable because the dataset is small, use K-fold or repeated split, but clearly document the limitation.

## Saved Artifacts

Save trained artifacts under:

```text
models/
```

Expected artifacts:

```text
models/best_model.joblib
models/preprocessor.joblib
models/model_bundle.joblib
models/metrics.json
models/feature_columns.json
```

Prefer saving a single `model_bundle.joblib` containing:

```python
{
    "model": trained_model,
    "preprocessor": preprocessor,
    "feature_columns": feature_columns,
    "target_name": "rating",
    "metadata": {
        "project_name": "StudySpot",
        "model_type": "...",
        "random_state": 42
    }
}
```

## Streamlit App Requirements

Create a Streamlit app that:

1. loads the saved model bundle,
2. loads the item dataset,
3. asks the user for study preferences,
4. creates one candidate row per study place,
5. predicts suitability rating for every place,
6. ranks places from highest to lowest predicted rating,
7. displays top-N recommendations,
8. explains each recommendation.

Recommended file:

```text
app.py
```

or:

```text
streamlit_app.py
```

## Streamlit Input Fields

The app should ask for these user preferences:

```text
study_frequency
main_study_purpose
usual_group_size
noise_tolerance
crowd_tolerance
wifi_importance
outlet_importance
table_capacity_importance
avg_session_spending
distance_tolerance
preferred_study_time
study_on_sundays
top_n
```

Use clear non-technical labels.

Examples:

```text
How often do you study outside home?
What is your main study purpose?
How many people usually study with you?
How much noise can you tolerate?
How crowded can the place be?
How important is Wi-Fi quality?
How important are power outlets?
How important is table capacity?
How much are you usually willing to spend per study session?
How far are you comfortable traveling from BINUS Alam Sutera?
When do you usually study?
Do you usually study on Sundays?
How many recommendations do you want?
```

## Streamlit Output Requirements

For each recommended place, display:

```text
rank
place_name
area
place_type
predicted suitability rating
average price
distance to BINUS
noise level
crowd level
Wi-Fi quality
power outlet availability
table capacity
open time and close time
Sunday availability
Google Maps link
short explanation
```

Predicted rating should be clipped to the 1–5 range for display.

Example:

```text
Predicted suitability: 4.3 / 5
```

## Explanation Logic

Recommendation explanations should be rule-based and interpretable.

Example explanation components:

```text
Matches your preferred distance range.
Has Wi-Fi quality suitable for laptop-based study.
Has enough table capacity for your group size.
More expensive than your usual spending comfort, but matches your facility preferences.
May be noisier than your stated tolerance.
Open on Sundays, matching your study habit.
```

Do not claim causal certainty.

Avoid wording like:

```text
The model knows you will like this place.
```

Use:

```text
This place is predicted to match your preferences based on similar user ratings and study-space attributes.
```

## Suggested Repository Structure

Use this structure unless the repository already has a better one:

```text
studyspot/
├── AGENTS.md
├── README.md
├── requirements.txt
├── app.py
├── data/
│   ├── StudySpot_User_Persona_Completed_Cleaned.csv
│   ├── StudySpot_Item_Dataset_With_Derived_Time_Features.csv
│   ├── StudySpot_Training_Interactions_Joined.csv
│   └── raw/
├── models/
│   ├── model_bundle.joblib
│   ├── metrics.json
│   └── feature_columns.json
├── notebooks/
│   └── 01_eda_and_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_validation.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── recommend.py
│   └── explain.py
├── reports/
│   ├── figures/
│   └── metrics_summary.csv
└── tests/
    ├── test_features.py
    └── test_recommend.py
```

## Required Python Libraries

Use these unless additional libraries are clearly necessary:

```text
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
seaborn
```

Optional:

```text
plotly
```

Do not add unnecessary heavy dependencies.

## Commands

Use these commands for local setup:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python -m src.train
```

Run app:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

Only add `pytest` if tests are created.

## README Requirements

Create or update `README.md` with:

1. project title,
2. problem statement,
3. ML formulation,
4. dataset description,
5. model list,
6. evaluation metrics,
7. how to train,
8. how to run the Streamlit app,
9. deployment instructions,
10. limitations.

Keep the README student-project appropriate and concise.

## PPT/Report Support

The implementation should make it easy to report:

```text
Dataset Description
EDA
ML Problem Formulation
Model Selection & Rationale
Training & Validation Setup
Evaluation Metrics & Results
Deployment Architecture
Application Screenshots
User Testing Design
User Testing Results
Limitations
Conclusion & Future Work
```

Save figures and metrics where useful.

## Coding Style

Write simple, readable Python.

Prioritize correctness and explainability over abstraction.

Avoid overengineering.

Use clear function names.

Add comments only where they clarify non-obvious logic.

Use type hints where useful, but do not make the code unnecessarily complex.

## Safety and Privacy

Do not expose real respondent identities.

Use anonymized user IDs only.

Do not collect private personal data in the Streamlit app.

Do not store user submissions unless explicitly implemented for user testing.

## Final Implementation Goal

The final app should convincingly demonstrate:

```text
A classical supervised hybrid recommender that predicts study-space suitability ratings from user preferences and study-space attributes, then ranks and explains recommended places.
```

The project should not be framed as a simple filter or static rule-based website.
