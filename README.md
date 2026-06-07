# StudySpot

StudySpot is a classical machine learning recommender for personalized study
spaces around BINUS Alam Sutera, Alam Sutera, Gading Serpong, and BSD. It was
created as an end-to-end BINUS University Machine Learning final project.

## Problem and ML Formulation

Students have different preferences for noise, crowds, Wi-Fi, outlets, group
capacity, price, distance, schedule, and Sunday availability. StudySpot frames
the task as a hybrid supervised regression recommender:

```text
user preference features + study-space features
-> predicted suitability rating from 1 to 5
-> ranked Top-N recommendations
```

The application executes a trained Scikit-learn model. It is not a static
filter: every candidate place receives a model prediction before ranking.

## Dataset

The raw data is stored in `data/raw/`:

- `ML_User_Persona_Dataset.csv`: 15 anonymized user profiles.
- `ML_User_Rating_Dataset.csv`: 180 submitted interactions, including blank
  ratings that are excluded from training.
- `ML_Item_Dataset.csv`: 40 study places and their attributes.

The training pipeline validates joins, normalizes inconsistent survey labels,
derives time features, and creates compatibility features such as `noise_gap`,
`wifi_gap`, `price_gap`, and `distance_gap`. User and place IDs are retained
only for joins and display, not model memorization.

Cleaning assumptions are explicit and reproducible: group-size ranges use the
upper bound so capacity checks are conservative, `>100000` spending is encoded
as Rp125,000 with a separate open-ended flag, and "distance does not matter" is
represented as 20 km with a separate unlimited-distance flag.

## Models and Evaluation

The project compares:

- Mean-rating baseline
- Rule-based preference baseline
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor

Primary evaluation uses five-fold grouped cross-validation by user to reduce
leakage between the training and validation sets. Metrics are MAE, RMSE, and
R². A secondary grouped holdout is also saved. Because the dataset is small,
results are presented as an MVP evaluation rather than population-level proof.

## Setup and Training

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
```

For Windows Command Prompt (`cmd.exe`), activate with:

```bat
.venv\Scripts\activate.bat
```

Training creates cleaned datasets, model artifacts under `models/`, metrics
under `reports/`, qualitative Top-N examples, model feature importance, and
EDA/model-comparison figures under `reports/figures/`.

## Run the Application

```powershell
streamlit run app.py
```

This command requires the virtual environment to be activated. You can also
run it without activation:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the exact URL printed by Streamlit, normally
`http://127.0.0.1:8501`. The project binds to the IPv4 loopback explicitly to
avoid Windows `localhost` IPv4/IPv6 resolution issues.

The Streamlit app collects study preferences, predicts a suitability rating
for every place, ranks the Top-N results, and explains relevant matches and
tradeoffs in non-technical language.

## Tests

```powershell
pytest
```

## Deployment

For Streamlit Community Cloud:

1. Push the repository, including the trained `models/model_bundle.joblib`.
2. Create a Streamlit app linked to the repository.
3. Set the entry point to `app.py`.
4. Confirm dependencies install from `requirements.txt`.

The same app can be demonstrated locally or through another Python-compatible
hosting platform.

## Project Structure

```text
app.py                  Streamlit interface
src/data_validation.py Data cleaning and schema validation
src/features.py        Reusable compatibility feature engineering
src/train.py           Model comparison, training, and artifact generation
src/recommend.py       Candidate scoring and ranking
src/explain.py         Interpretable rule-based explanations
src/evaluate.py        Evaluation helpers and baselines
tests/                 Focused feature and recommendation tests
```

## Limitations

- The dataset has only 15 users and relatively few ratings per user.
- Preferences and place attributes use simplified ordinal scales.
- Place details and opening hours may change and require manual updates.
- Rule-based explanations summarize observable feature matches; they are not
  causal explanations of individual model predictions.
- Formal course submission still requires testing with at least five external
  users, analysis of their feedback in the PPT, and a recorded demo. A guide
  and data-entry template are included under `docs/` and `reports/`.
