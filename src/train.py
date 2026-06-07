from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_validation import (
    PROJECT_ROOT,
    build_training_table,
    load_clean_datasets,
    save_clean_datasets,
)
from src.evaluate import (
    evaluate_mean_baseline_grouped,
    evaluate_pipeline_grouped,
    regression_metrics,
    rule_based_predictions,
    save_json,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    add_compatibility_features,
    select_model_features,
)
from src.recommend import recommend_places


RANDOM_STATE = 42
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def model_candidates() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=10, weights="distance"),
    }


def build_model_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("model", estimator)]
    )


def choose_final_model(metrics: dict[str, dict[str, float]]) -> str:
    learned_models = {
        name: values
        for name, values in metrics.items()
        if name not in {"Mean Rating Baseline", "Rule-Based Baseline"}
    }
    lowest_rmse = min(values["rmse"] for values in learned_models.values())
    forest_rmse = learned_models["Random Forest Regressor"]["rmse"]
    if forest_rmse <= lowest_rmse * 1.05:
        return "Random Forest Regressor"
    return min(learned_models, key=lambda name: learned_models[name]["rmse"])


def create_eda_figures(
    training_table: pd.DataFrame,
    items: pd.DataFrame,
    metrics_frame: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=training_table, x="rating", color="#2f7f73")
    plt.title("Distribution of Valid Suitability Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Interactions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_distribution.png", dpi=160)
    plt.close()

    plot_frame = metrics_frame.sort_values("rmse")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_frame, x="rmse", y="model", color="#3a86c8")
    plt.title("Grouped Cross-Validation RMSE by Model")
    plt.xlabel("RMSE (lower is better)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_rmse_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    area_order = items["area"].value_counts().index
    sns.countplot(
        data=items, y="area", order=area_order, color="#d8894b"
    )
    plt.title("Study Places by Area")
    plt.xlabel("Number of places")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "places_by_area.png", dpi=160)
    plt.close()

    place_ratings = (
        training_table.groupby("place_name", as_index=False)
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "size"))
        .query("rating_count >= 3")
        .sort_values(["mean_rating", "rating_count"], ascending=False)
        .head(10)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=place_ratings,
        x="mean_rating",
        y="place_name",
        color="#6f74b8",
    )
    plt.xlim(1, 5)
    plt.title("Highest-Rated Places with at Least 3 Ratings")
    plt.xlabel("Mean observed rating")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_observed_place_ratings.png", dpi=160)
    plt.close()


def save_model_inspection(
    model: object, preprocessor: ColumnTransformer
) -> None:
    transformed_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        value_name = "importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        value_name = "coefficient"
    else:
        return

    inspection = pd.DataFrame(
        {"transformed_feature": transformed_names, value_name: values}
    )
    inspection["absolute_value"] = inspection[value_name].abs()
    inspection = inspection.sort_values("absolute_value", ascending=False)
    inspection.to_csv(
        REPORTS_DIR / "model_feature_importance.csv", index=False
    )


def save_qualitative_recommendations(
    personas: pd.DataFrame,
    items: pd.DataFrame,
    bundle: dict,
) -> None:
    rows: list[pd.DataFrame] = []
    for _, persona in personas.iterrows():
        recommendations = recommend_places(
            persona.to_dict(), items, bundle, top_n=5
        )
        summary = recommendations[
            [
                "rank",
                "place_id",
                "place_name",
                "area",
                "predicted_rating",
                "average_price",
                "distance_to_binus_km",
                "explanation",
            ]
        ].copy()
        summary.insert(0, "user_id", persona["user_id"])
        rows.append(summary)

    pd.concat(rows, ignore_index=True).to_csv(
        REPORTS_DIR / "qualitative_top_n_recommendations.csv", index=False
    )


def train_and_save() -> dict:
    personas, ratings, items = load_clean_datasets()
    training_table = build_training_table(personas, ratings, items)
    training_table = add_compatibility_features(training_table)
    save_clean_datasets(personas, items, training_table)

    features = select_model_features(training_table)
    target = training_table["rating"].reset_index(drop=True)
    groups = training_table["user_id"].reset_index(drop=True)
    features = features.reset_index(drop=True)

    all_metrics: dict[str, dict[str, float]] = {}
    fold_results: list[pd.DataFrame] = []

    mean_metrics, mean_folds = evaluate_mean_baseline_grouped(target, groups)
    all_metrics["Mean Rating Baseline"] = mean_metrics
    mean_folds.insert(0, "model", "Mean Rating Baseline")
    fold_results.append(mean_folds)

    rule_predictions = rule_based_predictions(features)
    all_metrics["Rule-Based Baseline"] = regression_metrics(
        target, rule_predictions
    )

    candidates = model_candidates()
    for name, estimator in candidates.items():
        metrics, folds = evaluate_pipeline_grouped(
            build_model_pipeline(estimator), features, target, groups
        )
        all_metrics[name] = metrics
        folds.insert(0, "model", name)
        fold_results.append(folds)

    selected_name = choose_final_model(all_metrics)
    selected_pipeline = build_model_pipeline(candidates[selected_name])

    secondary_split = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )
    train_index, test_index = next(
        secondary_split.split(features, target, groups)
    )
    secondary_pipeline = build_model_pipeline(candidates[selected_name])
    secondary_pipeline.fit(features.iloc[train_index], target.iloc[train_index])
    secondary_predictions = secondary_pipeline.predict(features.iloc[test_index])
    secondary_metrics = regression_metrics(
        target.iloc[test_index], secondary_predictions
    )
    held_out_users = sorted(groups.iloc[test_index].unique().tolist())

    selected_pipeline.fit(features, target)
    preprocessor = selected_pipeline.named_steps["preprocessor"]
    model = selected_pipeline.named_steps["model"]

    metadata = {
        "project_name": "StudySpot",
        "model_type": selected_name,
        "random_state": RANDOM_STATE,
        "training_interactions": int(len(training_table)),
        "unique_users": int(training_table["user_id"].nunique()),
        "unique_places": int(items["place_id"].nunique()),
        "validation_strategy": "5-fold GroupKFold grouped by user",
        "rating_scale": [1, 5],
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_name": "rating",
        "metadata": metadata,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(bundle, MODELS_DIR / "model_bundle.joblib")
    save_model_inspection(model, preprocessor)
    save_qualitative_recommendations(personas, items, bundle)

    metrics_payload = {
        "selected_model": selected_name,
        "selection_rule": (
            "Prefer Random Forest when its grouped RMSE is within 5% of the "
            "best learned model; otherwise select the lowest grouped RMSE."
        ),
        "grouped_cross_validation": all_metrics,
        "secondary_group_holdout": {
            "held_out_users": held_out_users,
            **secondary_metrics,
        },
        "dataset": {
            "raw_rating_rows": 180,
            "valid_training_interactions": int(len(training_table)),
            "excluded_blank_ratings": int(180 - len(training_table)),
        },
        "limitations": [
            "The dataset contains only 15 users and limited ratings per user.",
            "Metrics describe an MVP recommender and should not be generalized "
            "to a large population.",
            "Rule-based explanations describe feature matches and tradeoffs; "
            "they are not causal explanations of the model.",
        ],
    }
    save_json(metrics_payload, MODELS_DIR / "metrics.json")
    save_json(
        {
            "all_features": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_identifiers": ["user_id", "place_id", "place_name"],
        },
        MODELS_DIR / "feature_columns.json",
    )

    metrics_rows = [
        {"model": name, **values} for name, values in all_metrics.items()
    ]
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    pd.concat(fold_results, ignore_index=True).to_csv(
        REPORTS_DIR / "grouped_cv_fold_metrics.csv", index=False
    )

    holdout_results = training_table.iloc[test_index][
        ["user_id", "place_id", "place_name", "rating"]
    ].copy()
    holdout_results["predicted_rating"] = np.clip(
        secondary_predictions, 1.0, 5.0
    )
    holdout_results["absolute_error"] = (
        holdout_results["rating"] - holdout_results["predicted_rating"]
    ).abs()
    holdout_results.to_csv(
        REPORTS_DIR / "predicted_vs_actual_holdout.csv", index=False
    )
    create_eda_figures(training_table, items, metrics_frame)

    return metrics_payload


def main() -> None:
    metrics = train_and_save()
    selected = metrics["selected_model"]
    grouped = metrics["grouped_cross_validation"][selected]
    print(f"StudySpot training complete. Selected model: {selected}")
    print(
        f"Grouped CV - MAE: {grouped['mae']:.3f}, "
        f"RMSE: {grouped['rmse']:.3f}, R2: {grouped['r2']:.3f}"
    )
    print(f"Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
