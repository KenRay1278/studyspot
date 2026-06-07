from __future__ import annotations

import pandas as pd


def explain_recommendation(row: pd.Series) -> list[str]:
    reasons: list[str] = []

    if row["distance_gap"] <= 0:
        reasons.append("It is within your preferred travel distance.")
    else:
        reasons.append(
            f"It is {row['distance_gap']:.1f} km beyond your preferred distance."
        )

    if row["price_gap"] <= 0:
        reasons.append("Its average price is within your spending comfort.")
    else:
        reasons.append(
            "It is more expensive than your usual spending comfort, "
            "so consider the facility tradeoff."
        )

    if row["wifi_gap"] >= 0:
        reasons.append("Its Wi-Fi quality meets your stated importance level.")
    else:
        reasons.append("Its Wi-Fi may be below your preferred level.")

    if row["outlet_gap"] >= 0:
        reasons.append("Power outlet availability matches your preference.")
    elif row["outlet_importance"] >= 4:
        reasons.append("Power outlets may be limited for your stated preference.")

    if row["table_capacity_gap"] >= 0:
        reasons.append("Its table capacity can accommodate your usual group.")
    else:
        reasons.append("Its table capacity may be tight for your usual group.")

    if row["noise_gap"] > 0:
        reasons.append("It may be noisier than your stated tolerance.")
    if row["crowd_gap"] > 0:
        reasons.append("It may feel more crowded than you prefer.")

    if row["study_on_sundays_numeric"] > 0 and row["open_on_sundays"] == 1:
        reasons.append("It is open on Sundays, matching your study habit.")
    elif row["study_on_sundays_numeric"] == 1 and row["open_on_sundays"] == 0:
        reasons.append("It is not open on Sundays.")

    if row["preferred_time_match"] == 1:
        reasons.append("Its opening hours cover your preferred study time.")
    else:
        reasons.append("Its opening hours may not cover your preferred study time.")

    return reasons[:5]


def explanation_text(row: pd.Series) -> str:
    return " ".join(explain_recommendation(row))

