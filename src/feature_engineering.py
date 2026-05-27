"""
Feature engineering for the FIFA 2026 World Cup Predictor.

Computes derived features from raw team statistics.
"""

import pandas as pd

EPSILON = 0.001
ENGINEERED_COLS = [
    "goal_efficiency",
    "dominance_index",
    "squad_value_per_star",
    "form_vs_rank",
    "value_density",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a raw team-stats DataFrame.

    New columns
    -----------
    goal_efficiency      : goals_scored_avg / (goals_conceded_avg + EPSILON)
    dominance_index      : shots_per_game * possession_avg
    squad_value_per_star : market_value_million_eur / (star_players_count + 1)
    form_vs_rank         : recent_form_score / (fifa_rank + EPSILON)  — high form relative to low rank
    value_density        : market_value_million_eur * avg_player_rating  — squad wealth × quality
    """
    df = df.copy()
    df["goal_efficiency"] = df["goals_scored_avg"] / (df["goals_conceded_avg"] + EPSILON)
    df["dominance_index"] = df["shots_per_game"] * df["possession_avg"]
    df["squad_value_per_star"] = df["market_value_million_eur"] / (df["star_players_count"] + 1)
    df["form_vs_rank"] = df["recent_form_score"] / (df["fifa_rank"] + EPSILON)
    df["value_density"] = df["market_value_million_eur"] * df["avg_player_rating"]
    return df
