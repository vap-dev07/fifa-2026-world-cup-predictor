"""
Feature engineering for the FIFA 2026 World Cup Predictor.

Computes derived features from raw team statistics.
"""

import pandas as pd

EPSILON = 0.001
ENGINEERED_COLS = ["goal_efficiency", "dominance_index", "squad_value_per_star"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a raw team-stats DataFrame.

    New columns
    -----------
    goal_efficiency      : goals_scored_avg / (goals_conceded_avg + EPSILON)
    dominance_index      : shots_per_game * possession_avg
    squad_value_per_star : market_value_million_eur / (star_players_count + 1)
    """
    df = df.copy()
    df["goal_efficiency"] = df["goals_scored_avg"] / (df["goals_conceded_avg"] + EPSILON)
    df["dominance_index"] = df["shots_per_game"] * df["possession_avg"]
    df["squad_value_per_star"] = df["market_value_million_eur"] / (df["star_players_count"] + 1)
    return df
