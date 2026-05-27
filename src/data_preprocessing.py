"""
Data loading and cleaning for the FIFA 2026 World Cup Predictor.

Reads train.csv and test.csv from data/raw/, cleans, and returns
feature matrices ready for modelling.

Columns
-------
Identifiers (dropped): team_name, country_code
Categorical (one-hot):  confederation  (UEFA, CONMEBOL, CAF, CONCACAF, AFC)
Numeric (standardised): all remaining feature columns
Target (train only):    winner  (1 = qualified/winner, 0 = not)
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

TARGET_COL = "winner"
ID_COLS = ["team_name", "country_code"]
CATEGORICAL_COLS = ["confederation"]

NUMERIC_COLS = [
    "fifa_rank",
    "fifa_points",
    "wins_last_10_matches",
    "losses_last_10_matches",
    "draws_last_10_matches",
    "win_rate_last_year",
    "goals_scored_avg",
    "goals_conceded_avg",
    "clean_sheets_last_10",
    "shots_per_game",
    "shots_on_target_ratio",
    "avg_player_rating",
    "star_players_count",
    "market_value_million_eur",
    "experience_avg_caps",
    "coach_experience_years",
    "recent_form_score",
    "possession_avg",
    "passing_accuracy",
    "host_advantage",
    "travel_distance_avg",
    "climate_similarity_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_raw() -> tuple:
    """Load CSVs as-is. Returns (train_df, test_df)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def report_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print shape, dtypes, missing values, and target distribution."""
    print(f"\n{'='*60}")
    print(f"{name}  shape={df.shape}")
    print(f"{'='*60}")
    print(df.dtypes.to_string())

    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print("\nNo missing values.")

    if TARGET_COL in df.columns:
        print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")


def clean(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple:
    """
    Clean and encode a dataframe (train features or test).

    Steps
    -----
    1. Drop identifier columns (team_name, country_code).
    2. Fill missing numeric values with column median.
    3. One-hot encode confederation.
    4. Standardise numeric features (fit on train, transform-only on test).

    Parameters
    ----------
    df    : feature dataframe (target column must already be removed)
    scaler: pre-fitted StandardScaler; required when fit=False
    fit   : True → fit+transform (train); False → transform-only (test)

    Returns
    -------
    (cleaned_df, scaler)
    """
    df = df.copy()

    # 1. Drop identifier columns
    df.drop(columns=[c for c in ID_COLS if c in df.columns], inplace=True)

    # 2. Fill missing numeric values with column median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 3. One-hot encode confederation
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # 4. Standardise numeric features
    num_present = [c for c in NUMERIC_COLS if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[num_present] = scaler.fit_transform(df[num_present])
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        df[num_present] = scaler.transform(df[num_present])

    return df, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_and_clean() -> dict:
    """
    Full preprocessing pipeline: load → report → clean.

    Returns
    -------
    dict with keys:
        X_train   : pd.DataFrame  cleaned training features
        y_train   : pd.Series     binary target (0/1)
        X_test    : pd.DataFrame  cleaned test features (same columns as X_train)
        scaler    : StandardScaler fitted on training numerics
        train_raw : pd.DataFrame  original train.csv
        test_raw  : pd.DataFrame  original test.csv
    """
    train_raw, test_raw = load_raw()

    report_summary(train_raw, "train.csv")
    report_summary(test_raw, "test.csv")

    y_train = train_raw[TARGET_COL].copy()
    train_features = train_raw.drop(columns=[TARGET_COL])

    X_train, scaler = clean(train_features, fit=True)
    X_test, _ = clean(test_raw, scaler=scaler, fit=False)

    # Align columns: test may lack some one-hot dummies if a confederation
    # is absent from the test split
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"\nX_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}  (class balance: {y_train.mean():.2%} positive)")
    print(f"X_test  : {X_test.shape}")
    print(f"\nFeature columns ({len(X_train.columns)}):\n{list(X_train.columns)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "scaler": scaler,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }


if __name__ == "__main__":
    data = load_and_clean()
    print("\nPreprocessing complete.")
