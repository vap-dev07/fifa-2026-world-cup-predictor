"""
Soft-voting ensemble pipeline for the FIFA 2026 World Cup Predictor.

Steps
-----
1. Load and clean data via data_preprocessing.load_and_clean()
2. Add 5 engineered features (33 total)
3. Fit a single XGBoost probe to rank feature importances; drop bottom 3 -> 30 features
4. Build a soft-voting ensemble: tuned XGBoost + RandomForest + scaled LogisticRegression
5. 5-fold stratified CV -> report per-fold and mean accuracy
6. Refit ensemble on the full pruned training set
7. Predict on test.csv and overwrite data/processed/predictions_output.csv
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_preprocessing import load_and_clean
from feature_engineering import add_features, ENGINEERED_COLS

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "predictions_output.csv"

RANDOM_STATE = 42

# XGBoost hyperparameters (best config from prior grid search on pruned features)
XGB_PARAMS = dict(
    max_depth=5,
    learning_rate=0.01,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Feature importance pruning
# ---------------------------------------------------------------------------


def prune_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, n: int = 3) -> tuple:
    """
    Fit a single XGBoost probe, print importances, and drop the n least useful columns.

    Returns (X_train_pruned, X_test_pruned, dropped_cols).
    """
    probe = xgb.XGBClassifier(**XGB_PARAMS)
    probe.fit(X_train, y_train)

    importances = pd.Series(probe.feature_importances_, index=X_train.columns)
    bottom_n = importances.nsmallest(n).index.tolist()

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCES  (XGBoost probe, ascending)")
    print(f"{'='*60}")
    print(importances.sort_values().to_string())
    print(f"\nDropping bottom {n}: {bottom_n}")

    return X_train.drop(columns=bottom_n), X_test.drop(columns=bottom_n), bottom_n


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def build_ensemble() -> VotingClassifier:
    """
    Soft-voting ensemble of three diverse classifiers.

    - XGBoost   : tuned tree boosting (best params from prior CV)
    - RandomForest : bagged trees, orthogonal to XGB's boosting
    - LogisticRegression : linear baseline; wrapped in a Pipeline so its
      own StandardScaler handles the mix of already-scaled numeric columns
      and raw engineered features
    """
    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE)),
    ])

    return VotingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf), ("lr", lr_clf)],
        voting="soft",
    )


def run_cv(ensemble: VotingClassifier, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """5-fold stratified CV; prints per-fold scores and mean. Returns scores array."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print(f"\n{'='*60}")
    print("5-FOLD CV RESULTS  (soft-voting ensemble, pruned 30 features)")
    print(f"{'='*60}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i} : {s:.4f}  ({s * 100:.2f}%)")
    print(f"  {'-'*30}")
    print(f"  Mean  : {scores.mean():.4f}  ({scores.mean() * 100:.2f}%)")
    print(f"  Std   : +-{scores.std():.4f}")

    return scores


# ---------------------------------------------------------------------------
# Prediction and output
# ---------------------------------------------------------------------------


def predict_and_save(model, X_test: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions on the test set and write the output CSV."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    output = pd.DataFrame({
        "team_name":          test_raw["team_name"].values,
        "country_code":       test_raw["country_code"].values,
        "confederation":      test_raw["confederation"].values,
        "winner_probability": np.round(proba, 6),
        "predicted_winner":   pred,
    }).sort_values("winner_probability", ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"PREDICTIONS SAVED -> {OUTPUT_PATH}")
    print(f"{'='*60}")
    print(f"Teams predicted as winners : {pred.sum()}")
    print(f"\nTop 10 predicted teams by win probability:")
    print(output.head(10).to_string(index=False))

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data = load_and_clean()

    train_eng = add_features(data["train_raw"])[ENGINEERED_COLS].reset_index(drop=True)
    test_eng  = add_features(data["test_raw"])[ENGINEERED_COLS].reset_index(drop=True)

    X_train = pd.concat([data["X_train"].reset_index(drop=True), train_eng], axis=1)
    X_test  = pd.concat([data["X_test"].reset_index(drop=True), test_eng],  axis=1)

    print(f"\nEngineered features added : {ENGINEERED_COLS}")
    print(f"Total features            : {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # Step 1: Probe XGBoost -> drop 3 least useful features
    # ------------------------------------------------------------------
    print("\n[Step 1] Fitting XGBoost probe for feature importance ranking ...")
    X_train_p, X_test_p, dropped = prune_features(X_train, X_test, data["y_train"], n=3)
    print(f"Pruned feature count      : {X_train_p.shape[1]}")

    # ------------------------------------------------------------------
    # Step 2: 5-fold CV on the pruned ensemble
    # ------------------------------------------------------------------
    print("\n[Step 2] Running 5-fold CV on soft-voting ensemble ...")
    ensemble = build_ensemble()
    scores = run_cv(ensemble, X_train_p, data["y_train"])

    # ------------------------------------------------------------------
    # Step 3: Refit on all training data, predict, save
    # ------------------------------------------------------------------
    print("\n[Step 3] Refitting ensemble on full training set ...")
    ensemble.fit(X_train_p, data["y_train"])

    train_pred = ensemble.predict(X_train_p)
    print(f"\nClassification report on full training set (sanity check):")
    print(classification_report(data["y_train"], train_pred, target_names=["Not Winner", "Winner"]))

    predict_and_save(ensemble, X_test_p, data["test_raw"])

    print("\nDone.")


if __name__ == "__main__":
    main()
