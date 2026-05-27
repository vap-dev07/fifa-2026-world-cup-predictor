"""
XGBoost training pipeline for the FIFA 2026 World Cup Predictor.

Steps
-----
1. Load and clean data via data_preprocessing.load_and_clean()
2. Add engineered features (goal_efficiency, dominance_index, squad_value_per_star)
3. 5-fold stratified cross-validation grid search over max_depth,
   learning_rate, and n_estimators (18 combinations total)
4. Report per-combination CV accuracy and highlight best params
5. Retrain best configuration on the full training set
6. Predict on test.csv and overwrite data/processed/predictions_output.csv
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_preprocessing import load_and_clean
from feature_engineering import add_features, ENGINEERED_COLS

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "predictions_output.csv"

RANDOM_STATE = 42

PARAM_GRID = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators":  [100, 200],
}

# Hyperparameters held fixed across all grid combinations
FIXED_PARAMS = dict(
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
# Grid search
# ---------------------------------------------------------------------------


def run_grid_search(X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """
    5-fold stratified CV grid search over PARAM_GRID.

    Returns the fitted GridSearchCV object; best_estimator_ is already
    retrained on the full dataset (refit=True by default).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_model = xgb.XGBClassifier(**FIXED_PARAMS)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        refit=True,       # retrain best config on all data automatically
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search


def print_grid_results(search: GridSearchCV) -> None:
    """Print a sorted table of all CV results and highlight the winner."""
    results = pd.DataFrame(search.cv_results_)
    cols = ["param_max_depth", "param_learning_rate", "param_n_estimators",
            "mean_test_score", "std_test_score", "rank_test_score"]
    results = (
        results[cols]
        .rename(columns={
            "param_max_depth":     "max_depth",
            "param_learning_rate": "lr",
            "param_n_estimators":  "n_est",
            "mean_test_score":     "cv_acc",
            "std_test_score":      "cv_std",
            "rank_test_score":     "rank",
        })
        .sort_values("rank")
        .reset_index(drop=True)
    )
    results["cv_acc"] = results["cv_acc"].map(lambda x: f"{x:.4f}")
    results["cv_std"] = results["cv_std"].map(lambda x: f"±{x:.4f}")

    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS  (5-fold stratified CV, sorted by rank)")
    print(f"{'='*60}")
    print(results.to_string(index=False))

    bp = search.best_params_
    bs = search.best_score_
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"  max_depth     : {bp['max_depth']}")
    print(f"  learning_rate : {bp['learning_rate']}")
    print(f"  n_estimators  : {bp['n_estimators']}")
    print(f"  CV accuracy   : {bs:.4f}  ({bs*100:.2f}%)")


# ---------------------------------------------------------------------------
# Prediction and output
# ---------------------------------------------------------------------------


def predict_and_save(model, X_test: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions on the test set and write the output CSV.

    Output columns
    --------------
    team_name, country_code, confederation,
    winner_probability, predicted_winner
    """
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
    print(f"\nRunning grid search — {len(PARAM_GRID['max_depth']) * len(PARAM_GRID['learning_rate']) * len(PARAM_GRID['n_estimators'])} combinations × 5 folds …")

    search = run_grid_search(X_train, data["y_train"])
    print_grid_results(search)

    # best_estimator_ is already fitted on all training data (refit=True)
    best_model = search.best_estimator_

    # Show full classification report on the training set as a sanity check
    train_pred = best_model.predict(X_train)
    print(f"\nClassification report on full training set (sanity check):")
    print(classification_report(data["y_train"], train_pred, target_names=["Not Winner", "Winner"]))

    predict_and_save(best_model, X_test, data["test_raw"])

    print("\nDone.")


if __name__ == "__main__":
    main()
