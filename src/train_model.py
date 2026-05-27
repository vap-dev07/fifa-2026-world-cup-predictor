"""
XGBoost training pipeline for the FIFA 2026 World Cup Predictor.

Steps
-----
1. Load and clean data via data_preprocessing.load_and_clean()
2. 80/20 stratified train/validation split
3. Train XGBClassifier
4. Print accuracy + classification report on the validation set
5. Predict on test.csv and write data/processed/predictions_output.csv
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Allow importing sibling module when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_preprocessing import load_and_clean

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "predictions_output.csv"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
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
# Training and evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(X_train, y_train):
    """
    Split into train/val (80/20), train XGBoost, report metrics.

    Returns (fitted_model, X_val, y_val).
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"\nTrain split : {X_tr.shape[0]} rows")
    print(f"Val split   : {X_val.shape[0]} rows")

    model = build_model()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, val_pred)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=["Not Winner", "Winner"]))

    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Prediction and output
# ---------------------------------------------------------------------------


def predict_and_save(model, X_test, test_raw: pd.DataFrame) -> pd.DataFrame:
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
        "team_name": test_raw["team_name"].values,
        "country_code": test_raw["country_code"].values,
        "confederation": test_raw["confederation"].values,
        "winner_probability": np.round(proba, 6),
        "predicted_winner": pred,
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

    model, _, _ = train_and_evaluate(data["X_train"], data["y_train"])

    predict_and_save(model, data["X_test"], data["test_raw"])

    print("\nDone.")


if __name__ == "__main__":
    main()
