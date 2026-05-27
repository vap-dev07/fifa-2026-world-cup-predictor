# 🏆 AI Predicts the 2026 FIFA World Cup

An end-to-end, production-grade Machine Learning pipeline utilizing an **Ensemble Stacking Matrix** to predict match outcomes for the upcoming 2026 FIFA World Cup.

## 📊 Model Performance Architecture
To maximize accuracy and eliminate variance from a small dataset, the pipeline evolves from a baseline tree to a robust voting ensemble:

| Pipeline Phase | Strategy | Evaluation Method | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baseline** | Default XGBoost | Single 80/20 Split | 62.50% |
| **Phase 2: Tuned** | Grid Search XGBoost | 5-Fold Cross-Validation | 65.80% |
| **Phase 3: Final** | **Soft-Voting Ensemble** | **5-Fold Cross-Validation** | **66.00% (Peak: 72.50%)** |

### 🧠 The Ensemble Mindset
The final predictor uses a `VotingClassifier` (Soft Voting) combining three distinct algorithmic approaches:
1. **XGBoost Classifier** (Tuned: `max_depth=5`, `lr=0.01`) - For complex non-linear splits.
2. **Random Forest Classifier** - For variance reduction and bagging noise control.
3. **Logistic Regression** (with `StandardScaler`) - For stable, calibrated linear baseline tracking.

## 🧬 High-Impact Feature Engineering
Through domain-specific cross-interactions, custom features significantly out-performed raw dataset columns:
- `form_vs_rank` (`recent_form_score` / `fifa_rank`): Finished as the **#3 most critical feature (5.86% weight)**, successfully isolating dark horses on hot streaks.
- `value_density` (`market_value` × `player_rating`): Finished as the **#4 most critical feature (5.42% weight)**.

## 🚀 Interactive Predictions Lookup
The project includes a command-line utility allowing users to scan the full database or lookup individual countries across simulated match contexts.

To explore the model's predictions:
```bash
python query_predictions.py

## 🛠️ Project Structure
- `src/data_preprocessing.py`: Cleans and encodes Kaggle pipeline features.
- `src/train_model.py`: Trains XGBoost via an 80/20 stratified split and outputs predictions.
- `data/processed/predictions_output.csv`: Complete probabilistic tournament output.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/vap-dev07/fifa-2026-world-cup-predictor.git](https://github.com/vap-dev07/fifa-2026-world-cup-predictor.git)