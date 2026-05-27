# 🏆 AI Predicts the 2026 FIFA World Cup

An end-to-end, production-grade Machine Learning pipeline and Single Page Application (SPA) utilizing an **Ensemble Stacking Matrix** to predict match outcomes for the upcoming 2026 FIFA World Cup.

---

## 💻 Interactive Web Dashboard
The project features a built-in, modern Tailwind CSS web interface served directly via FastAPI. Users can interactively select any two qualified tournament teams via dynamic dropdown menus, trigger an asynchronous background prediction request, and view color-coded, animated outcome probability distributions (Win/Draw/Loss) in real-time without refreshing the page.

---

## 📊 Model Performance Architecture
To maximize accuracy and eliminate variance from a small dataset (~1,000 rows), the pipeline evolved from a standard baseline tree to a robust voting ensemble matrix:

| Pipeline Phase | Strategy | Evaluation Method | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baseline** | Default XGBoost | Single 80/20 Split | 62.50% |
| **Phase 2: Tuned** | Grid Search XGBoost | 5-Fold Cross-Validation | 65.80% |
| **Phase 3: Final** | **Soft-Voting Ensemble** | **5-Fold Stratified CV** | **66.00% (Peak: 72.50%)** |

### 🧠 The Ensemble Mindset
The final predictor uses a `VotingClassifier` (Soft Voting) combining three distinct algorithmic approaches:
1. **XGBoost Classifier** (Tuned: `max_depth=5`, `lr=0.01`, `n_estimators=100`) - Captures complex, non-linear feature interactions.
2. **Random Forest Classifier** - Mitigates overfitting through variance reduction and bagging noise control.
3. **Logistic Regression** (with `StandardScaler` pipeline) - Establishes a calibrated, robust linear baseline.

---

## 🧬 High-Impact Feature Engineering
Through domain-specific cross-domain interactions, engineered features significantly out-performed raw dataset columns:
* `form_vs_rank` (`recent_form_score` / `fifa_rank`): Emerged as the **#3 most critical feature (5.86% weight)**, successfully isolating dark horses on hot streaks.
* `value_density` (`market_value_million_eur` × `avg_player_rating`): Emerged as the **#4 most critical feature (5.42% weight)**.
* **Feature Pruning:** Automatically isolates and drops zero-importance noise columns (such as categorical confederation indicators) to streamline the model matrix from 33 to 30 clean dimensions.

---

## 🛠️ Project Structure
```text
├── app/
│   └── main.py                 # FastAPI server & responsive Tailwind frontend
├── data/
│   ├── raw/                    # Original raw Kaggle tournament records
│   └── processed/
│       ├── predictions_output.csv # Static pipeline simulation predictions
│       └── team_features_db.csv   # Pruned, standardized feature matrix for API lookup
├── models/
│   └── world_cup_ensemble.pkl  # Serialized, live VotingClassifier production binary
├── src/
│   ├── data_preprocessing.py   # Baseline cleaning and ingestion pipeline
│   └── train_model.py          # Pruning, 5-Fold Stratified CV, & joblib export routine
├── query_predictions.py        # CLI diagnostic lookup tool
└── requirements.txt            # Project application dependencies