# 🏆 AI Predicts the 2026 FIFA World Cup

An end-to-end Machine Learning pipeline utilizing an optimized **XGBoost Classifier** to predict team success and match outcomes for the upcoming 2026 FIFA World Cup.

## 📊 Model Performance (Optimized)
- **Algorithm:** XGBoost Classifier
- **Evaluation Strategy:** 5-Fold Stratified Cross-Validation
- **Validation Accuracy:** 66.20% (📈 +3.70% over baseline)
- **Best Hyperparameters:** `max_depth=5`, `learning_rate=0.01`, `n_estimators=100`

## 🔮 Model Calibration Insights
Through hyperparameter tuning, the model's prediction confidence has been realisticly calibrated. Top match win probabilities now scale dynamically within a realistic `0.69–0.72` band rather than overestimating outcomes at `0.99`.

## 🔮 Top Contenders (According to AI)
Based on historical form, FIFA rankings, and squad metrics, the model ranks these teams with the highest probability of winning their tournament matches:
1. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England
2. 🇦🇷 Argentina
3. 🇯🇵 Japan
4. 🇧🇷 Brazil
5. 🇺🇸 USA


## 🔍 Interactive Predictions Lookup

You can query individual team metrics and see their calibrated win probabilities across different simulation contexts directly from the terminal.

To run the interactive lookup:
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