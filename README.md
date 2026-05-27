# 🏆 AI Predicts the 2026 FIFA World Cup

An end-to-end Machine Learning pipeline utilizing **XGBoost** to predict team success and match outcomes for the upcoming 2026 FIFA World Cup. 

## 📊 Model Performance (Baseline)
- **Algorithm:** XGBoost Classifier
- **Validation Accuracy:** 62.50%
- **F1-Score (Winner Class):** 0.61

## 🔮 Top Contenders (According to AI)
Based on historical form, FIFA rankings, and squad metrics, the model ranks these teams with the highest probability of winning their tournament matches:
1. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England
2. 🇦🇷 Argentina
3. 🇯🇵 Japan
4. 🇧🇷 Brazil
5. 🇺🇸 USA

## 🛠️ Project Structure
- `src/data_preprocessing.py`: Cleans and encodes Kaggle pipeline features.
- `src/train_model.py`: Trains XGBoost via an 80/20 stratified split and outputs predictions.
- `data/processed/predictions_output.csv`: Complete probabilistic tournament output.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/vap-dev07/fifa-2026-world-cup-predictor.git](https://github.com/vap-dev07/fifa-2026-world-cup-predictor.git)