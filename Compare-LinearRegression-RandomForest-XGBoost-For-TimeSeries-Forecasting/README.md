# ⏱️ Compare Linear Regression, Random Forest, and XGBoost for Time Series Forecasting

## 🎯 Objective
This project compares **Linear Regression**, **Random Forest**, and **XGBoost** for univariate time series forecasting.
Dataset: *Daily Minimum Temperatures in Melbourne (1981–1990)*.

---

## ⚙️ Workflow
1. **Data Preparation**
   - Load & clean dataset
   - Create lag, rolling mean, and date-based features
2. **Modeling**
   - Train LR, RF, XGB using `TimeSeriesSplit`
   - Optional: Hyperparameter tuning
3. **Evaluation**
   - Metrics: `MAE`, `RMSE`, `sMAPE`, `MASE`
   - Compare across models with visual plots
4. **Visualization (optional)**
   - Gradio app to visualize predictions interactively

---

## 📊 Results Example
| Model | MAE | RMSE | sMAPE | MASE |
|--------|-----|------|-------|------|
| Linear Regression | 1.23 | 1.65 | 9.5% | 0.89 |
| Random Forest | 1.05 | 1.42 | 8.1% | 0.76 |
| XGBoost | **0.98** | **1.36** | **7.9%** | **0.72** |

---

## ▶️ How to Run
<!-- ```bash
pip install -r requirements.txt
python run_experiment.py --model xgb --lags 7 --n_splits 5
``` -->
🪄 Features
- Lag & rolling window feature engineering
- Walk-forward validation
- Model performance comparison
- Optional XAI: SHAP (XGB), feature importance (RF)

🧠 Next Steps
- Multi-step forecasting
- Add Prophet / LSTM baseline
- Integrate MLflow for experiment tracking

## 🕒 Time Series Projects
- [Compare Linear Regression, Random Forest, and XGBoost for Time Series Forecasting](./TimeSeries-Project-Compare-LR-RF-XGB)
  - Focus: univariate forecasting
  - Methods: feature engineering, walk-forward validation, model comparison

>_ℹ️ This project is migrate from source: [Link](https://github.com/mylethidiem/zero-to-hero/tree/main/data_ai_core/case_studies/time-series)_