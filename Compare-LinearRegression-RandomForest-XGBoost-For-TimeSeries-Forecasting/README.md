# 🚀 Compare Linear Regression, Random Forest, and XGBoost for Time Series Forecasting

### 🎯 Objective

Compare three ML algorithms for univariate time series forecasting (Daily Minimum Temperatures).

### 📊 Dataset

- Source: [Daily Minimum Temperatures, Melbourne (1981–1990)](https://drive.google.com/uc?id=1PWPwhW8QNOhPSOA1AtT7cu8-Uaxwo5UX)
- Samples: 3,650 daily records
- Target: Temperature (°C)

______________________________________________________________________

### 🧩 Methodology

1. **Feature Engineering**
   - Lags: 1–7–14–28 days
   - Rolling mean/std windows
   - Calendar features (day, month)
1. **Models**
   - Linear Regression (baseline)
   - Random Forest Regressor
   - XGBoost Regressor
1. **Validation**
   - Walk-forward (TimeSeriesSplit, 5 folds)
   - Metrics: MAE, RMSE, sMAPE, MASE

______________________________________________________________________

### 📈 Results Summary

| Model             | MAE      | RMSE     | sMAPE    | MASE     |
| ----------------- | -------- | -------- | -------- | -------- |
| Linear Regression | 1.23     | 1.65     | 9.5%     | 0.89     |
| Random Forest     | 1.05     | 1.42     | 8.1%     | 0.76     |
| XGBoost           | **0.98** | **1.36** | **7.9%** | **0.72** |

✅ **XGBoost** performed best (7.9% sMAPE, 0.72 MASE), improving ~20% vs baseline.

______________________________________________________________________

### 🖼️ Visual Results

| ![forecast](reports/figures/forecast_comparison.png) | ![feature_importance](reports/figures/feature_importance.png) |
| :--------------------------------------------------: | :-----------------------------------------------------------: |
|           Actual vs Predicted Temperatures           |                 Feature Importance (XGBoost)                  |

______________________________________________________________________

### 🧠 Insights

- Temperature exhibits clear weekly seasonality.
- Lag(1–7) + rolling mean(7) are most predictive features.
- Linear Regression underfits; XGB captures nonlinearities well.

______________________________________________________________________

### 💡 Tech Stack

`Python, pandas, scikit-learn, xgboost, matplotlib, seaborn`

______________________________________________________________________

### 🔍 Next Steps

- Multi-step forecasting (7-day ahead)
- Add Prophet / LSTM baseline
- Integrate SHAP explanations

______________________________________________________________________

## 🕒 Time Series Projects

- [Compare Linear Regression, Random Forest, and XGBoost for Time Series Forecasting](./TimeSeries-Project-Compare-LR-RF-XGB)
  - Focus: univariate forecasting
  - Methods: feature engineering, walk-forward validation, model comparison
  - Document: [Link](https://www.notion.so/So-s-nh-Linear-Regression-Random-Forest-v-XGBoost-trong-D-b-o-Time-Series-28c0730a9673805c9e1edb58640d13a7?v=2240730a967380c8a397000c5c7e4026&source=copy_link)

> _ℹ️ This project is migrate from source: [Link](https://github.com/mylethidiem/zero-to-hero/tree/main/data_ai_core/case_studies/time-series)_
