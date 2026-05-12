# Stock Market Prediction Using Machine Learning

Comparative analysis of supervised, unsupervised, and ensemble methods applied to S&P 500 stock data — with an interactive HTML dashboard output.

---

## Overview

This project implements and compares **7 machine learning models** across three problem types:

| Task | Type | Target |
|------|------|--------|
| Return Prediction | Supervised — Regression | Next-day % price change |
| Direction Prediction | Supervised — Classification | Up (1) / Down (0) |
| Stock Grouping | Unsupervised — Clustering | Risk-return profiles |

---

## Models Implemented

**Regression**
- Linear Regression with Ridge regularization (α=1.0)
- Decision Tree Regressor (max_depth=15)
- Random Forest Regressor (300 estimators)
- Gradient Boosting Regressor (200 estimators, lr=0.05)

**Classification**
- Decision Tree Classifier (max_depth=10)
- Random Forest Classifier (100 estimators)

**Unsupervised**
- K-Means Clustering (k=3, on annualized return + volatility)

---

## Results Summary

### Regression (target = next-day return)

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Linear Regression (Ridge) | -0.098 | 0.0076 | 0.0109 |
| Decision Tree | -0.755 | 0.0091 | 0.0137 |
| **Random Forest** | **-0.044** | **0.0074** | **0.0106** |
| Gradient Boosting | -0.148 | 0.0080 | 0.0111 |

> All R² values are negative — expected for next-day return prediction and consistent with the efficient market hypothesis. Random Forest achieves the best (least negative) score.

### Classification (target = price direction)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Decision Tree** | **50.4%** | 56.5% | 44.5% | **0.498** |
| Random Forest | 50.0% | 63.8% | 21.9% | 0.326 |

> Both models perform near the 50% random baseline, reflecting the difficulty of direction prediction.

### Clustering

| Metric | Value |
|--------|-------|
| Silhouette Score | **0.521** |
| Clusters | 3 (Low / Medium / High risk) |
| Stocks Analysed | 14 major S&P 500 tickers |

---

## Dataset

**S&P 500 Historical Data — `all_stocks_5yr.csv`**  
~500 tickers · ~1,259 trading days each · daily OHLCV format

[Download from Google Drive](https://drive.google.com/file/d/1qrf7cU6mMDzsL6pXPw-jpz6MmaTWfWYg/view?usp=sharing)

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Trading date |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `close` | float | Closing price |
| `volume` | int | Shares traded |
| `ticker` | string | Stock symbol |

---

## Feature Engineering

30 technical features were derived from raw OHLCV data:

- **Moving averages** — SMA (5, 20), EMA (10, 20)
- **Bollinger Bands** — width, % position within bands
- **Volatility** — rolling std over 5 and 20 days, ATR-14
- **Momentum** — RSI-14, MACD + signal + histogram, 5/10/20-day momentum
- **Volume** — volume ratio vs 10-day MA, OBV, lagged volume
- **Lag features** — return lags (1, 2, 3, 5 days), H-L range lags
- **Price ratios** — close/SMA5, close/SMA20, close/EMA10 (scale-free, no leakage)

> Raw price lags (`close_lag1` etc.) were intentionally excluded to avoid data leakage — `close_lag1 ≈ today's close` would let models trivially reconstruct the return denominator.

---

## Project Structure

```
├── stock_ml_dashboard.py     # Main pipeline (run in Google Colab)
├── stock_ml_dashboard.html   # Generated interactive dashboard (output)
├── README.md
└── report/
    └── CS245_ML_Project_Report.pdf
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `all_stocks_5yr.csv` to `/content/`
3. Upload and run `stock_ml_dashboard.py`
4. Enter any S&P 500 ticker when prompted (e.g. `AAPL`, `MSFT`, `GOOGL`)
5. The dashboard renders inline in Colab and is saved to `/content/stock_ml_dashboard.html`

```bash
# Install dependencies (auto-handled in script)
pip install scikit-learn pandas numpy
```

---

## Interactive Dashboard

The output is a **self-contained HTML file** — no Python or server required, opens in any browser.

**7 tabbed panels, one per model:**
- Metric cards (R², MAE, RMSE, Accuracy, F1, etc.)
- Actual vs predicted scatter plots
- Residuals distribution histogram
- Top feature importances (Random Forest)
- Confusion matrices (classifiers)
- PCA cluster scatter + Return vs Volatility bubble chart
- Strengths & weaknesses analysis per model

**Tech stack:** Chart.js 4.4.1 · HTML/CSS/JS · Dark theme (Syne + DM Mono)

---

## Key Findings

- **Stock return prediction is hard.** Negative R² across all models is expected and not a bug — it reflects the near-random nature of daily returns (consistent with the semi-strong EMH).
- **Ensembles beat single models.** Random Forest (-0.044) significantly outperforms Decision Tree (-0.755) by reducing variance through averaging.
- **Direction prediction barely beats random.** 50.4% accuracy confirms historical price patterns have very limited predictive power for next-day direction.
- **Clustering reveals real structure.** Silhouette score of 0.521 confirms stocks naturally group into distinct low/medium/high risk-return profiles, useful for portfolio diversification.
- **Most important features:** `volatility_5`, `volume_lag1`, momentum indicators, `BB_pct`, `RSI_14`.

---


