# 🚀 AI Multi-Horizon Stock Prediction Platform

This project is an advanced stock prediction platform that leverages a hybrid AI approach to forecast stock price movements across multiple time horizons. It combines comprehensive technical analysis with real-time news sentiment to provide predictions for the **Next Day**, **One Week**, and **One Month**.

The platform is built with a Streamlit dashboard for interactive analysis and uses a sophisticated ensemble of machine learning models (LightGBM, XGBoost, Random Forest) alongside a deep learning (LSTM) model for robust forecasting.

## 📸 Dashboard Preview

![](https://drive.google.com/drive/u/1/folders/1xVpMevHLdZroGOinJRKSnh_0uwhcER7J)



---

## 🎯 Key Features

* **Multi-Horizon Forecasting:** Predicts stock direction (Up/Down) for three distinct timeframes:
    * 📅 **Next Day** (1 trading day)
    * 📊 **One Week** (5 trading days)
    * 📈 **One Month** (21 trading days)
* **Hybrid AI Model:** Uses a sophisticated ensemble of models for each horizon:
    * **Ensemble Models:** LightGBM, XGBoost, and Random Forest.
    * **Deep Learning:** A Long Short-Term Memory (LSTM) network for sequence analysis.
* **News Sentiment Analysis:**
    * Fetches real-time news headlines from **The Guardian API**.
    * Analyzes sentiment using **NLTK's VADER** to quantify market mood.
* **Rich Feature Engineering:**
    * Generates over 50 technical indicators, including RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic Oscillator, and various moving averages.
    * Includes price momentum, volatility, and lagged features.
* **Interactive Streamlit Dashboard:**
    * Allows users to select from a list of stocks.
    * Provides interactive charts, sentiment analysis, and multi-horizon predictions.
    * Calculates a final "Trading Signal" (e.g., Strong Buy, Sell, Hold) based on model consensus.
* **Data-Driven Training:**
    * **Handles Class Imbalance:** Uses **SMOTE** (oversampling) and **RandomUnderSampler** (undersampling) for a more balanced training dataset.
    * **Finds Optimal Thresholds:** Calculates the best classification threshold for each model to maximize F1-score, rather than using a default 0.5.

---

## 🛠️ Technology Stack

* **Data Science & ML/DL:**
    * `pandas` & `numpy` (Data manipulation)
    * `scikit-learn` (Scaling, metrics, traditional ML)
    * `tensorflow` / `keras` (LSTM model)
    * `lightgbm`, `xgboost` (Gradient boosting models)
    * `imblearn` (Handling imbalanced data)
* **Web Dashboard:**
    * `streamlit` (Interactive web app)
    * `plotly` (Interactive charting)
* **Data Collection:**
    * `yfinance` (Historical stock data)
    * `requests` (Guardian API)
* **NLP:**
    * `nltk` (VADER sentiment analysis)
* **Utilities:**
    * `joblib` (Saving/loading ML models)

---

## 🏗️ Project Architecture

The project is divided into three main components:

1.  **`data_collection.py`**:
    * Fetches historical stock data (OHLCV) from **Yahoo Finance**.
    * Fetches historical news headlines from **The Guardian API**.
    * Saves the raw data into the `../data` directory as CSV files.

2.  **`training_script.py`**:
    * Loads the raw stock and news data.
    * Performs sentiment analysis on news and merges it with stock data.
    * Engineers all technical indicators and features.
    * For **each** prediction horizon (next_day, one_week, one_month):
        * Creates the target variable (`Price_Up`).
        * Applies resampling (SMOTE/RandomUnderSampler).
        * Trains and saves an **Ensemble** (LGBM, XGB, RF) and an **LSTM** model.
        * Saves the scaler, feature names, and optimal thresholds.
    * Saves all artifacts (models, scalers) to the `../saved_models` directory.

3.  **`app.py`**:
    * A Streamlit web application.
    * Fetches *live* data from Yahoo Finance and The Guardian.
    * Loads the pre-trained models, scalers, and features for the selected stock.
    * Takes user input for "tomorrow's expected values" to generate the final feature row.
    * Generates predictions for all three horizons using their respective models.
    * Displays the results, charts, and sentiment in an interactive dashboard.

4.  **`.streamlit/secret.toml`**:
    * Paste your The_guadian_API here
    * GUARDIAN_API_KEY='your guradian api key'
     

---

## 🔧 Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/rishiyadav-DA/DS-project]

cd rishiyadav-DA/DS-project
Link to Demo:-
https://rishiyadav-da-ds-project-app-cxjd8x.streamlit.app/


