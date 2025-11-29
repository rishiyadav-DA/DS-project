import os
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import warnings
import time

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Multi-Horizon Stock Prediction",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #667eea44;
    }
    .prediction-card {
        border: 3px solid;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .horizon-card {
        border: 2px solid;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        background: linear-gradient(135deg, #1e3c7222 0%, #2a529844 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .insight-box {
        background: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .horizon-tab {
        font-size: 16px;
        font-weight: bold;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Constants ---
COMPANIES = {
    "Tata Motors": "TATAMOTORS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon.com, Inc.": "AMZN",
    "Meta Platforms, Inc.": "META",
    "Tesla, Inc.": "TSLA",
    "NVIDIA Corporation": "NVDA",
    "Berkshire Hathaway Inc.": "BRK.B",
    "JPMorgan Chase & Co.": "JPM",
    "Johnson & Johnson": "JNJ",
    "Visa Inc.": "V",
    "UnitedHealth Group Incorporated": "UNH",
    "Procter & Gamble Company": "PG",
    "Mastercard Incorporated": "MA",
    "Home Depot, Inc.": "HD",
    "Exxon Mobil Corporation": "XOM",
    "Bank of America Corporation": "BAC",
    "Walt Disney Company": "DIS",
    "Pfizer Inc.": "PFE",
    "AbbVie Inc.": "ABBV",
    "Merck & Co., Inc.": "MRK",
    "PepsiCo, Inc.": "PEP",
    "Coca-Cola Company": "KO",
    "Chevron Corporation": "CVX",
    "Intel Corporation": "INTC",
    "Cisco Systems, Inc.": "CSCO",
    "Adobe Inc.": "ADBE",
    "Salesforce, Inc.": "CRM",
    "Netflix, Inc.": "NFLX",
    "Oracle Corporation": "ORCL",
    "Walmart Inc.": "WMT",
    "McDonald's Corporation": "MCD",
    "Nike, Inc.": "NKE",
    "Texas Instruments Incorporated": "TXN",
    "Qualcomm Incorporated": "QCOM",
    "Costco Wholesale Corporation": "COST",
    "Thermo Fisher Scientific Inc.": "TMO",
    "Medtronic plc": "MDT",
    "Danaher Corporation": "DHR",
    "Abbott Laboratories": "ABT",
    "Eli Lilly and Company": "LLY",
    "Amgen Inc.": "AMGN",
    "Gilead Sciences, Inc.": "GILD",
    "Intuitive Surgical, Inc.": "ISRG",
    "Zoetis Inc.": "ZTS",
    "Cigna Group": "CI",
    "Regeneron Pharmaceuticals, Inc.": "REGN",
    "Vertex Pharmaceuticals Incorporated": "VRTX",
    "Broadcom Inc.": "AVGO",
    "Analog Devices, Inc.": "ADI",
    "Micron Technology, Inc.": "MU",
    "Fidelity National Information Services": "FIS",
    "FedEx Corporation": "FDX",
    "Deere & Company": "DE",
    "General Motors Company": "GM",
    "Ford Motor Company": "F",
    "General Electric Company": "GE",
    "Boeing Company": "BA",
    "Caterpillar Inc.": "CAT",
    "3M Company": "MMM",
    "International Business Machines Corporation": "IBM",
    "Altria Group, Inc.": "MO",
    "Philip Morris International Inc.": "PM",
    "Colgate-Palmolive Company": "CL",
    "Kimberly-Clark Corporation": "KMB",
    "Estée Lauder Companies Inc.": "EL",
    "Ross Stores, Inc.": "ROST",
    "TJX Companies, Inc.": "TJX",
    "Dollar General Corporation": "DG",
    "Dollar Tree, Inc.": "DLTR",
    "eBay Inc.": "EBAY",
    "Etsy, Inc.": "ETSY",
    "PayPal Holdings, Inc.": "PYPL",
    "Block, Inc.": "SQ",
    "Shopify Inc.": "SHOP",
    "Twilio Inc.": "TWLO",
    "Snowflake Inc.": "SNOW",
    "Palantir Technologies Inc.": "PLTR",
    "Zscaler, Inc.": "ZS",
    "CrowdStrike Holdings, Inc.": "CRWD",
    "Datadog, Inc.": "DDOG",
    "Okta, Inc.": "OKTA",
    "DocuSign, Inc.": "DOCU",
    "Roku, Inc.": "ROKU",
    "Uber Technologies, Inc.": "UBER",
    "Lyft, Inc.": "LYFT",
    "Rivian Automotive, Inc.": "RIVN",
    "Lucid Group, Inc.": "LCID",
    "SoFi Technologies, Inc.": "SOFI",
    "Robinhood Markets, Inc.": "HOOD",
    "American Express Company": "AXP",
    "S&P Global Inc.": "SPGI",
    "Union Pacific Corporation": "UNP",
    "Honeywell International Inc.": "HON",
    "Lowe's Companies, Inc.": "LOW",
    "Starbucks Corporation": "SBUX",
    "NextEra Energy, Inc.": "NEE",
    "United Parcel Service, Inc.": "UPS",
    "Southern Company": "SO",
    "Dominion Energy, Inc.": "D",
    "Exelon Corporation": "EXC",
    "Duke Energy Corporation": "DUK"
}

MODEL_DIR = '../saved_models'

# Prediction horizons matching training script
PREDICTION_HORIZONS = {
    'next_day': {'days': 1, 'lookback': 30, 'label': '📅 Next Day'},
    'one_week': {'days': 5, 'lookback': 40, 'label': '📊 One Week'},
    'one_month': {'days': 21, 'lookback': 60, 'label': '📈 One Month'}
}

# --- API Key Handling ---
try:
    GUARDIAN_API_KEY = st.secrets["GUARDIAN_API_KEY"]
except (FileNotFoundError, KeyError):
    GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "")
    if not GUARDIAN_API_KEY:
        st.warning("⚠️ Guardian API key not found. News sentiment will be unavailable.")


# --- Helper Functions for Technical Indicators ---
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = data.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_stochastic(df, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=k_window, min_periods=1).min()
    high_max = df['High'].rolling(window=k_window, min_periods=1).max()
    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
    return k_percent, d_percent


def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv


def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    eps = 1e-10
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / (atr + eps))
    minus_di = 100 * (minus_dm_smooth / (atr + eps))

    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + eps))
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx


def add_enhanced_features(df):
    """Add comprehensive technical indicators matching training script"""
    df = df.copy()
    eps = 1e-10

    # Basic Price Features
    df['Price_Change_Pct'] = df['Close'].pct_change().fillna(0) * 100
    df['High_Low_Range_Pct'] = ((df['High'] - df['Low']) / (df['Close'] + eps)).fillna(0) * 100
    df['Close_Open_Diff_Pct'] = ((df['Close'] - df['Open']) / (df['Open'] + eps)).fillna(0) * 100
    df['Log_Return'] = np.log((df['Close'] + eps) / (df['Close'].shift(1) + eps)).fillna(0)

    # Moving Averages - REDUCED to match training
    for period in [5, 10, 20, 50]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        df[f'Close_MA_{period}_Ratio'] = df['Close'] / (df[f'MA_{period}'] + eps)

    # EMA
    for period in [12, 26]:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False, min_periods=period).mean()

    # MA differences
    df['MA_5_20_Diff'] = ((df['MA_5'] - df['MA_20']) / (df['MA_20'] + eps) * 100).fillna(0)
    df['MA_10_50_Diff'] = ((df['MA_10'] - df['MA_50']) / (df['MA_50'] + eps) * 100).fillna(0)

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    df['MACD'] = df['MACD'].fillna(0)
    df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
    df['MACD_Histogram'] = df['MACD_Histogram'].fillna(0)

    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14).fillna(50)

    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)
    df['Stoch_K'] = df['Stoch_K'].fillna(50)
    df['Stoch_D'] = df['Stoch_D'].fillna(50)

    # Momentum - REDUCED
    for period in [5, 10, 21]:
        df[f'ROC_{period}'] = (
                    (df['Close'] - df['Close'].shift(period)) / (df['Close'].shift(period) + eps) * 100).fillna(0)

    # Volatility - REDUCED
    for period in [10, 20]:
        df[f'Volatility_{period}'] = df['Close'].rolling(window=period, min_periods=1).std()
        df[f'Volatility_{period}_Pct'] = ((df[f'Volatility_{period}'] / (df['Close'] + eps)) * 100).fillna(0)

    # Bollinger Bands - SINGLE period
    period = 20
    df[f'BB_Middle'] = df['Close'].rolling(window=period, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=period, min_periods=1).std()
    df[f'BB_Upper'] = df[f'BB_Middle'] + (bb_std * 2)
    df[f'BB_Lower'] = df[f'BB_Middle'] - (bb_std * 2)
    df[f'BB_Width'] = (((df[f'BB_Upper'] - df[f'BB_Lower']) / (df[f'BB_Middle'] + eps)) * 100).fillna(0)
    df[f'BB_Position'] = ((df['Close'] - df[f'BB_Lower']) / (df[f'BB_Upper'] - df[f'BB_Lower'] + eps)).fillna(0.5)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    df['ATR_Pct'] = ((df['ATR'] / (df['Close'] + eps)) * 100).fillna(0)

    # Volume indicators - SIMPLIFIED
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
    df['OBV'] = calculate_obv(df)

    for period in [10, 20]:
        df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period, min_periods=1).mean()
        df[f'Volume_Ratio_{period}'] = (df['Volume'] / (df[f'Volume_MA_{period}'] + eps)).fillna(1.0)

    # Price patterns
    df['Higher_High'] = ((df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))).astype(
        int).fillna(0)
    df['Lower_Low'] = ((df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))).astype(int).fillna(
        0)

    # Gap detection
    df['Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int).fillna(0)
    df['Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int).fillna(0)

    # ADX
    df['ADX'] = calculate_adx(df).fillna(20)

    # Lagged features - REDUCED
    for lag in [1, 2, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag).fillna(method='bfill')
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag).fillna(50)

    return df


# --- NLTK VADER Setup ---
@st.cache_resource
def download_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

# This line actually runs the download
download_vader()


download_vader()
SIA = SentimentIntensityAnalyzer()


# --- Caching Data & Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
@st.cache_resource
def load_horizon_models(ticker, horizon_name):
    """Load models for a specific prediction horizon"""
    loaded = {"models": {}, "scaler": None, "features": None, "thresholds": None}

    # Debug: Print what we are looking for (Visible in App logs)
    # print(f"Searching for models in {MODEL_DIR} for {ticker}...")

    # Load individual models
    model_files = {
        'LightGBM': ('stock_model_lgb', joblib.load),
        'XGBoost': ('stock_model_xgb', joblib.load),
        'Random Forest': ('stock_model_rf', joblib.load),
        'LSTM': ('stock_model_lstm', load_model) 
    }

    for model_name, (file_prefix, loader) in model_files.items():
        ext = '.h5' if model_name == 'LSTM' else '.pkl'
        
        # specific fix for Reliance vs RELIANCE.NS formatting if needed
        # safe_ticker = ticker.replace('.NS', '') # Uncomment if your files don't have .NS
        
        model_filename = f'{file_prefix}_{horizon_name}_{ticker}{ext}'
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        if os.path.exists(model_path):
            try:
                loaded["models"][model_name] = loader(model_path)
            except Exception as e:
                st.warning(f"⚠️ Error loading {model_name}: {e}")
                # Detailed error in logs
                print(f"Failed to load {model_path}: {e}")
        else:
            # Silent failure is bad for debugging. Let's log it.
            print(f"❌ File not found: {model_path}")

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{horizon_name}_{ticker}.pkl')
    if os.path.exists(scaler_path):
        try:
            loaded["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")

    # Load feature names
    feature_path = os.path.join(MODEL_DIR, f'feature_names_{horizon_name}_{ticker}.pkl')
    if os.path.exists(feature_path):
        try:
            loaded["features"] = joblib.load(feature_path)
        except Exception as e:
            st.warning(f"Could not load features: {e}")

    # Load optimal thresholds
    threshold_path = os.path.join(MODEL_DIR, f'optimal_thresholds_{horizon_name}_{ticker}.pkl')
    if os.path.exists(threshold_path):
        try:
            loaded["thresholds"] = joblib.load(threshold_path)
        except Exception as e:
            loaded["thresholds"] = {'ensemble': 0.5, 'lstm': 0.5}
    else:
        loaded["thresholds"] = {'ensemble': 0.5, 'lstm': 0.5}

    return loaded


@st.cache_data(ttl=900)


@st.cache_data(ttl=3600)
def get_live_stock_data(ticker, days_history=600):
    """
    Fetches historical stock data with robust error handling, retries,
    and rate limiting for Streamlit Cloud.
    """
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_history)
    
    # Retry configuration
    max_retries = 3
    delay = 2  # Seconds to wait between retries

    for attempt in range(max_retries):
        try:
            # --- 1. RATE LIMIT PROTECTION ---
            # Pause before request to avoid "429 Too Many Requests"
            time.sleep(1) 

            # --- 2. DOWNLOAD ---
            # progress=False prevents log spam
            stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # --- 3. VALIDATION ---
            if stock_df.empty:
                # If we get empty data, raise an error to trigger the retry logic
                if attempt < max_retries - 1:
                    raise ValueError("Received empty DataFrame")
                else:
                    return pd.DataFrame()

            # --- 4. DATA CLEANING (Preserving your original logic) ---
            # Fix MultiIndex columns (Common issue in yfinance > 0.2.0)
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)

            stock_df.reset_index(inplace=True)

            # Ensure numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in stock_df.columns:
                    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

            # Drop rows with missing values
            cols_to_check = [c for c in numeric_cols if c in stock_df.columns]
            stock_df.dropna(subset=cols_to_check, inplace=True)

            return stock_df

        except Exception as e:
            # If this was the last attempt, show error and return empty
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch stock data for {ticker}: {e}")
                return pd.DataFrame()
            
            # Exponential backoff: Wait 2s, then 4s, then 6s...
            time.sleep(delay * (attempt + 1))
    
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_latest_news(company_name, api_key):
    """Fetches recent news from The Guardian API"""
    if not api_key:
        return pd.DataFrame(columns=['Date', 'Headline'])

    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = (f'https://content.guardianapis.com/search?q="{company_name}"&from-date={date_from}'
           f'&order-by=newest&lang=en&page-size=50&api-key={api_key}')
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get('response', {}).get('results', [])
        if not articles:
            return pd.DataFrame(columns=['Date', 'Headline'])
        headlines = [
            {'Date': pd.to_datetime(a['webPublicationDate']).tz_localize(None).date(),
             'Headline': a['webTitle']}
            for a in articles
        ]
        return pd.DataFrame(headlines)
    except Exception as e:
        return pd.DataFrame(columns=['Date', 'Headline'])


def analyze_sentiment(text):
    return SIA.polarity_scores(text)['compound']


def get_sentiment_category(score):
    if score >= 0.05:
        return 'Positive', '🟢'
    elif score <= -0.05:
        return 'Negative', '🔴'
    return 'Neutral', '🟡'


def make_prediction_for_horizon(stock_df_full, models, scaler, feature_names, thresholds,
                                lookback, avg_sentiment, total_news_count, input_data):
    """Make prediction for a specific horizon"""
    # Get historical data
    historical_subset = stock_df_full.iloc[-300:].copy()

    # Create new row
    new_row = pd.DataFrame({
        'Date': [datetime.now().date()],
        'Open': [input_data['open']],
        'High': [input_data['high']],
        'Low': [input_data['low']],
        'Close': [input_data['close']],
        'Volume': [input_data['volume']],
        'Sentiment': [avg_sentiment],
        'Num_News': [total_news_count]
    })

    # Combine and calculate features
    combined_data = pd.concat([historical_subset, new_row], ignore_index=True)
    combined_data['Sentiment_MA_3'] = combined_data['Sentiment'].rolling(window=3, min_periods=1).mean()
    combined_data = add_enhanced_features(combined_data)
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.dropna(inplace=True)

    # Get prediction row
    prediction_row = combined_data.iloc[-1:].copy()

    if prediction_row.empty:
        return None

    # Check for missing features
    missing_features = [f for f in feature_names if f not in prediction_row.columns]
    if missing_features:
        st.error(f"Missing features: {len(missing_features)}")
        return None

    features_df = prediction_row[feature_names]

    if features_df.isnull().any().any():
        return None

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Make predictions
    predictions = []
    confidences = []
    pred_probas = []
    model_names = []

    for model_name, model in models.items():
        try:
            if model_name == 'LSTM':
                lstm_sequence_data = combined_data.iloc[-lookback:].copy()

                if len(lstm_sequence_data) < lookback:
                    continue

                lstm_features = lstm_sequence_data[feature_names].values
                lstm_scaled = scaler.transform(lstm_features)
                sequence = lstm_scaled.reshape(1, lookback, len(feature_names))

                pred_proba = model.predict(sequence, verbose=0)[0][0]
                threshold = thresholds.get('lstm', 0.5)
                prediction = 1 if pred_proba > threshold else 0
                confidence = pred_proba if prediction == 1 else (1 - pred_proba)
            else:
                pred_proba = model.predict_proba(features_scaled)[0][1]
                threshold = thresholds.get('ensemble', 0.5)
                prediction = 1 if pred_proba > threshold else 0
                confidence = pred_proba if prediction == 1 else (1 - pred_proba)

            predictions.append(prediction)
            confidences.append(confidence)
            pred_probas.append(pred_proba)
            model_names.append(model_name)

        except Exception as e:
            continue

    if not predictions:
        return None

    return {
        'predictions': predictions,
        'confidences': confidences,
        'pred_probas': pred_probas,
        'model_names': model_names,
        'thresholds': thresholds
    }


# --- Main Application ---
st.title('🚀 AI Multi-Horizon Stock Prediction Platform')
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%); padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;">
    Advanced <b>Multi-Timeframe Analysis</b> with predictions for <b>Next Day, One Week, and One Month</b>. 
    Leveraging ensemble ML models + LSTM with optimized thresholds for each horizon.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header('📊 Configuration')
company_name = st.sidebar.selectbox('Select Company:', list(COMPANIES.keys()), index=0)
ticker = COMPANIES[company_name]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Options")
chart_days = st.sidebar.slider("Chart History (Days)", 30, 180, 90)
show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_news_details = st.sidebar.checkbox("Show Detailed News", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Multi-Horizon Predictions**: Get insights for different time frames!")

# Load data
with st.spinner("📡 Fetching live market data..."):
    stock_df_full = get_live_stock_data(ticker)
    news_df = get_latest_news(company_name, GUARDIAN_API_KEY) if GUARDIAN_API_KEY else pd.DataFrame(
        columns=['Date', 'Headline'])

if stock_df_full.empty:
    st.error("❌ Could not fetch stock data. Please try again later.")
    st.stop()

# Process news sentiment
st.markdown("---")
st.subheader("📰 Real-Time News Sentiment Analysis")

avg_sentiment = 0.0
total_news_count = 0
daily_sentiment_df = pd.DataFrame(columns=['Date', 'Sentiment', 'Num_News'])

if not news_df.empty:
    news_df['Sentiment Score'] = news_df['Headline'].apply(analyze_sentiment)
    news_df['Sentiment Category'], news_df['Emoji'] = zip(*news_df['Sentiment Score'].apply(get_sentiment_category))
    avg_sentiment = news_df['Sentiment Score'].mean()
    total_news_count = len(news_df)

    daily_sentiment_df = news_df.groupby('Date').agg(
        Sentiment=('Sentiment Score', 'mean'),
        Num_News=('Headline', 'count')
    ).reset_index()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        sentiment_counts = news_df['Sentiment Category'].value_counts()
        color_map = {'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'}
        colors = [color_map.get(cat, 'grey') for cat in sentiment_counts.index]

        fig_sentiment = go.Figure(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values, marker_color=colors,
                   text=sentiment_counts.values, textposition='auto'))
        fig_sentiment.update_layout(title='📊 Sentiment Distribution', template='plotly_dark',
                                    height=280, showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        sentiment_cat, sentiment_emoji = get_sentiment_category(avg_sentiment)
        sentiment_color = "#00CC96" if avg_sentiment >= 0.05 else "#EF553B" if avg_sentiment <= -0.05 else "#FFA15A"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {sentiment_color}88 0%, {sentiment_color}44 100%);">
            <h3 style="margin: 0; font-size: 16px;">Avg Sentiment</h3>
            <h1 style="margin: 15px 0; font-size: 48px;">{sentiment_emoji}</h1>
            <h2 style="margin: 10px 0;">{avg_sentiment:.3f}</h2>
            <p style="margin: 0; opacity: 0.8;">{sentiment_cat}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 16px;">News Volume</h3>
            <h1 style="margin: 20px 0; font-size: 42px;">📰</h1>
            <h2 style="margin: 10px 0;">{total_news_count}</h2>
            <p style="margin: 0; opacity: 0.8;">Articles (7 days)</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("ℹ️ No recent news available. Using neutral sentiment (0.0).")

# Merge sentiment and calculate features
with st.spinner("🔬 Calculating technical indicators..."):
    stock_df_full['Date'] = pd.to_datetime(stock_df_full['Date']).dt.date
    stock_df_full = pd.merge(stock_df_full, daily_sentiment_df, on='Date', how='left')
    stock_df_full[['Sentiment', 'Num_News']] = stock_df_full[['Sentiment', 'Num_News']].fillna(0)
    stock_df_full['Sentiment_MA_3'] = stock_df_full['Sentiment'].rolling(window=3, min_periods=1).mean()

    stock_df_full = add_enhanced_features(stock_df_full)
    stock_df_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    indicator_cols = [col for col in stock_df_full.columns
                      if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Num_News']]
    stock_df_full[indicator_cols] = stock_df_full[indicator_cols].fillna(method='ffill')
    stock_df_full.dropna(inplace=True)

if stock_df_full.empty or len(stock_df_full) < 100:
    st.error(f"❌ Not enough historical data. Need at least 100 days, got {len(stock_df_full)}.")
    st.stop()

st.success(f"✅ Processed {len(stock_df_full)} days of market data")

# Current Price Section
st.markdown("---")
st.header(f"📈 {company_name} ({ticker}) - Market Overview")

latest = stock_df_full.iloc[-1]
prev = stock_df_full.iloc[-2]
price_change = latest['Close'] - prev['Close']
price_change_pct = (price_change / prev['Close']) * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Price", f"${latest['Close']:.2f}",
              f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
with col2:
    st.metric("Volume", f"{latest['Volume'] / 1e6:.1f}M",
              f"{((latest['Volume'] - prev['Volume']) / prev['Volume'] * 100):+.1f}%")
with col3:
    st.metric("RSI (14)", f"{latest['RSI']:.1f}",
              f"{(latest['RSI'] - prev['RSI']):+.1f}")
with col4:
    st.metric("MACD", f"{latest['MACD']:.2f}",
              f"{(latest['MACD'] - prev['MACD']):+.2f}")
with col5:
    st.metric("ATR", f"{latest['ATR']:.2f}",
              f"{((latest['ATR'] - prev['ATR']) / prev['ATR'] * 100):+.1f}%")

# Stock Chart
st.markdown("---")
st.subheader("📊 Interactive Price Chart")

stock_display = stock_df_full.iloc[-chart_days:].copy()
stock_display['Date'] = pd.to_datetime(stock_display['Date'])

fig_price = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.7, 0.3],
    subplot_titles=('Price with Moving Averages', 'Volume')
)

# Candlestick
fig_price.add_trace(
    go.Candlestick(
        x=stock_display['Date'],
        open=stock_display['Open'],
        high=stock_display['High'],
        low=stock_display['Low'],
        close=stock_display['Close'],
        name='OHLC',
        increasing_line_color='#00CC96',
        decreasing_line_color='#EF553B'
    ),
    row=1, col=1
)

# Moving Averages
if show_technical:
    colors_ma = {'MA_20': '#FFA500', 'MA_50': '#1E90FF'}
    for ma, color in colors_ma.items():
        if ma in stock_display.columns:
            fig_price.add_trace(
                go.Scatter(
                    x=stock_display['Date'],
                    y=stock_display[ma],
                    name=ma.replace('_', ' '),
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )

# Volume
colors_vol = ['#EF553B' if row['Close'] < row['Open'] else '#00CC96'
              for _, row in stock_display.iterrows()]
fig_price.add_trace(
    go.Bar(x=stock_display['Date'], y=stock_display['Volume'],
           name='Volume', marker_color=colors_vol, showlegend=False),
    row=2, col=1
)

fig_price.update_layout(
    height=600,
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
    showlegend=True,
    hovermode='x unified'
)
st.plotly_chart(fig_price, use_container_width=True)

# Multi-Horizon Prediction Section
st.markdown("---")
st.header("🔮 Multi-Horizon AI Predictions")
st.markdown("""
<div class="insight-box">
    <b>🎯 Three Prediction Horizons</b><br>
    Get AI predictions for Next Day, One Week (5 days), and One Month (21 days) ahead.
    Each horizon uses specialized models optimized for that timeframe.
</div>
""", unsafe_allow_html=True)

# Input form
with st.form(key='prediction_form'):
    st.markdown("**📝 Input Tomorrow's Expected Values**")
    col1, col2, col3 = st.columns(3)
    with col1:
        open_price = st.number_input('Open', value=float(latest['Open']), format="%.2f")
        high_price = st.number_input('High', value=float(latest['High']), format="%.2f")
    with col2:
        low_price = st.number_input('Low', value=float(latest['Low']), format="%.2f")
        close_price = st.number_input('Close', value=float(latest['Close']), format="%.2f")
    with col3:
        volume = st.number_input('Volume', value=int(latest['Volume']))
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(
            label='🚀 Generate Multi-Horizon Predictions',
            use_container_width=True,
            type='primary'
        )

if submit_button:
    input_data = {
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume
    }

    # Create tabs for each horizon
    tab1, tab2, tab3 = st.tabs([
        PREDICTION_HORIZONS['next_day']['label'],
        PREDICTION_HORIZONS['one_week']['label'],
        PREDICTION_HORIZONS['one_month']['label']
    ])

    all_results = {}

    for tab, (horizon_name, horizon_config) in zip([tab1, tab2, tab3], PREDICTION_HORIZONS.items()):
        with tab:
            st.markdown(f"### {horizon_config['label']} Prediction ({horizon_config['days']} trading days)")

            with st.spinner(f"🤖 Loading {horizon_name} models..."):
                horizon_assets = load_horizon_models(ticker, horizon_name)

                if not horizon_assets["models"]:
                    st.error(f"❌ No models found for {horizon_name}. Please train models first.")
                    continue

                if not horizon_assets["scaler"] or not horizon_assets["features"]:
                    st.error(f"❌ Missing scaler or features for {horizon_name}.")
                    continue

            # Display loaded models
            model_info_cols = st.columns(len(horizon_assets["models"]) + 1)
            with model_info_cols[0]:
                st.markdown("**✅ Models:**")
            for idx, model_name in enumerate(horizon_assets["models"].keys(), 1):
                with model_info_cols[idx]:
                    st.success(f"✓ {model_name}")

            # Make predictions
            with st.spinner(f"🤖 Generating {horizon_name} predictions..."):
                result = make_prediction_for_horizon(
                    stock_df_full,
                    horizon_assets["models"],
                    horizon_assets["scaler"],
                    horizon_assets["features"],
                    horizon_assets["thresholds"],
                    horizon_config['lookback'],
                    avg_sentiment,
                    total_news_count,
                    input_data
                )

                if result is None:
                    st.error(f"❌ Could not generate predictions for {horizon_name}")
                    continue

                all_results[horizon_name] = result

            # Display individual model predictions
            st.markdown("#### 🤖 Individual Model Predictions")
            pred_cols = st.columns(len(result['model_names']))

            for i, model_name in enumerate(result['model_names']):
                with pred_cols[i]:
                    prediction = result['predictions'][i]
                    confidence = result['confidences'][i]
                    pred_proba = result['pred_probas'][i]

                    pred_text = "📈 UP" if prediction == 1 else "📉 DOWN"
                    pred_color = "#00CC96" if prediction == 1 else "#EF553B"

                    threshold_key = 'lstm' if model_name == 'LSTM' else 'ensemble'
                    threshold = result['thresholds'].get(threshold_key, 0.5)

                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, {pred_color}88 0%, {pred_color}44 100%); border-color: {pred_color};">
                        <h4 style="margin: 0; opacity: 0.9;">{model_name}</h4>
                        <h1 style="margin: 15px 0; font-size: 32px;">{pred_text}</h1>
                        <p style="margin: 0; font-size: 16px;"><b>{confidence * 100:.1f}%</b> confidence</p>
                        <p style="margin: 5px 0 0 0; font-size: 11px; opacity: 0.7;">Prob: {pred_proba:.3f} | Threshold: {threshold:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Ensemble prediction
            st.markdown("#### ✨ Ensemble Prediction")

            up_votes = sum(result['predictions'])
            down_votes = len(result['predictions']) - up_votes
            final_prediction = 1 if up_votes > down_votes else 0
            avg_confidence = np.mean(result['confidences']) * 100
            avg_proba = np.mean(result['pred_probas'])
            consensus = max(up_votes, down_votes) / len(result['predictions']) * 100

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                if final_prediction == 1:
                    pred_text = "GO UP 📈"
                    pred_emoji = "🚀"
                    pred_color = "#00CC96"
                    pred_gradient = "linear-gradient(135deg, #00CC96 0%, #00A86B 100%)"
                else:
                    pred_text = "GO DOWN 📉"
                    pred_emoji = "⚠️"
                    pred_color = "#EF553B"
                    pred_gradient = "linear-gradient(135deg, #EF553B 0%, #CC3333 100%)"

                st.markdown(f"""
                <div style="background: {pred_gradient}; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
                    <h2 style="color: white; margin: 0; font-size: 20px;">{horizon_config['label']} Forecast</h2>
                    <h1 style="font-size: 60px; margin: 15px 0;">{pred_emoji}</h1>
                    <h1 style="color: white; margin: 0; font-size: 36px; font-weight: bold;">{pred_text}</h1>
                    <p style="color: white; font-size: 18px; margin: 15px 0 0 0; opacity: 0.9;">Probability: {avg_proba * 100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if final_prediction == 1:
                    if consensus >= 80 and avg_confidence >= 70:
                        suggestion = "🔥 STRONG BUY"
                        sugg_color = "#00A86B"
                        sugg_desc = "High confidence bullish"
                    elif consensus >= 70:
                        suggestion = "✅ BUY"
                        sugg_color = "#00CC96"
                        sugg_desc = "Moderate bullish signal"
                    else:
                        suggestion = "⚡ HOLD"
                        sugg_color = "#FFA15A"
                        sugg_desc = "Mixed signals"
                else:
                    if consensus >= 80 and avg_confidence >= 70:
                        suggestion = "🔻 STRONG SELL"
                        sugg_color = "#8B0000"
                        sugg_desc = "High confidence bearish"
                    elif consensus >= 70:
                        suggestion = "❌ SELL"
                        sugg_color = "#EF553B"
                        sugg_desc = "Moderate bearish signal"
                    else:
                        suggestion = "⚡ HOLD"
                        sugg_color = "#FFA15A"
                        sugg_desc = "Mixed signals"

                st.markdown(f"""
                <div style="background: {sugg_color}33; border: 3px solid {sugg_color}; border-radius: 15px; padding: 30px; text-align: center;">
                    <h3 style="color: {sugg_color}; margin: 0;">Trading Signal</h3>
                    <h1 style="color: {sugg_color}; margin: 15px 0; font-size: 32px; font-weight: bold;">{suggestion}</h1>
                    <p style="color: {sugg_color}; font-size: 14px; margin: 0;">{sugg_desc}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                consensus_text = "🔥 Strong" if consensus >= 75 else "👍 Moderate" if consensus >= 60 else "⚠️ Weak"
                st.markdown(f"""
                <div class="metric-card" style="height: 100%;">
                    <h4>Consensus</h4>
                    <h1 style="font-size: 40px; margin: 15px 0;">{consensus:.0f}%</h1>
                    <p>{consensus_text}</p>
                    <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
                    <p style="font-size: 13px;"><b>{up_votes}</b> UP</p>
                    <p style="font-size: 13px;"><b>{down_votes}</b> DOWN</p>
                </div>
                """, unsafe_allow_html=True)

            # Vote breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            vote_fig = go.Figure()
            vote_fig.add_trace(go.Bar(
                x=['UP 📈', 'DOWN 📉'],
                y=[up_votes, down_votes],
                marker_color=['#00CC96', '#EF553B'],
                text=[up_votes, down_votes],
                textposition='auto',
                textfont=dict(size=18, color='white')
            ))
            vote_fig.update_layout(
                title=f'{horizon_config["label"]} - Model Vote Distribution',
                template='plotly_dark',
                height=300,
                showlegend=False
            )
            st.plotly_chart(vote_fig, use_container_width=True)

    # Comparative Analysis across horizons
    if len(all_results) > 1:
        st.markdown("---")
        st.header("📊 Multi-Horizon Comparative Analysis")

        comparison_data = []
        for horizon_name, result in all_results.items():
            up_votes = sum(result['predictions'])
            down_votes = len(result['predictions']) - up_votes
            final_pred = 1 if up_votes > down_votes else 0
            consensus = max(up_votes, down_votes) / len(result['predictions']) * 100
            avg_conf = np.mean(result['confidences']) * 100

            comparison_data.append({
                'Horizon': PREDICTION_HORIZONS[horizon_name]['label'],
                'Days': PREDICTION_HORIZONS[horizon_name]['days'],
                'Prediction': '📈 UP' if final_pred == 1 else '📉 DOWN',
                'Consensus': f"{consensus:.1f}%",
                'Avg Confidence': f"{avg_conf:.1f}%",
                'Models Agree': f"{max(up_votes, down_votes)}/{len(result['predictions'])}"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Visualization
        fig_comparison = go.Figure()

        horizons = [d['Horizon'] for d in comparison_data]
        up_counts = [sum(all_results[h]['predictions']) for h in all_results.keys()]
        down_counts = [len(all_results[h]['predictions']) - sum(all_results[h]['predictions'])
                       for h in all_results.keys()]

        fig_comparison.add_trace(go.Bar(
            name='UP Predictions',
            x=horizons,
            y=up_counts,
            marker_color='#00CC96',
            text=up_counts,
            textposition='auto'
        ))

        fig_comparison.add_trace(go.Bar(
            name='DOWN Predictions',
            x=horizons,
            y=down_counts,
            marker_color='#EF553B',
            text=down_counts,
            textposition='auto'
        ))

        fig_comparison.update_layout(
            title='Model Predictions Across Horizons',
            barmode='group',
            template='plotly_dark',
            height=400,
            xaxis_title="Prediction Horizon",
            yaxis_title="Number of Models"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Strategic insights
        st.markdown("### 💡 Strategic Insights")

        all_up = all(sum(r['predictions']) > len(r['predictions']) / 2 for r in all_results.values())
        all_down = all(sum(r['predictions']) < len(r['predictions']) / 2 for r in all_results.values())

        if all_up:
            st.success("""
            🟢 **Strong Bullish Signal Across All Horizons**
            - All timeframes predict upward movement
            - Consider entering long positions
            - Monitor for optimal entry points
            """)
        elif all_down:
            st.error("""
            🔴 **Strong Bearish Signal Across All Horizons**
            - All timeframes predict downward movement
            - Consider exiting long positions or shorting
            - Preserve capital and wait for reversal
            """)
        else:
            st.warning("""
            🟡 **Mixed Signals Across Horizons**
            - Different timeframes show conflicting predictions
            - Short-term and long-term trends may differ
            - Consider your investment timeline carefully
            - Wait for clearer signals or use hedging strategies
            """)

# Disclaimer
st.markdown("---")
st.error("""
### ⚠️ CRITICAL DISCLAIMER

**This multi-horizon prediction tool is for educational purposes ONLY and should NOT be considered as financial advice.**

**Important Warnings:**
- 🚫 AI predictions are NOT guarantees of future performance
- 📉 Stock markets are inherently unpredictable
- 💰 Only invest what you can afford to lose
- 🏦 Consult licensed financial advisors before investing
- 📚 Conduct thorough independent research
- ⚖️ Consider your personal risk tolerance

**Model Limitations:**
- Each horizon has different accuracy levels
- Longer horizons typically have higher uncertainty
- Unexpected events can invalidate predictions
- Past performance does not guarantee future results

By using this tool, you acknowledge full responsibility for your investment decisions.
""")

# Footer
st.markdown("---")
st.markdown("### 📊 Multi-Horizon Technology Stack")

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("""
    **🎯 Prediction Horizons**
    - Next Day (1 day)
    - One Week (5 days)
    - One Month (21 days)
    """)
with footer_col2:
    st.markdown("""
    **🤖 AI Models per Horizon**
    - LightGBM
    - XGBoost
    - Random Forest
    - LSTM (Deep Learning)
    """)
with footer_col3:
    st.markdown("""
    **📊 Features**
    - Adaptive thresholds
    - Optimized lookback windows
    - Real-time sentiment analysis
    """)

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: gray; font-size: 0.9em;">
    <p>🚀 <b>AI Multi-Horizon Stock Prediction Platform</b> | Built with Streamlit, TensorFlow, and Scikit-learn</p>
    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version 3.0</p>
    <p>Made with ❤️ for educational purposes | © 2024</p>
</div>

""", unsafe_allow_html=True)





