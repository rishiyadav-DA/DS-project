import os
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

# --- Configuration ---
COMPANIES = {
    "Tata Motors": "TATAMOTORS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "Apple Inc": "AAPL",
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

DATA_DIR = '../data'
MODEL_DIR = '../saved_models'
LOOK_BACK = 30  # Reduced from 60 for better stability
MIN_TRAINING_SAMPLES = 500  # Minimum samples needed

# Prediction horizons
PREDICTION_HORIZONS = {
    'next_day': 1,  # Next day prediction
    'one_week': 5,  # 1 week (5 trading days)
    'one_month': 21  # 1 month (21 trading days)
}


# --- Helper Functions ---
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index with proper handling"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD with proper initialization"""
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
    """Add comprehensive technical indicators - IMPROVED VERSION"""
    eps = 1e-10

    # Basic price features
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    df['High_Low_Range_Pct'] = ((df['High'] - df['Low']) / (df['Close'] + eps)) * 100
    df['Close_Open_Diff_Pct'] = ((df['Close'] - df['Open']) / (df['Open'] + eps)) * 100
    df['Log_Return'] = np.log((df['Close'] + eps) / (df['Close'].shift(1) + eps))

    # Moving Averages - REDUCED to most important ones
    for period in [5, 10, 20, 50]:
        df[f'MA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        df[f'Close_MA_{period}_Ratio'] = df['Close'] / (df[f'MA_{period}'] + eps)

    # EMA
    for period in [12, 26]:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False, min_periods=period).mean()

    # Key MA differences
    df['MA_5_20_Diff'] = (df['MA_5'] - df['MA_20']) / (df['MA_20'] + eps) * 100
    df['MA_10_50_Diff'] = (df['MA_10'] - df['MA_50']) / (df['MA_50'] + eps) * 100

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])

    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)

    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)

    # Momentum - REDUCED
    for period in [5, 10, 21]:  # Added 21 for monthly momentum
        df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / (df['Close'].shift(period) + eps)) * 100

    # Volatility - REDUCED
    for period in [10, 20]:
        df[f'Volatility_{period}'] = df['Close'].rolling(window=period, min_periods=1).std()
        df[f'Volatility_{period}_Pct'] = (df[f'Volatility_{period}'] / (df['Close'] + eps)) * 100

    # Bollinger Bands - SINGLE period
    period = 20
    df[f'BB_Middle'] = df['Close'].rolling(window=period, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=period, min_periods=1).std()
    df[f'BB_Upper'] = df[f'BB_Middle'] + (bb_std * 2)
    df[f'BB_Lower'] = df[f'BB_Middle'] - (bb_std * 2)
    df[f'BB_Width'] = ((df[f'BB_Upper'] - df[f'BB_Lower']) / (df[f'BB_Middle'] + eps)) * 100
    df[f'BB_Position'] = (df['Close'] - df[f'BB_Lower']) / (df[f'BB_Upper'] - df[f'BB_Lower'] + eps)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    df['ATR_Pct'] = (df['ATR'] / (df['Close'] + eps)) * 100

    # Volume indicators - SIMPLIFIED
    df['Volume_Change'] = df['Volume'].pct_change()
    df['OBV'] = calculate_obv(df)

    for period in [10, 20]:
        df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period, min_periods=1).mean()
        df[f'Volume_Ratio_{period}'] = df['Volume'] / (df[f'Volume_MA_{period}'] + eps)

    # Price patterns
    df['Higher_High'] = ((df['High'] > df['High'].shift(1)) &
                         (df['High'].shift(1) > df['High'].shift(2))).astype(int)
    df['Lower_Low'] = ((df['Low'] < df['Low'].shift(1)) &
                       (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)

    # Gap detection
    df['Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int)

    # ADX
    df['ADX'] = calculate_adx(df)

    # Lagged features - REDUCED
    for lag in [1, 2, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)

    return df


def create_dataset(X, y, look_back=1):
    """Create sequences for LSTM"""
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back), :])
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY)


def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal classification threshold using F1 score"""
    thresholds = np.arange(0.35, 0.65, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def create_ensemble_model(X_train, y_train, X_test):
    """Train ensemble of models with improved hyperparameters"""

    # Model 1: LightGBM with better parameters
    lgb_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    lgb_model.fit(X_train, y_train)

    # Model 2: XGBoost with better parameters
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # Model 3: Random Forest with conservative settings
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Get probability predictions
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Balanced ensemble
    ensemble_proba = (
            0.40 * lgb_pred_proba +
            0.35 * xgb_pred_proba +
            0.25 * rf_pred_proba
    )

    return lgb_model, xgb_model, rf_model, ensemble_proba


def train_horizon_models(stock_df, ticker, horizon_name, horizon_days):
    """Train models for a specific prediction horizon"""
    print(f"\n{'=' * 80}")
    print(f"Training for {horizon_name.upper()} ({horizon_days} days ahead)")
    print(f"{'=' * 80}")

    # Create a copy to avoid modifying original
    df = stock_df.copy()

    # Calculate target based on horizon
    price_changes = df['Close'].pct_change(periods=horizon_days).shift(-horizon_days) * 100

    # Adaptive threshold based on horizon
    if horizon_days == 1:
        threshold_factor = 0.3
    elif horizon_days == 5:
        threshold_factor = 0.5
    else:  # 21 days
        threshold_factor = 0.7

    threshold = price_changes.std() * threshold_factor
    df['Price_Up'] = np.where(price_changes > threshold, 1, 0)

    print(f"Using adaptive threshold: {threshold:.3f}%")

    # Remove rows with NaN targets
    df = df[:-horizon_days]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    print(f"Total rows: {len(df)}")

    class_dist = df['Price_Up'].value_counts(normalize=True)
    print(f"Class Distribution:")
    print(f"  Down (0): {class_dist.get(0, 0) * 100:.2f}%")
    print(f"  Up (1): {class_dist.get(1, 0) * 100:.2f}%")

    # Feature selection
    feature_columns = [col for col in df.columns
                       if col not in ['Date', 'Price_Up'] and
                       df[col].dtype in ['float64', 'int64']]

    X = df[feature_columns].values
    y = df['Price_Up'].values.astype(int)

    # Check minimum data requirement
    min_required = max(MIN_TRAINING_SAMPLES, LOOK_BACK + horizon_days + 100)
    if len(X) < min_required:
        print(f"❌ Insufficient data. Need {min_required} rows, have {len(X)}. Skipping.")
        return None

    # Train-test split (chronological)
    train_size = int(len(X) * 0.8)
    X_train_raw = X[:train_size]
    X_test_raw = X[train_size:]
    y_train_raw = y[:train_size]
    y_test_raw = y[train_size:]

    print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # Scale data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{horizon_name}_{ticker}.pkl'))

    # Balanced resampling
    print("Applying balanced resampling...")
    oversample = SMOTE(sampling_strategy=0.8, random_state=42)
    undersample = RandomUnderSampler(sampling_strategy=0.9, random_state=42)

    X_temp, y_temp = oversample.fit_resample(X_train_scaled, y_train_raw)
    X_train_balanced, y_train_balanced = undersample.fit_resample(X_temp, y_temp)

    print(f"After resampling - Train samples: {len(X_train_balanced)}")
    print(f"  Class 0: {np.sum(y_train_balanced == 0)}, Class 1: {np.sum(y_train_balanced == 1)}")

    # Train ensemble
    print(f"\n{'=' * 50}")
    print(f"Training Ensemble Models for {horizon_name}")
    print(f"{'=' * 50}")

    lgb_model, xgb_model, rf_model, ensemble_proba = create_ensemble_model(
        X_train_balanced, y_train_balanced, X_test_scaled
    )

    # Find optimal threshold
    ensemble_threshold, best_f1 = find_optimal_threshold(y_test_raw, ensemble_proba)
    print(f"Optimal threshold: {ensemble_threshold:.3f} (F1: {best_f1:.4f})")

    # Evaluate ensemble
    ensemble_pred = (ensemble_proba > ensemble_threshold).astype(int)
    ensemble_accuracy = accuracy_score(y_test_raw, ensemble_pred)
    ensemble_f1 = f1_score(y_test_raw, ensemble_pred, zero_division=0)

    try:
        ensemble_auc = roc_auc_score(y_test_raw, ensemble_proba)
    except:
        ensemble_auc = 0.0

    print(f"\n{'=' * 40}")
    print(f"Ensemble Results - {horizon_name}")
    print(f"{'=' * 40}")
    print(f"Accuracy: {ensemble_accuracy:.4f}")
    print(f"F1 Score: {ensemble_f1:.4f}")
    print(f"ROC-AUC: {ensemble_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_raw, ensemble_pred,
                                target_names=['Down', 'Up'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_raw, ensemble_pred))

    # Save ensemble models
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, f'stock_model_lgb_{horizon_name}_{ticker}.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, f'stock_model_xgb_{horizon_name}_{ticker}.pkl'))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, f'stock_model_rf_{horizon_name}_{ticker}.pkl'))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, f'feature_names_{horizon_name}_{ticker}.pkl'))
    joblib.dump({'ensemble': ensemble_threshold},
                os.path.join(MODEL_DIR, f'optimal_thresholds_{horizon_name}_{ticker}.pkl'))

    # LSTM with adjusted architecture based on horizon
    print(f"\n{'=' * 50}")
    print(f"Training LSTM Model for {horizon_name}")
    print(f"{'=' * 50}")

    # Adjust LOOK_BACK based on horizon
    if horizon_days == 1:
        lstm_lookback = 30
        lstm_units = 64
    elif horizon_days == 5:
        lstm_lookback = 40
        lstm_units = 80
    else:  # 21 days
        lstm_lookback = 60
        lstm_units = 96

    X_train_lstm, y_train_lstm = create_dataset(X_train_balanced, y_train_balanced, lstm_lookback)
    X_test_lstm, y_test_lstm = create_dataset(X_test_scaled, y_test_raw, lstm_lookback)

    if X_train_lstm.shape[0] < 100:
        print(f"❌ Insufficient LSTM training data. Skipping LSTM.")
        return {
            'ensemble': {
                'accuracy': ensemble_accuracy,
                'f1': ensemble_f1,
                'auc': ensemble_auc
            }
        }

    print(f"LSTM - Train: {len(X_train_lstm)}, Test: {len(X_test_lstm)}")

    # LSTM architecture adjusted for horizon
    lstm_model = Sequential([
        LSTM(lstm_units, return_sequences=True,
             input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15,
                               restore_best_weights=True, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=0.00001, mode='min', verbose=1)

    print("Training LSTM...")
    history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate LSTM
    print("\nEvaluating LSTM...")
    lstm_pred_probs = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    lstm_threshold, lstm_best_f1 = find_optimal_threshold(y_test_lstm, lstm_pred_probs)
    print(f"Optimal LSTM threshold: {lstm_threshold:.3f} (F1: {lstm_best_f1:.4f})")

    lstm_pred = (lstm_pred_probs > lstm_threshold).astype(int)
    lstm_accuracy = accuracy_score(y_test_lstm, lstm_pred)
    lstm_f1 = f1_score(y_test_lstm, lstm_pred, zero_division=0)

    try:
        lstm_auc = roc_auc_score(y_test_lstm, lstm_pred_probs)
    except:
        lstm_auc = 0.0

    print(f"\n{'=' * 40}")
    print(f"LSTM Results - {horizon_name}")
    print(f"{'=' * 40}")
    print(f"Accuracy: {lstm_accuracy:.4f}")
    print(f"F1 Score: {lstm_f1:.4f}")
    print(f"ROC-AUC: {lstm_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_lstm, lstm_pred,
                                target_names=['Down', 'Up'], zero_division=0))

    # Save LSTM
    lstm_model.save(os.path.join(MODEL_DIR, f'stock_model_lstm_{horizon_name}_{ticker}.h5'))

    thresholds = joblib.load(os.path.join(MODEL_DIR, f'optimal_thresholds_{horizon_name}_{ticker}.pkl'))
    thresholds['lstm'] = lstm_threshold
    joblib.dump(thresholds, os.path.join(MODEL_DIR, f'optimal_thresholds_{horizon_name}_{ticker}.pkl'))

    return {
        'ensemble': {
            'accuracy': ensemble_accuracy,
            'f1': ensemble_f1,
            'auc': ensemble_auc
        },
        'lstm': {
            'accuracy': lstm_accuracy,
            'f1': lstm_f1,
            'auc': lstm_auc
        }
    }


# --- Main Execution ---
if __name__ == "__main__":
    nltk.download('vader_lexicon', quiet=True)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    tf.random.set_seed(42)
    np.random.seed(42)

    for company_name, ticker in COMPANIES.items():
        print(f"\n{'=' * 80}")
        print(f"Processing {company_name} ({ticker})")
        print(f"{'=' * 80}")

        stock_data_path = os.path.join(DATA_DIR, f'stock_data_{ticker}.csv')
        news_data_path = os.path.join(DATA_DIR, f'news_headlines_{ticker}.csv')

        if not os.path.exists(stock_data_path):
            print(f"❌ Stock data not found. Skipping.")
            continue

        # Load data
        stock_df = pd.read_csv(stock_data_path)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

        stock_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        stock_df.sort_values('Date', inplace=True)
        stock_df.reset_index(drop=True, inplace=True)

        # News sentiment (optional) - IMPROVED WITH DEBUGGING
        if os.path.exists(news_data_path):
            try:
                print(f"Loading news data from: {news_data_path}")
                news_df = pd.read_csv(news_data_path)

                # Debug: Show news dataframe info
                print(f"  News data shape: {news_df.shape}")
                print(f"  News columns: {list(news_df.columns)}")

                # Handle different possible column names
                date_col = None
                news_col = None

                # Try to find date column
                for col in news_df.columns:
                    if col.lower() in ['date', 'datetime', 'timestamp', 'time']:
                        date_col = col
                        break

                # Try to find news/headline column
                for col in news_df.columns:
                    if col.lower() in ['news', 'headline', 'headlines', 'title', 'text', 'description']:
                        news_col = col
                        break

                if date_col is None or news_col is None:
                    raise ValueError(f"Could not find date or news column. Columns found: {list(news_df.columns)}")

                print(f"  Using date column: '{date_col}', news column: '{news_col}'")

                # Rename columns for consistency
                news_df = news_df.rename(columns={date_col: 'Date', news_col: 'News'})
                news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')

                # Drop rows with invalid dates
                news_df = news_df.dropna(subset=['Date'])

                if len(news_df) == 0:
                    raise ValueError("No valid news data after date parsing")

                print(f"  Valid news entries: {len(news_df)}")

                sia = SentimentIntensityAnalyzer()


                def get_sentiment_and_count(news_series):
                    if news_series.empty or news_series.isnull().all():
                        return pd.Series([0.0, 0], index=['Sentiment', 'Num_News'])

                    # Filter out empty/nan values
                    valid_news = [str(s) for s in news_series.dropna() if str(s).strip() and str(s).lower() != 'nan']

                    if not valid_news:
                        return pd.Series([0.0, 0], index=['Sentiment', 'Num_News'])

                    combined_text = ' '.join(valid_news)
                    sentiment = sia.polarity_scores(combined_text)['compound']
                    count = len(valid_news)

                    return pd.Series([sentiment, count], index=['Sentiment', 'Num_News'])


                # Group by date and calculate sentiment
                print(f"  Calculating sentiment for each date...")

                # Create lists to store results
                dates = []
                sentiments = []
                news_counts = []

                for date, news_group in news_df.groupby('Date')['News']:
                    result = get_sentiment_and_count(news_group)
                    dates.append(date)
                    sentiments.append(result['Sentiment'])
                    news_counts.append(result['Num_News'])

                # Create dataframe with proper structure
                news_df_grouped = pd.DataFrame({
                    'Date': dates,
                    'Sentiment': sentiments,
                    'Num_News': news_counts
                })

                print(f"  Grouped news by date: {len(news_df_grouped)} unique dates")
                print(f"  Grouped columns: {list(news_df_grouped.columns)}")

                # Merge with stock data
                stock_df = pd.merge(stock_df, news_df_grouped, on='Date', how='left')

                # Fill missing sentiments
                stock_df['Sentiment'] = stock_df['Sentiment'].fillna(0.0)
                stock_df['Num_News'] = stock_df['Num_News'].fillna(0).astype(int)

                # Count how many dates have news
                dates_with_news = (stock_df['Num_News'] > 0).sum()

                # Add sentiment momentum
                stock_df['Sentiment_MA_3'] = stock_df['Sentiment'].rolling(
                    window=3, min_periods=1).mean()

                print(f"✓ News sentiment added successfully!")
                print(
                    f"  Dates with news: {dates_with_news}/{len(stock_df)} ({dates_with_news / len(stock_df) * 100:.1f}%)")
                print(f"  Avg sentiment: {stock_df['Sentiment'].mean():.3f}")
                print(f"  Avg news per day: {stock_df['Num_News'].mean():.2f}")

            except Exception as e:
                print(f"⚠️  News sentiment skipped - Error: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                import traceback

                print(f"  Traceback: {traceback.format_exc()}")
                stock_df['Sentiment'] = 0.0
                stock_df['Num_News'] = 0
                stock_df['Sentiment_MA_3'] = 0.0
        else:
            print(f"⚠️  News file not found: {news_data_path}")
            stock_df['Sentiment'] = 0.0
            stock_df['Num_News'] = 0
            stock_df['Sentiment_MA_3'] = 0.0

        print("Adding enhanced features...")
        stock_df = add_enhanced_features(stock_df)

        # Store results for summary
        results = {}

        # Train models for each horizon
        for horizon_name, horizon_days in PREDICTION_HORIZONS.items():
            result = train_horizon_models(stock_df.copy(), ticker, horizon_name, horizon_days)
            if result:
                results[horizon_name] = result

        # Print comprehensive summary
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE SUMMARY for {company_name} ({ticker})")
        print(f"{'=' * 80}")

        for horizon_name in ['next_day', 'one_week', 'one_month']:
            if horizon_name in results:
                print(f"\n{horizon_name.upper().replace('_', ' ')}:")
                print(f"{'-' * 40}")

                if 'ensemble' in results[horizon_name]:
                    ens = results[horizon_name]['ensemble']
                    print(f"  Ensemble - Acc: {ens['accuracy']:.4f}, "
                          f"F1: {ens['f1']:.4f}, AUC: {ens['auc']:.4f}")

                if 'lstm' in results[horizon_name]:
                    lstm = results[horizon_name]['lstm']
                    print(f"  LSTM     - Acc: {lstm['accuracy']:.4f}, "
                          f"F1: {lstm['f1']:.4f}, AUC: {lstm['auc']:.4f}")

        print(f"\n{'=' * 80}\n")

    print("\n" + "=" * 80)
    print("✅ Multi-horizon model training complete!")
    print("=" * 80)
    print("\nTrained models for:")
    print("  • Next Day (1 day ahead)")
    print("  • One Week (5 trading days ahead)")
    print("  • One Month (21 trading days ahead)")
    print("\nModels saved in:", MODEL_DIR)