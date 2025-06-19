# crypto-volatility-prediction-guide
# crypto_volatility_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import streamlit as st

# Load and preprocess dataset
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values(by=['symbol', 'date'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

# Feature engineering
def engineer_features(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_volatility'] = df.groupby('symbol')['log_return'].transform(lambda x: x.rolling(window=14).std())
    df['liquidity_ratio'] = df['volume'] / df['market_cap']
    df['ma7'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=7).mean())
    df['ma30'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=30).mean())
    df['bb_upper'] = df['ma30'] + 2 * df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=30).std())
    df['bb_lower'] = df['ma30'] - 2 * df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=30).std())
    df.dropna(inplace=True)
    return df

# Model training
def train_model(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'market_cap',
                'log_return', 'rolling_volatility', 'liquidity_ratio', 'ma7', 'ma30', 'bb_upper', 'bb_lower']
    target = 'rolling_volatility'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    joblib.dump(model, 'volatility_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Streamlit App
def launch_app():
    st.title("Cryptocurrency Volatility Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        df = engineer_features(df)

        features = ['open', 'high', 'low', 'close', 'volume', 'market_cap',
                    'log_return', 'rolling_volatility', 'liquidity_ratio', 'ma7', 'ma30', 'bb_upper', 'bb_lower']

        scaler = joblib.load('scaler.pkl')
        model = joblib.load('volatility_model.pkl')

        X = scaler.transform(df[features])
        predictions = model.predict(X)
        df['predicted_volatility'] = predictions

        st.write(df[['date', 'symbol', 'predicted_volatility']].tail(50))
        st.line_chart(df.set_index('date')['predicted_volatility'])

# Main Execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        data = load_data("crypto_data.csv")
        data = engineer_features(data)
        train_model(data)
    else:
        launch_app()
