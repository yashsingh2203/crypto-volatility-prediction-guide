# crypto-volatility-prediction-guide
# 🪙 Cryptocurrency Volatility Prediction

This project uses machine learning to predict cryptocurrency market volatility using historical OHLC data, trading volume, and market capitalization. It helps traders and institutions identify periods of high risk and make informed decisions.

---

## 📊 Features

- Daily OHLC, volume, and market cap analysis
- Rolling volatility calculation
- Liquidity ratios and technical indicators
- Machine learning model using Random Forest Regressor
- Streamlit web app for interactive prediction

---

## 🧰 Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Git/GitHub

---

## 📁 Project Structure

```bash
├── crypto_volatility_model.py      # Full pipeline (preprocessing, training, Streamlit app)
├── crypto_data.csv                 # Input dataset
├── volatility_model.pkl            # Trained model
├── scaler.pkl                      # Trained scaler
├── README.md                       # This file
├── requirements.txt                # Python dependencies

