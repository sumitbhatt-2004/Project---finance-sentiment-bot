import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from main import prepare_data, build_lstm_model
from sentiment_engine import get_sentiment_score

st.set_page_config(page_title="Fin-AI: Sentiment & LSTM Predictor", layout="wide")

st.title("📈 Fin-AI Dashboard")
st.subheader("Deep Learning & NLP based Market Analysis")

# Sidebar for Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
analyze_button = st.sidebar.button("Run Analysis")

if analyze_button:
    with st.spinner('Fetching market data and training model...'):
        # 1. Fetch Price Data
        data = yf.download(ticker, period="2y")
        
        if data.empty:
            st.error("Could not find data for this ticker. Please check the symbol.")
        else:
            # 2. Display Price Chart
            st.line_chart(data['Close'])

            # 3. Deep Learning Prediction (LSTM)
            x_train, y_train, scaler = prepare_data(data)
            model = build_lstm_model((x_train.shape[1], 1))
            
            # Training for a few epochs for demonstration
            model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # Predict the next price
            last_60_days = data['Close'][-60:].values.reshape(-1, 1)
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = np.array([last_60_days_scaled])
            prediction_scaled = model.predict(X_test)
            predicted_price = scaler.inverse_transform(prediction_scaled)

            # 4. Sentiment Analysis (Simulated for Demo)
            # In a real app, we would scrape news. For now, we analyze a general mood.
            news_headline = f"Market analysts weigh in on {ticker} performance and future growth."
            score = get_sentiment_score(news_headline)

            # 5. Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Tomorrow's Price", f"${predicted_price[0][0]:.2f}")
            with col2:
                sentiment_label = "Positive" if score >= 0 else "Negative"
                st.metric("Market Sentiment Score", f"{score:.2f}", delta=sentiment_label)

            st.write("---")
            st.info(f"**Strategy Note:** As a Finance student, remember that LSTM predicts patterns, while Sentiment predicts mood. Always use both for a holistic view.")
