from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import logging
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from datetime import datetime, time

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize NewsAPI client
NEWS_API_KEY = 'c241b84a89b245d0bd43c63294f75e27'
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def is_market_open():
    now = datetime.now()
    market_open_time = time(9, 15)
    market_close_time = time(15, 30)
    return market_open_time <= now.time() <= market_close_time

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the frontend
        data = request.get_json()
        stock_name = data.get('stock_name', "").strip().upper()

        if not stock_name:
            return jsonify({"error": "Please provide a valid stock symbol."}), 400

        # Append .NS for Indian stocks if not provided
        if not stock_name.endswith(('.NS', '.BO')):  
            stock_name += '.NS'

        logging.info(f"Fetching data for stock: {stock_name}")

        # Fetch stock data
        stock_data = yf.download(stock_name, period="6mo", interval="1d")

        if stock_data.empty:
            logging.error(f"No data found for stock: {stock_name}")
            return jsonify({"error": f"No data found for {stock_name}. Please check the ticker symbol."}), 404

        stock_data['Date'] = stock_data.index
        stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

        # Get the last known closing price
        last_row = stock_data.iloc[-1]
        current_price = last_row['Close'] if is_market_open() else last_row['Close']

        # Prepare data for prediction
        X = stock_data['Days'].values.reshape(-1, 1)
        y = stock_data['Close'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next 10 days
        last_day = stock_data['Days'].iloc[-1]
        future_days = np.arange(last_day + 1, last_day + 11).reshape(-1, 1)
        predictions = model.predict(future_days)

        # Sentiment Analysis: Get news related to the stock name
        sentiment_score = 0
        try:
            articles = newsapi.get_everything(q=stock_name, language='en', sort_by='publishedAt', page_size=5)
            if articles['status'] == 'ok' and articles['totalResults'] > 0:
                for article in articles['articles']:
                    text = article['title'] + " " + article['description']
                    sentiment = analyzer.polarity_scores(text)
                    sentiment_score += sentiment['compound']

                sentiment_score /= len(articles['articles'])
        except Exception as e:
            logging.warning(f"NewsAPI error: {str(e)}")
            sentiment_score = 0  # Default to neutral if API fails

        sentiment_adjustment = sentiment_score * 10
        adjusted_predictions = predictions + sentiment_adjustment

        # Convert predictions to JSON
        predictions_list = [
            {"Day": "0 (Current)", "Predicted Price": f"₹{round(float(current_price), 2)}"}
        ]
        predictions_list += [
            {"Day": i + 1, "Predicted Price": f"₹{round(float(price), 2)}"} for i, price in enumerate(adjusted_predictions)
        ]

        return jsonify({"predictions": predictions_list})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
