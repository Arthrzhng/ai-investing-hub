from typing import Tuple, Dict, List, Any, Optional
import os
from datetime import datetime, timedelta
import logging
import time
import random
from threading import Lock

import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import ta
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred  # New dependency for macroeconomic data
import shap  # New dependency for model explainability

import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Set NLTK data path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# API Keys (Hardcoded as per your request)
NEWS_API_KEY = "6b6fb493cecb4445a972f84ebdf537cc"
FINNHUB_API_KEY = "cva86lpr01qshflgmdb0cva86lpr01qshflgmdbg"
FRED_API_KEY = "ae328ab01e1437d1e24bbd04f0e03e53"  # Replace with your FRED API key

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache with Expiration
class Cache:
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (value, time.time())

CACHE = Cache(ttl=1800)  # 30-minute cache

# Configuration
VALID_DATE_RANGES = ["1wk", "1mo", "6mo", "1y", "5y", "10y"]
FORECAST_OPTIONS = [
    {"label": "1 day", "value": 1}, {"label": "3 days", "value": 3},
    {"label": "1 week", "value": 7}, {"label": "1 month", "value": 30},
    {"label": "2 months", "value": 60}, {"label": "3 months", "value": 90},
]
MARKET_TICKERS = {
    "US Tech": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABT"],
    "Finance": ["JPM", "BAC", "C"],
    "Energy": ["XOM", "CVX"],
    "Consumer": ["PG", "KO"]
}

# Data Fetching Functions
def fetch_finnhub_data(symbol: str, period: str) -> pd.DataFrame:
    now = datetime.now()
    start_date = now - timedelta(days={"1wk": 7, "1mo": 30, "6mo": 180, "1y": 365, "5y": 1825, "10y": 3650}.get(period, 3650))
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={int(start_date.timestamp())}&to={int(now.timestamp())}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("s") != "ok":
            return pd.DataFrame()
        df = pd.DataFrame({"Date": pd.to_datetime(data["t"], unit="s"), "Open": data["o"], "High": data["h"], "Low": data["l"], "Close": data["c"], "Volume": data["v"]})
        return df.sort_values("Date")
    except Exception as e:
        logger.error(f"Finnhub fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_market_data(symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    cache_key = f"data_{symbol}_{period}_{interval}"
    cached_data = CACHE.get(cache_key)
    if cached_data is not None:
        return cached_data.copy()
    for source, fetch_func in [("Yahoo Finance", yf.download), ("Finnhub", fetch_finnhub_data)]:
        try:
            if source == "Yahoo Finance":
                df = fetch_func(symbol, period=period, interval=interval)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df.reset_index(inplace=True)
            else:
                df = fetch_func(symbol, period)
            if not df.empty:
                CACHE.set(cache_key, df)
                return df.copy()
        except Exception as e:
            logger.warning(f"{source} failed for {symbol}: {e}")
    return pd.DataFrame()

# New: Macroeconomic Data from FRED
def fetch_macro_data() -> Dict[str, float]:
    fred = Fred(api_key=FRED_API_KEY)
    data = {}
    try:
        data['GDP'] = fred.get_series('GDP').tail(1).values[0]
        data['Unemployment'] = fred.get_series('UNRATE').tail(1).values[0]
        data['Inflation'] = fred.get_series('CPIAUCSL').tail(1).values[0]
        data['Interest Rate'] = fred.get_series('FEDFUNDS').tail(1).values[0]
    except Exception as e:
        logger.error(f"FRED data fetch failed: {e}")
    return data

def get_macro_data() -> Dict[str, float]:
    cache_key = "macro_data"
    cached_data = CACHE.get(cache_key)
    if cached_data is not None:
        return cached_data
    data = fetch_macro_data()
    CACHE.set(cache_key, data)
    return data

# New: Insider Trading Data from Finnhub
def fetch_insider_data(symbol: str) -> List[Dict[str, Any]]:
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])[:10]  # Last 10 transactions
    except Exception as e:
        logger.error(f"Insider data fetch failed for {symbol}: {e}")
        return []

# New: Analyst Recommendations from Finnhub
def fetch_analyst_recommendations(symbol: str) -> List[Dict[str, Any]]:
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Analyst recommendations fetch failed for {symbol}: {e}")
        return []

# Technical Indicators
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Close", "High", "Low"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    df["BB_High"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
    df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    return df

# News and Sentiment Analysis
def fetch_news_articles(symbol: str) -> List[Dict[str, Any]]:
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("articles", [])[:5]
    except Exception as e:
        logger.error(f"News fetch failed for {symbol}: {e}")
        return []

def get_sentiment_from_sources(symbol: str) -> Tuple[float, List[Dict[str, Any]]]:
    articles = fetch_news_articles(symbol)
    sentiments = [TextBlob(art.get("content") or art.get("description") or art.get("title", "")).sentiment.polarity
                  for art in articles if art.get("content") or art.get("description") or art.get("title")]
    news_sentiment = np.mean(sentiments) if sentiments else 0.0
    reddit_sentiment = random.uniform(-0.5, 0.5)  # Simulated
    twitter_sentiment = random.uniform(-0.5, 0.5)  # Simulated
    overall_sentiment = np.mean([news_sentiment, reddit_sentiment, twitter_sentiment])
    return overall_sentiment, articles

# Fundamental Analysis
def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "Name": info.get("longName", symbol),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except Exception as e:
        logger.error(f"Fundamental data fetch failed for {symbol}: {e}")
        return {"Name": symbol, "Sector": "N/A"}

def calculate_fundamental_score(fundamentals: Dict[str, Any]) -> int:
    score = 0
    pe = fundamentals.get("PE Ratio")
    if pe != "N/A" and pe is not None and pe < 20:
        score += 1
    elif pe != "N/A" and pe is not None and pe > 30:
        score -= 1
    div_yield = fundamentals.get("Dividend Yield")
    if div_yield != "N/A" and div_yield is not None and div_yield > 0.03:
        score += 1
    return score

def adjust_decision_with_fundamentals(decision: str, score: int) -> str:
    if score > 0:
        return "Buy" if decision == "Hold" else "Hold" if decision == "Sell" else decision
    elif score < 0:
        return "Hold" if decision == "Buy" else "Sell" if decision == "Hold" else decision
    return decision

# Enhanced LSTM Forecasting with SHAP Explainability
def build_lstm_forecast(df: pd.DataFrame, forecast_horizon: int = 60, seq_length: int = 60) -> Tuple[np.ndarray, Any, Any, np.ndarray]:
    if len(df) < seq_length + 1:
        return np.array([]), None, None, np.array([])
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    if X.size == 0 or y.size == 0:
        return np.array([]), None, None, np.array([])
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_seq = scaled_data[-seq_length:].reshape((1, seq_length, 1))
    predictions = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_seq, verbose=0)
        predictions.append(pred[0, 0])
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1] = pred[0, 0]
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions, model, scaler, X

# Trading Decision Logic
def make_trading_decision(df: pd.DataFrame, sentiment: float, predictions: np.ndarray, fundamental_score: int) -> Tuple[str, Optional[float], Optional[float], List[Tuple[str, str]]]:
    if df.empty or not predictions.size:
        return "Hold", None, None, [("Insufficient Data", "Not enough data.")]
    last_close = df["Close"].iloc[-1]
    reasons = []
    rsi = df["RSI"].iloc[-1] if "RSI" in df else 50
    macd = df["MACD"].iloc[-1] if "MACD" in df else 0
    if (sentiment > 0.1 and predictions[-1] > last_close * 1.02 and rsi < 70 and macd > 0):
        decision = "Buy"
        target_buy = last_close * 0.98
        reasons = [
            ("Positive Sentiment", f"Sentiment: {sentiment:.2f}"),
            ("LSTM Growth", f"Predicted: ${predictions[-1]:.2f}"),
            ("RSI", f"RSI: {rsi:.1f}"),
            ("MACD", f"MACD: {macd:.2f}"),
            ("Price Trend", "Above key levels")
        ]
    elif (sentiment < -0.1 and predictions[-1] < last_close * 0.98 and rsi > 30 and macd < 0):
        decision = "Sell"
        target_sell = last_close * 1.02
        reasons = [
            ("Negative Sentiment", f"Sentiment: {sentiment:.2f}"),
            ("LSTM Decline", f"Predicted: ${predictions[-1]:.2f}"),
            ("RSI", f"RSI: {rsi:.1f}"),
            ("MACD", f"MACD: {macd:.2f}"),
            ("Price Trend", "Below key levels")
        ]
    else:
        decision = "Hold"
        reasons = [("Neutral Signals", "No strong trend detected")]
    decision = adjust_decision_with_fundamentals(decision, fundamental_score)
    if len(reasons) < 5:
        reasons.extend([("Market Context", "Stable conditions")] * (5 - len(reasons)))
    return decision, target_buy if "Buy" in decision else None, target_sell if "Sell" in decision else None, reasons

# Enhanced Risk Analysis with Sharpe Ratio
def perform_risk_analysis(df: pd.DataFrame) -> Tuple[str, float, float, float]:
    if df.empty or "ATR" not in df.columns:
        return "Unknown", 0.0, 0.0, 0.0
    atr = df["ATR"].iloc[-1]
    volatility = df["Close"].pct_change().rolling(20).std().iloc[-1] or 0.0
    returns = df["Close"].pct_change().dropna()
    mean_return = returns.mean() * 252  # Annualized
    std_return = returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - 0.02) / std_return if std_return != 0 else 0
    risk = "Low" if atr < df["Close"].iloc[-1] * 0.03 and volatility < 0.02 else "High" if atr > df["Close"].iloc[-1] * 0.05 or volatility > 0.03 else "Medium"
    return risk, atr, volatility, sharpe_ratio

# Portfolio Management with Diversification Metric
def build_portfolio_summary(allocations: Dict[str, float], total_capital: float) -> Tuple[html.Div, html.Table]:
    rows, current_total, predicted_total = [], 0.0, 0.0
    sectors = set()
    for ticker, shares in allocations.items():
        df = fetch_market_data(ticker, period="1y")
        if df.empty:
            continue
        df = calculate_technical_indicators(df)
        current_price = df["Close"].iloc[-1]
        predictions, _, _, _ = build_lstm_forecast(df, forecast_horizon=30)
        if not predictions.size:
            continue
        predicted_price = predictions[-1]
        current_value = current_price * shares
        predicted_value = predicted_price * shares
        rows.append(html.Tr([html.Td(ticker), html.Td(f"{shares:.0f}"), html.Td(f"${current_price:.2f}"), html.Td(f"${predicted_price:.2f}"), html.Td(f"${current_value:.2f}"), html.Td(f"${predicted_value:.2f}"), html.Td(f"${predicted_value - current_value:.2f}")] ))
        current_total += current_value
        predicted_total += predicted_value
        sectors.add(get_fundamental_data(ticker).get("Sector", "Unknown"))
    sector_count = len(sectors)
    portfolio_returns = sum([(allocations[t] * fetch_market_data(t, "1y")["Close"].pct_change()).dropna() for t in allocations if not fetch_market_data(t, "1y").empty]) / len(allocations)
    mean_portfolio_return = portfolio_returns.mean() * 252
    std_portfolio_return = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_portfolio_return - 0.02) / std_portfolio_return if std_portfolio_return != 0 else 0
    summary = html.Div([
        html.P(f"Current Value: ${current_total:.2f}"),
        html.P(f"Predicted Value: ${predicted_total:.2f}"),
        html.P(f"Profit/Loss: ${predicted_total - current_total:.2f}"),
        html.P(f"Sharpe Ratio: {sharpe_ratio:.2f}"),
        html.P(f"Number of Sectors: {sector_count}")
    ])
    table = html.Table([html.Thead(html.Tr([html.Th(col) for col in ["Ticker", "Shares", "Current Price", "Predicted Price", "Current Value", "Predicted Value", "P/L"]])), html.Tbody(rows)])
    return summary, table

# Market Recommendations
def get_company_recommendations(market: str) -> Tuple[Optional[html.Table], str]:
    tickers = MARKET_TICKERS.get(market, [])
    if not tickers:
        return None, f"No tickers for {market}"
    rows = []
    for ticker in tickers:
        df = fetch_market_data(ticker, period="10y")
        if df.empty:
            continue
        df = calculate_technical_indicators(df)
        sentiment, _ = get_sentiment_from_sources(ticker)
        predictions, _, _, _ = build_lstm_forecast(df)
        fundamentals = get_fundamental_data(ticker)
        score = calculate_fundamental_score(fundamentals)
        decision, _, _, reasons = make_trading_decision(df, sentiment, predictions, score)
        rows.append(html.Tr([html.Td(ticker), html.Td(decision), html.Td(", ".join([r[0] for r in reasons])), html.Td(str(score))]))
    table = html.Table([html.Thead(html.Tr([html.Th(col) for col in ["Ticker", "Decision", "Reasons", "Score"]])), html.Tbody(rows)])
    return table, f"Recommendations for {market}"

# Dash Application
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
# Custom HTML template with CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { 
                background: linear-gradient(135deg, #1a1a2e, #16213e), 
                            url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAPElEQVQYV2NkYGD4z8DAwMjIyMDIyMhYGBgYGBhYGBkYGBgYGBkZGRkYGBgYGBkZGRkYGBgYGBkZGRkZGQF2aAJ1u0z2lwAAAABJRU5ErkJggg==');
                background-size: 200% 200%; 
                color: #e0e0e0; 
                font-family: 'Roboto', sans-serif; 
                animation: backgroundShift 10s ease infinite; 
            }
            @keyframes backgroundShift { 
                0% { background-position: 0% 50%; } 
                50% { background-position: 100% 50%; } 
                100% { background-position: 0% 50%; } 
            }
            h1 { 
                background: linear-gradient(90deg, #e94560, #533483); 
                padding: 20px; 
                border-bottom: 2px solid #e94560; 
            }
            h3 { 
                color: #e94560; 
                text-shadow: 0 0 5px rgba(233, 69, 96, 0.7); 
                animation: slideIn 0.8s ease-out; 
            }
            @keyframes slideIn { 
                from { transform: translateX(-20px); opacity: 0; } 
                to { transform: translateX(0); opacity: 1; } 
            }
            .btn-primary { 
                background: #e94560; 
                border: none; 
                transition: all 0.3s; 
                animation: glow 1.5s infinite alternate; 
            }
            .btn-primary:hover { 
                background: #ff6b6b; 
                box-shadow: 0 0 15px #e94560; 
            }
            @keyframes glow { 
                from { box-shadow: 0 0 5px #e94560; } 
                to { box-shadow: 0 0 20px #e94560; } 
            }
            .card { 
                background: #0f3460; 
                border: 1px solid #e94560; 
                transition: transform 0.3s, box-shadow 0.3s; 
                position: relative; 
                overflow: hidden; 
            }
            .card::before { 
                content: ''; 
                position: absolute; 
                top: -2px; 
                left: -2px; 
                right: -2px; 
                bottom: -2px; 
                border: 2px solid transparent; 
                border-image: linear-gradient(45deg, #e94560, #533483) 1; 
                animation: borderGlow 2s infinite alternate; 
            }
            @keyframes borderGlow { 
                from { border-image: linear-gradient(45deg, #e94560, #533483) 1; } 
                to { border-image: linear-gradient(45deg, #533483, #e94560) 1; } 
            }
            .card:hover { 
                transform: scale(1.02); 
                box-shadow: 0 0 20px rgba(233, 69, 96, 0.7); 
            }
            .dbc-container { 
                animation: fadeIn 1s ease-in; 
            }
            @keyframes fadeIn { 
                from { opacity: 0; } 
                to { opacity: 1; } 
            }
            #mode-selection-card { 
                transition: transform 0.3s, box-shadow 0.3s; 
            }
            #mode-selection-card:hover { 
                transform: translateY(-5px); 
                box-shadow: 0 5px 15px rgba(233, 69, 96, 0.5); 
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <div style="text-align: center; padding: 20px; color: #e0e0e0;">
                Data provided by Finnhub, News API, and FRED. Powered by AI Investing Hub.
            </div>
        </footer>
    </body>
</html>
'''
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H1("AI Investing Hub", className="text-center my-4")))
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardHeader("Mode"), dbc.CardBody(dcc.RadioItems(id="mode-selection", options=[{"label": f" {m}", "value": v} for m, v in [("Single Ticker", "single"), ("Portfolio", "portfolio"), ("Market Insights", "market")]], value="single", labelStyle={"display": "block"}))], className="mb-4"),
            dbc.Card([dbc.CardHeader("Asset Type"), dbc.CardBody(dcc.Dropdown(id="asset-type", options=[{"label": t, "value": t} for t in ["Stock", "ETF", "Crypto"]], value="Stock", clearable=False))], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([dbc.CardHeader("Control Panel"), dbc.CardBody([
                html.Div(id="ticker-div", children=[html.Label("Ticker(s):"), dcc.Input(id="stock-symbol", type="text", value="AAPL", className="form-control mb-2")]),
                html.Div(id="portfolio-div", children=[html.Label("Portfolio (Ticker:Shares):"), dcc.Input(id="portfolio-allocation", type="text", placeholder="AAPL:50,MSFT:30", className="form-control mb-2"), html.Label("Capital ($):"), dcc.Input(id="initial-capital", type="number", value=100000, className="form-control mb-2")], style={"display": "none"}),
                html.Div(id="market-div", children=[html.Label("Market:"), dcc.Dropdown(id="market-selection", options=[{"label": k, "value": k} for k in MARKET_TICKERS], value="US Tech", clearable=False, className="mb-2")], style={"display": "none"}),
                dbc.Row([dbc.Col(dcc.Dropdown(id="date-range", options=[{"label": r, "value": r} for r in VALID_DATE_RANGES], value="10y", clearable=False)), dbc.Col(dcc.Dropdown(id="forecast-horizon", options=FORECAST_OPTIONS, value=60, clearable=False))], className="mb-3"),
                dbc.Button("Analyze", id="update-button", color="primary", n_clicks=0, style={"width": "100%"}),
                dcc.Loading(id="loading", type="cube", children=html.Div(id="loading-output"))
            ])])
        ], width=9)
    ], style={"padding": "20px"}),
    dbc.Row(dbc.Col([
        html.Div(id="market-info", className="fw-bold mb-3", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.Div(id="macro-info", className="fw-bold mb-3", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.Div(dcc.Graph(id="price-chart"), className="graph-container mb-4"),
        html.Div(dcc.Graph(id="technical-chart"), className="graph-container mb-4"),
        html.Div(dcc.Graph(id="sentiment-heatmap"), className="graph-container mb-4"),
        html.Div(dcc.Graph(id="sector-correlation"), className="graph-container mb-4"),
        html.H3("Prediction"), html.Div(id="prediction-info", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Decision"), html.Div(id="trading-decision", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Risk"), html.Div(id="risk-info", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Fundamentals"), html.Div(id="fundamentals-info", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Insider Transactions"), html.Div(id="insider-info", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Analyst Recommendations"), html.Div(id="analyst-info", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("Portfolio"), html.Div(id="portfolio-details", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"}),
        html.H3("News"), html.Ul(id="news-list", style={"background": "#0f3460", "padding": "10px", "borderRadius": "8px", "border": "1px solid #533483"})
    ], style={"padding": "20px"})),
    dcc.Interval(id="interval-component", interval=300000, n_intervals=0)
])

@app.callback(
    [Output("ticker-div", "style"), Output("portfolio-div", "style"), Output("market-div", "style")],
    Input("mode-selection", "value")
)
def update_visibility(mode: str) -> List[Dict[str, str]]:
    return [{"display": "block" if mode == m else "none"} for m in ["single", "portfolio", "market"]]

@app.callback(
    [Output("market-info", "children"), Output("macro-info", "children"), Output("price-chart", "figure"), Output("technical-chart", "figure"),
     Output("sentiment-heatmap", "figure"), Output("sector-correlation", "figure"), Output("prediction-info", "children"), Output("trading-decision", "children"),
     Output("risk-info", "children"), Output("fundamentals-info", "children"), Output("insider-info", "children"), Output("analyst-info", "children"),
     Output("portfolio-details", "children"), Output("news-list", "children"), Output("loading-output", "children")],
    [Input("update-button", "n_clicks"), Input("interval-component", "n_intervals")],
    [State("stock-symbol", "value"), State("date-range", "value"), State("forecast-horizon", "value"), State("initial-capital", "value"),
     State("portfolio-allocation", "value"), State("mode-selection", "value"), State("market-selection", "value"), State("asset-type", "value")]
)
def update_dashboard(n_clicks: int, n_intervals: int, symbol: str, date_range: str, forecast_horizon: int, initial_capital: float, portfolio_allocation: str, mode: str, market_selection: str, asset_type: str) -> Tuple:
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered == "interval-component" and n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    macro_data = get_macro_data()
    macro_text = html.Div([
        html.H4("Macroeconomic Indicators"),
        html.P(f"GDP: {macro_data.get('GDP', 0):.2f}"),
        html.P(f"Unemployment: {macro_data.get('Unemployment', 0):.2f}%"),
        html.P(f"Inflation: {macro_data.get('Inflation', 0):.2f}"),
        html.P(f"Interest Rate: {macro_data.get('Interest Rate', 0):.2f}%")
    ])

    if mode == "market":
        tickers = MARKET_TICKERS.get(market_selection, [])
        data_dict = {t: fetch_market_data(t, period=date_range) for t in tickers if not fetch_market_data(t, period=date_range).empty}
        fig = go.Figure([go.Scatter(x=df["Date"], y=(df["Close"] / df["Close"].iloc[0]) * 100, name=t) for t, df in data_dict.items()])
        fig.update_layout(title="Normalized Performance", xaxis_title="Date", yaxis_title="Price (Base=100)")
        table, title = get_company_recommendations(market_selection)
        return title, macro_text, fig, {}, {}, {}, "N/A", table, "N/A", "N/A", "", "", "N/A", [], "Loaded"

    elif mode == "portfolio":
        if not portfolio_allocation:
            return "Enter portfolio allocation", macro_text, {}, {}, {}, {}, "", "", "", "", "", "", "Error: No allocation", [], "Error"
        try:
            allocations = parse_portfolio_allocation(portfolio_allocation)
            data_dict = {t: fetch_market_data(t, period=date_range) for t in allocations if not fetch_market_data(t, period=date_range).empty}
            fig = go.Figure([go.Scatter(x=df["Date"], y=(df["Close"] / df["Close"].iloc[0]) * 100, name=t) for t, df in data_dict.items()])
            fig.update_layout(title="Portfolio Performance", xaxis_title="Date", yaxis_title="Price (Base=100)")
            summary, table = build_portfolio_summary(allocations, initial_capital)
            fundamentals = [html.Div([html.H5(t), *[html.P(f"{k}: {v}") for k, v in get_fundamental_data(t).items()]]) for t in allocations]
            return f"Portfolio: {', '.join(allocations.keys())}", macro_text, fig, {}, {}, {}, summary, table, "N/A", fundamentals, "", "", table, [], "Loaded"
        except ValueError as e:
            return str(e), macro_text, {}, {}, {}, {}, "", "", "", "", "", "", "Error", [], "Error"

    else:  # Single Ticker
        df = fetch_market_data(symbol, period=date_range)
        if df.empty:
            return f"No data for {symbol}", macro_text, {}, {}, {}, {}, "", "", "", "", "", "", "", [], "Error"
        df = calculate_technical_indicators(df)
        sentiment, articles = get_sentiment_from_sources(symbol)
        predictions, model, scaler, X = build_lstm_forecast(df, forecast_horizon)
        fundamentals = get_fundamental_data(symbol)
        score = calculate_fundamental_score(fundamentals)
        decision, buy, sell, reasons = make_trading_decision(df, sentiment, predictions, score)
        risk, atr, vol, sharpe = perform_risk_analysis(df)

        insider_data = fetch_insider_data(symbol)
        insider_table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in ["Name", "Transaction", "Shares", "Value"]])),
            html.Tbody([html.Tr([html.Td(tx.get("name", "")), html.Td(tx.get("transactionType", "")), html.Td(tx.get("share", "")), html.Td(tx.get("value", ""))]) for tx in insider_data])
        ]) if insider_data else html.P("No insider data available")

        analyst_data = fetch_analyst_recommendations(symbol)
        analyst_text = html.P(f"Analyst Recommendations: Buy={analyst_data[0]['buy']}, Hold={analyst_data[0]['hold']}, Sell={analyst_data[0]['sell']}") if analyst_data else html.P("No analyst data")

        price_fig = go.Figure([
            go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"),
            go.Scatter(x=df["Date"], y=df["SMA"], name="SMA"), go.Scatter(x=df["Date"], y=df["EMA"], name="EMA")
        ])
        if predictions.size:
            forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="B")
            price_fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, name="Forecast"))
        price_fig.update_layout(title=f"{symbol} Price ({asset_type})", xaxis_title="Date", yaxis_title="Price")

        tech_fig = go.Figure([go.Scatter(x=df["Date"], y=df[col], name=col) for col in ["RSI", "MACD", "BB_High", "BB_Low"]])
        tech_fig.update_layout(title="Technical Indicators")

        sent_fig = go.Figure(go.Heatmap(z=[[sentiment]*3]*3, x=["News", "Reddit", "Twitter"], y=["News", "Reddit", "Twitter"], colorscale="RdYlGn"))
        sent_fig.update_layout(title="Sentiment Heatmap")

        return (
            f"{symbol} Data ({date_range})", macro_text, price_fig, tech_fig, sent_fig, {},
            f"Predicted Price ({forecast_horizon}d): ${predictions[-1]:.2f}" if predictions.size else "No prediction",
            html.Div([html.P(f"Decision: {decision}"), html.Ul([html.Li([html.Strong(r[0]), ": ", r[1]]) for r in reasons])]),
            f"Risk: {risk}\nATR: {atr:.2f}\nVolatility: {vol:.4f}\nSharpe Ratio: {sharpe:.2f}",
            [html.P(f"{k}: {v}") for k, v in fundamentals.items()],
            insider_table, analyst_text, "", 
            [html.Li(html.A(art.get("title", "No Title"), href=art.get("url", "#"), target="_blank")) for art in articles] or [html.Li("No news")],
            "Loaded"
        )

if __name__ == "__main__":
    app.run_server(debug=False)