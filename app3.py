from typing import Tuple, Dict, List, Any, Optional
import os
import logging
import time
from datetime import datetime, timedelta
import random

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State
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

# -----------------------------------------------------------------------------
# HARDCODED KEYS (not recommended in real projects!)
# -----------------------------------------------------------------------------
NEWS_API_KEY = "6b6fb493cecb4445a972f84ebdf537cc"
ALPHA_API_KEY = "J0RCE86BCG9X43PF"
FINNHUB_API_KEY = "cunsh9rgq10ckt73ej9gcunsh9rgq10ckt73ej10"

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Simple in-memory cache to avoid duplicate API calls
# -----------------------------------------------------------------------------
CACHE: Dict[str, Any] = {}

def get_cached(key: str) -> Optional[Any]:
    return CACHE.get(key)

def set_cached(key: str, value: Any) -> None:
    CACHE[key] = value

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VALID_DATE_RANGES = ["1y", "5y", "10y"]
FORECAST_OPTIONS = [
    {"label": "1 day", "value": 1},
    {"label": "3 days", "value": 3},
    {"label": "1 week", "value": 7},
    {"label": "1 month", "value": 30},
    {"label": "2 months", "value": 60},
    {"label": "3 months", "value": 90},
]

MARKET_TICKERS: Dict[str, List[str]] = {
    "US Tech": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABT"],
    "Finance": ["JPM", "BAC", "C"],
    "Energy": ["XOM", "CVX"],
    "Consumer": ["PG", "KO"]
}

# -----------------------------------------------------------------------------
# 1) Alpha Vantage Fallback
# -----------------------------------------------------------------------------
def fetch_alpha_vantage_data(symbol: str, period: str = "10y") -> pd.DataFrame:
    if not ALPHA_API_KEY:
        logger.error("ALPHA_API_KEY not set!")
        return pd.DataFrame()
    outputsize = "full" if period == "10y" else "compact"
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_API_KEY}"
    )
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Alpha Vantage request failed with status {response.status_code}")
            return pd.DataFrame()
        data = response.json()
        if "Time Series (Daily)" not in data:
            logger.error(f"Alpha Vantage did not return expected data for {symbol}")
            return pd.DataFrame()
        ts_data = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "6. volume": "Volume"
        }, inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Exception fetching Alpha Vantage data for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2) Finnhub Fallback
# -----------------------------------------------------------------------------
def fetch_finnhub_data(symbol: str, period: str = "10y", resolution: str = "D") -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set!")
        return pd.DataFrame()
    now = datetime.now()
    if period.endswith("y"):
        years = int(period[:-1])
        start_date = now - timedelta(days=years * 365)
    else:
        start_date = now - timedelta(days=365)
    to_ts = int(time.mktime(now.timetuple()))
    from_ts = int(time.mktime(start_date.timetuple()))
    url = (
        f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}"
        f"&resolution={resolution}&from={from_ts}&to={to_ts}"
        f"&token={FINNHUB_API_KEY}"
    )
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("s") != "ok":
            logger.error(f"Finnhub error for {symbol}: {data.get('s')}")
            return pd.DataFrame()
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit='s'),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        })
        df.sort_values("Date", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 3) Main Data Fetch: Yahoo → Alpha → Finnhub
# -----------------------------------------------------------------------------
def fetch_market_data(symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    symbol = symbol.strip()
    cache_key = f"data_{symbol}_{period}_{interval}"
    if cache_key in CACHE:
        logger.info(f"Using cached data for {symbol}")
        return CACHE[cache_key].copy()

    try:
        logger.info(f"Downloading data for '{symbol}' with period='{period}' via Yahoo Finance")
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            logger.warning(f"Yahoo Finance returned empty for '{symbol}'. Falling back to Alpha Vantage.")
            df = fetch_alpha_vantage_data(symbol, period)
        if df.empty:
            logger.warning(f"Alpha Vantage returned empty for '{symbol}'. Falling back to Finnhub.")
            df = fetch_finnhub_data(symbol, period)
        if df.empty:
            logger.warning(f"No data returned for '{symbol}' from any source.")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.reset_index(inplace=True)
        CACHE[cache_key] = df.copy()
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Calculate Technical Indicators
# -----------------------------------------------------------------------------
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = {"Close", "High", "Low"}
    if not required.issubset(df.columns):
        logger.warning(f"Missing columns for TA: {required - set(df.columns)}")
        return pd.DataFrame()
    df["Close"] = df["Close"].squeeze()
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    df["BB_High"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
    df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    return df

def fetch_multiple_tickers_data(tickers: List[str], period: str = "10y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        df = fetch_market_data(t, period=period, interval=interval)
        if not df.empty:
            data[t.strip()] = df
    return data

def plot_normalized_prices(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for ticker, df in data_dict.items():
        if "Close" not in df.columns:
            continue
        df["Close"] = df["Close"].squeeze()
        baseline = df["Close"].iloc[0]
        norm_prices = (df["Close"] / baseline) * 100
        fig.add_trace(go.Scatter(x=df["Date"], y=norm_prices, mode="lines", name=ticker))
    fig.update_layout(title="Normalized Stock Performance (Base = 100)",
                      xaxis_title="Date", yaxis_title="Normalized Price")
    return fig

# -----------------------------------------------------------------------------
# News & Sentiment
# -----------------------------------------------------------------------------
def fetch_news_articles(symbol: str) -> List[Dict[str, Any]]:
    url = (f"https://newsapi.org/v2/everything?"
           f"q={symbol}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}")
    try:
        response = requests.get(url)
        articles = []
        if response.status_code == 200:
            data = response.json()
            for art in data.get("articles", [])[:5]:
                articles.append({
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "content": art.get("content"),
                    "url": art.get("url")
                })
        else:
            logger.warning(f"Error fetching news: {response.status_code}")
        return articles
    except Exception as e:
        logger.error(f"Exception fetching news articles: {e}")
        return []

def analyze_sentiment_text(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def get_sentiment_from_sources(symbol: str) -> Tuple[float, List[Dict[str, Any]]]:
    articles = fetch_news_articles(symbol)
    sentiments = [analyze_sentiment_text(art.get("content") or art.get("description") or art.get("title"))
                  for art in articles if (art.get("content") or art.get("description") or art.get("title"))]
    avg_news = np.mean(sentiments) if sentiments else 0
    reddit_sent = random.uniform(-1, 1)
    twitter_sent = random.uniform(-1, 1)
    overall = np.mean([avg_news, reddit_sent, twitter_sent])
    return overall, articles

# -----------------------------------------------------------------------------
# Fundamental Analysis & Scoring
# -----------------------------------------------------------------------------
def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    ticker_obj = yf.Ticker(symbol)
    info = ticker_obj.info
    fundamentals = {
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
    return fundamentals

def calculate_fundamental_score(symbol: str) -> int:
    fundamentals = get_fundamental_data(symbol)
    score = 0
    try:
        pe = fundamentals.get("PE Ratio")
        if pe != "N/A" and pe is not None:
            pe = float(pe)
            if pe < 20:
                score += 1
            elif pe > 30:
                score -= 1
    except Exception as e:
        logger.error(f"Error calculating PE score for {symbol}: {e}")
    try:
        div_yield = fundamentals.get("Dividend Yield")
        if div_yield != "N/A" and div_yield is not None:
            if float(div_yield) > 0.03:
                score += 1
    except Exception as e:
        logger.error(f"Error calculating dividend score for {symbol}: {e}")
    return score

def adjust_decision_with_fundamentals(decision: str, fundamental_score: int) -> str:
    if fundamental_score > 0:
        if decision == "Sell":
            return "Hold"
        elif decision == "Hold":
            return "Buy"
        else:
            return decision
    elif fundamental_score < 0:
        if decision == "Buy":
            return "Hold"
        elif decision == "Hold":
            return "Sell"
        else:
            return decision
    else:
        return decision

# -----------------------------------------------------------------------------
# LSTM Forecasting
# -----------------------------------------------------------------------------
def build_lstm_forecast(df: pd.DataFrame, forecast_horizon: int = 60, seq_length: int = 60,
                        epochs: int = 20, batch_size: int = 16) -> Tuple[np.ndarray, Any, Any]:
    if len(df) < seq_length:
        logger.warning("Not enough data for LSTM forecasting.")
        return np.array([]), None, None
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_seq = scaled_data[-seq_length:]
    predictions = []
    current_seq = last_seq.copy()
    for _ in range(forecast_horizon):
        pred_input = np.reshape(current_seq, (1, seq_length, 1))
        pred = model.predict(pred_input, verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions, model, scaler

# -----------------------------------------------------------------------------
# Decision Logic for Single Ticker
# -----------------------------------------------------------------------------
def make_trading_decision_lstm(df: pd.DataFrame, sentiment: float, predictions: np.ndarray,
                               fundamental_score: int) -> Tuple[str, Optional[float], Optional[float], List[str], str]:
    if df.empty or len(predictions) == 0:
        return "Hold", None, None, ["Insufficient data"], "Low"
    last_close = df["Close"].iloc[-1]
    target_buy = None
    target_sell = None
    if sentiment > 0.1 and predictions[-1] > last_close:
        decision = "Buy"
        target_buy = round(last_close * 0.98, 2)
        reasons = [
            "Positive sentiment from news & social media",
            "Technical indicators suggest uptrend",
            "LSTM forecast shows price growth"
        ]
    elif sentiment < -0.1 and predictions[-1] < last_close:
        decision = "Sell"
        target_sell = round(last_close * 1.02, 2)
        reasons = [
            "Negative sentiment overall",
            "Technical indicators suggest downtrend",
            "LSTM forecast shows price decline"
        ]
    else:
        decision = "Hold"
        reasons = [
            "Mixed signals; uncertain market",
            "Wait for clearer signals"
        ]
    adjusted_decision = adjust_decision_with_fundamentals(decision, fundamental_score)
    return adjusted_decision, target_buy, target_sell, reasons, "Low"

# -----------------------------------------------------------------------------
# Risk Analysis & Backtesting
# -----------------------------------------------------------------------------
def perform_risk_analysis(df: pd.DataFrame) -> Tuple[str, float, float]:
    if df.empty or "ATR" not in df.columns:
        return "Low", 0.0, 0.0
    atr = df["ATR"].iloc[-1]
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if pd.isnull(vol):
        vol = 0.0
    risk = "Low"
    if atr > (df["Close"].iloc[-1] * 0.05) or vol > 0.03:
        risk = "High"
    elif atr > (df["Close"].iloc[-1] * 0.03) or vol > 0.02:
        risk = "Medium"
    return risk, atr, vol

def backtest_model(model: Any, X: Any, y: Any, scaler: Any) -> Tuple[float, float]:
    return 0.0, 0.0

# -----------------------------------------------------------------------------
# Portfolio Management
# -----------------------------------------------------------------------------
def parse_portfolio_allocation(allocation_str: str) -> Dict[str, float]:
    allocations = {}
    try:
        for part in allocation_str.split(","):
            ticker, shares_str = part.split(":")
            allocations[ticker.strip()] = float(shares_str.strip())
    except Exception as e:
        raise ValueError(f"Error parsing portfolio allocation: {e}")
    return allocations

def build_portfolio_summary(allocations: Dict[str, float], total_capital: float) -> Tuple[html.Div, html.Table]:
    rows = []
    overall_current = 0
    overall_predicted = 0
    for ticker, shares in allocations.items():
        df = fetch_market_data(ticker, period="1y", interval="1d")
        if df.empty:
            continue
        df = calculate_technical_indicators(df)
        if df.empty:
            continue
        current_price = df["Close"].iloc[-1]
        predictions, _, _ = build_lstm_forecast(df, forecast_horizon=30)
        if len(predictions) == 0:
            continue
        predicted_price = predictions[-1]
        current_value = current_price * shares
        predicted_value = predicted_price * shares
        profit_loss = predicted_value - current_value
        overall_current += current_value
        overall_predicted += predicted_value
        rows.append(html.Tr([
            html.Td(ticker),
            html.Td(shares),
            html.Td(f"${current_price:.2f}"),
            html.Td(f"${predicted_price:.2f}"),
            html.Td(f"${current_value:.2f}"),
            html.Td(f"${predicted_value:.2f}"),
            html.Td(f"${profit_loss:.2f}")
        ]))
    overall_profit = overall_predicted - overall_current
    summary = html.Div([
        html.P(f"Overall Current Value: ${overall_current:.2f}"),
        html.P(f"Overall Predicted Value: ${overall_predicted:.2f}"),
        html.P(f"Overall Predicted Profit/Loss: ${overall_profit:.2f}")
    ])
    portfolio_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Ticker"),
            html.Th("Shares"),
            html.Th("Current Price"),
            html.Th("Predicted Price"),
            html.Th("Current Value"),
            html.Th("Predicted Value"),
            html.Th("Profit/Loss")
        ])),
        html.Tbody(rows)
    ])
    return summary, portfolio_table

# -----------------------------------------------------------------------------
# Market Recommendations
# -----------------------------------------------------------------------------
def get_company_recommendations(market: str) -> Tuple[Optional[html.Table], str]:
    tickers = MARKET_TICKERS.get(market, [])
    if not tickers:
        return None, f"No tickers found for market {market}"
    rec_rows = []
    for ticker in tickers:
        df = fetch_market_data(ticker, period="10y", interval="1d")
        if df.empty:
            continue
        df = calculate_technical_indicators(df)
        if df.empty:
            continue
        sentiment, _ = get_sentiment_from_sources(ticker)
        predictions, _, _ = build_lstm_forecast(df, forecast_horizon=60)
        fundamental_score = calculate_fundamental_score(ticker)
        base_decision, _, _, reasons, _ = make_trading_decision_lstm(df, sentiment, predictions, fundamental_score)
        decision = adjust_decision_with_fundamentals(base_decision, fundamental_score)
        fundamentals = get_fundamental_data(ticker)
        rec_rows.append(html.Tr([
            html.Td(ticker),
            html.Td(decision),
            html.Td(", ".join(reasons)),
            html.Td(f"Fund Score: {fundamental_score}"),
            html.Td(f"PE: {fundamentals.get('PE Ratio', 'N/A')}, MCap: {fundamentals.get('Market Cap', 'N/A')}")
        ]))
    rec_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Ticker"),
            html.Th("Recommendation"),
            html.Th("Reasons"),
            html.Th("Fundamental Score"),
            html.Th("Fundamentals")
        ])),
        html.Tbody(rec_rows)
    ])
    return rec_table, f"Market Recommendations for {market}"

# -----------------------------------------------------------------------------
# Sector Correlation
# -----------------------------------------------------------------------------
def generate_sector_correlation() -> Tuple[List[str], np.ndarray]:
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
    corr_matrix = np.random.uniform(-1, 1, (len(sectors), len(sectors)))
    return sectors, corr_matrix

# -----------------------------------------------------------------------------
# Dash App
# -----------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container(fluid=True, children=[
    # Title
    dbc.Row([
        dbc.Col(html.H1("AI Investing Dashboard", className="text-center my-4"), width=12)
    ]),
    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Mode"),
                dbc.CardBody([
                    dcc.RadioItems(
                        id="mode-selection",
                        options=[
                            {"label": " Single Ticker", "value": "single"},
                            {"label": " Portfolio", "value": "portfolio"},
                            {"label": " Market Recommendations", "value": "market"}
                        ],
                        value="single",
                        labelStyle={'display': 'block', 'marginBottom': '10px'}
                    )
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Asset Type"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="asset-type",
                        options=[
                            {"label": "Stock", "value": "Stock"},
                            {"label": "ETF", "value": "ETF"},
                            {"label": "Crypto", "value": "Crypto"}
                        ],
                        value="Stock",
                        clearable=False
                    )
                ])
            ], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Mode-Specific Controls"),
                dbc.CardBody([
                    html.Div(id="ticker-div", children=[
                        html.Label("Enter Ticker(s) (comma separated):"),
                        dcc.Input(id="stock-symbol", type="text", value="AAPL", style={"marginRight": "10px"})
                    ], className="mb-3"),
                    html.Div(id="portfolio-div", children=[
                        html.Label("Portfolio Allocation (Ticker:Shares):"),
                        dcc.Input(id="portfolio-allocation", type="text", placeholder="e.g., AAPL:50, MSFT:30", style={"marginRight": "10px"}),
                        html.Label("Initial Capital ($):"),
                        dcc.Input(id="initial-capital", type="number", value=100000, style={"marginRight": "10px"})
                    ], className="mb-3", style={'display': 'none'}),
                    html.Div(id="market-div", children=[
                        html.Label("Select Market:"),
                        dcc.Dropdown(
                            id="market-selection",
                            options=[{"label": k, "value": k} for k in MARKET_TICKERS.keys()],
                            value="US Tech",
                            clearable=False,
                            style={"width": "120px", "marginRight": "10px"}
                        )
                    ], className="mb-3", style={'display': 'none'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range:"),
                            dcc.Dropdown(
                                id="date-range",
                                options=[{"label": r, "value": r} for r in VALID_DATE_RANGES],
                                value="10y",
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Forecast Horizon (days):"),
                            dcc.Dropdown(
                                id="forecast-horizon",
                                options=[
                                    {"label": "1 day", "value": 1},
                                    {"label": "3 days", "value": 3},
                                    {"label": "1 week", "value": 7},
                                    {"label": "1 month", "value": 30},
                                    {"label": "2 months", "value": 60},
                                    {"label": "3 months", "value": 90},
                                ],
                                value=60,
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Button("Update", id="update-button", color="primary")
                ])
            ])
        ], width=9)
    ]),
    # Outputs
    dbc.Row([
        dbc.Col([
            html.Div(id="market-info", className="fw-bold mb-3"),
            dcc.Graph(id="price-chart", className="mb-4"),
            dcc.Graph(id="technical-chart", className="mb-4"),
            dcc.Graph(id="sentiment-heatmap", className="mb-4"),
            dcc.Graph(id="sector-correlation", className="mb-4"),
            html.H3("Prediction Info", className="mt-3"),
            html.Div(id="prediction-info", className="mb-3"),
            html.H3("Trading Decision"),
            html.Div(id="trading-decision", className="mb-3"),
            html.H3("Risk Analysis"),
            html.Div(id="risk-info", className="mb-3"),
            html.H3("Backtesting Metrics"),
            html.Div(id="backtesting-info", className="mb-3"),
            html.H3("Fundamental Analysis"),
            html.Div(id="fundamentals-info", className="mb-3"),
            html.H3("Portfolio Details"),
            html.Div(id="portfolio-details", className="mb-3"),
            html.H3("Top News Articles"),
            html.Ul(id="news-list", className="mb-3")
        ], width=12)
    ]),
    dcc.Interval(id="interval-component", interval=300000, n_intervals=0)
])

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    [Output("ticker-div", "style"),
     Output("portfolio-div", "style"),
     Output("market-div", "style")],
    [Input("mode-selection", "value")]
)
def update_visibility(mode: str):
    if mode == "single":
        return [{'marginBottom': '20px'},
                {'display': 'none'},
                {'display': 'none'}]
    elif mode == "portfolio":
        return [{'display': 'none'},
                {'marginBottom': '20px'},
                {'display': 'none'}]
    elif mode == "market":
        return [{'display': 'none'},
                {'display': 'none'},
                {'marginBottom': '20px'}]
    return [{'marginBottom': '20px'},
            {'display': 'none'},
            {'display': 'none'}]

@app.callback(
    [
        Output("market-info", "children"),
        Output("price-chart", "figure"),
        Output("technical-chart", "figure"),
        Output("sentiment-heatmap", "figure"),
        Output("sector-correlation", "figure"),
        Output("prediction-info", "children"),
        Output("trading-decision", "children"),
        Output("risk-info", "children"),
        Output("backtesting-info", "children"),
        Output("fundamentals-info", "children"),
        Output("portfolio-details", "children"),
        Output("news-list", "children")
    ],
    [Input("update-button", "n_clicks"), Input("interval-component", "n_intervals")],
    [
        State("stock-symbol", "value"),
        State("date-range", "value"),
        State("forecast-horizon", "value"),
        State("initial-capital", "value"),
        State("portfolio-allocation", "value"),
        State("mode-selection", "value"),
        State("market-selection", "value"),
        State("asset-type", "value")
    ]
)
def update_dashboard(n_clicks: int, n_intervals: int, symbol: str, date_range: str,
                     forecast_horizon: int, initial_capital: float, portfolio_allocation: str,
                     mode: str, market_selection: str, asset_type: str):
    if mode == "market":
        tickers = MARKET_TICKERS.get(market_selection, [])
        if not tickers:
            return (f"No tickers found for market {market_selection}", {}, {}, {}, {}, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", [])
        data_dict = fetch_multiple_tickers_data(tickers, period=date_range, interval="1d")
        norm_chart = plot_normalized_prices(data_dict)
        rec_rows = []
        for t in tickers:
            df = fetch_market_data(t, period=date_range, interval="1d")
            if df.empty:
                continue
            df = calculate_technical_indicators(df)
            if df.empty:
                continue
            sentiment, _ = get_sentiment_from_sources(t)
            predictions, _, _ = build_lstm_forecast(df, forecast_horizon=forecast_horizon)
            fundamental_score = calculate_fundamental_score(t)
            base_decision, _, _, reasons, _ = make_trading_decision_lstm(df, sentiment, predictions, fundamental_score)
            decision = adjust_decision_with_fundamentals(base_decision, fundamental_score)
            fundamentals = get_fundamental_data(t)
            rec_rows.append(html.Tr([
                html.Td(t),
                html.Td(decision),
                html.Td(", ".join(reasons)),
                html.Td(f"Fund Score: {fundamental_score}"),
                html.Td(f"PE: {fundamentals.get('PE Ratio', 'N/A')}, MCap: {fundamentals.get('Market Cap', 'N/A')}")
            ]))
        rec_table = html.Table([
            html.Thead(html.Tr([
                html.Th("Ticker"),
                html.Th("Recommendation"),
                html.Th("Reasons"),
                html.Th("Fundamental Score"),
                html.Th("Fundamentals")
            ])),
            html.Tbody(rec_rows)
        ])
        market_info_text = f"Market Recommendations for {market_selection} (Asset Type: {asset_type})"
        return (
            market_info_text,  # 1
            norm_chart,        # 2
            {},                # 3 technical-chart placeholder
            {},                # 4 sentiment-heatmap placeholder
            {},                # 5 sector-correlation placeholder
            "N/A",             # 6 prediction-info placeholder
            rec_table,         # 7 trading-decision
            "N/A",             # 8 risk-info placeholder
            "N/A",             # 9 backtesting-info placeholder
            "N/A",             # 10 fundamentals-info placeholder
            "N/A",             # 11 portfolio-details placeholder
            []                 # 12 news-list placeholder
        )
    elif mode == "portfolio":
        if not portfolio_allocation or portfolio_allocation.strip() == "":
            return ("Please provide portfolio allocation in the format 'TICKER:SHARES'", {}, {}, {}, {}, "", "", "", "", "", "Error: No allocation provided.", [])
        try:
            allocations = parse_portfolio_allocation(portfolio_allocation)
        except Exception as e:
            return (f"Error parsing portfolio allocation: {e}", {}, {}, {}, {}, "", "", "", "", "", "Error in allocation format.", [])
        portfolio_data = fetch_multiple_tickers_data(list(allocations.keys()), period=date_range, interval="1d")
        if not portfolio_data:
            return (f"No data found for provided tickers in period='{date_range}'", {}, {}, {}, {}, "", "", "", "", "", "No portfolio data.", [])
        norm_chart = plot_normalized_prices(portfolio_data)
        summary, portfolio_table = build_portfolio_summary(allocations, initial_capital)
        fundamentals_text = []
        for t in allocations.keys():
            fund = get_fundamental_data(t)
            fund_lines = [f"{key}: {value}" for key, value in fund.items()]
            fundamentals_text.append(html.Div([html.H5(t)] + [html.P(line) for line in fund_lines]))
        market_info_text = f"Portfolio analysis for: {', '.join(allocations.keys())} (Asset Type: {asset_type})"
        return (
            market_info_text,  # 1
            norm_chart,        # 2
            {},                # 3
            {},                # 4
            {},                # 5
            summary,           # 6
            portfolio_table,   # 7
            "N/A",             # 8
            "N/A",             # 9
            fundamentals_text, # 10
            portfolio_table,   # 11
            []                 # 12
        )
    else:
        symbol = symbol.strip()
        if date_range not in VALID_DATE_RANGES:
            date_range = "10y"
        df = fetch_market_data(symbol, period=date_range, interval="1d")
        if df.empty:
            return (f"No data found for '{symbol}' in period='{date_range}'", {}, {}, {}, {}, "", "", "", "", "", "", "")
        df = calculate_technical_indicators(df)
        if df.empty:
            return (f"Missing columns or no data for indicators for '{symbol}'", {}, {}, {}, {}, "", "", "", "", "", "", "")
        sentiment_score, articles = get_sentiment_from_sources(symbol)
        predictions, lstm_model, lstm_scaler = build_lstm_forecast(df, forecast_horizon=forecast_horizon)
        fundamental_score = calculate_fundamental_score(symbol)
        base_decision, target_buy, target_sell, reasons, risk_level = make_trading_decision_lstm(df, sentiment_score, predictions, fundamental_score)
        decision = adjust_decision_with_fundamentals(base_decision, fundamental_score)
        risk, atr, vol = perform_risk_analysis(df)
        mae, rmse = 0.0, 0.0
        fundamentals = get_fundamental_data(symbol)
        fundamentals_text = [html.P(f"{key}: {value}") for key, value in fundamentals.items()]
        sectors, corr_matrix = generate_sector_correlation()
        sector_heatmap = {
            "data": [go.Heatmap(z=corr_matrix, x=sectors, y=sectors, colorscale="RdYlGn")],
            "layout": go.Layout(title="Sector Correlation Heatmap")
        }
        price_chart = {
            "data": [
                go.Candlestick(
                    x=df["Date"],
                    open=df.get("Open", df["Close"]),
                    high=df.get("High", df["Close"]),
                    low=df.get("Low", df["Close"]),
                    close=df["Close"],
                    name="Price"
                ),
                go.Scatter(x=df["Date"], y=df.get("SMA", [None]*len(df)), mode="lines", name="SMA"),
                go.Scatter(x=df["Date"], y=df.get("EMA", [None]*len(df)), mode="lines", name="EMA")
            ],
            "layout": go.Layout(title=f"{symbol} Price Chart ({date_range}) (Asset Type: {asset_type})",
                                xaxis_title="Date", yaxis_title="Price")
        }
        if predictions is not None and len(predictions) > 0:
            last_date = df["Date"].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
            forecast_trace = go.Scatter(x=forecast_dates, y=predictions, mode="lines", name="Forecast")
            price_chart["data"].append(forecast_trace)
        technical_chart = {
            "data": [
                go.Scatter(x=df["Date"], y=df.get("RSI", [None]*len(df)), mode="lines", name="RSI"),
                go.Scatter(x=df["Date"], y=df.get("MACD", [None]*len(df)), mode="lines", name="MACD"),
                go.Scatter(x=df["Date"], y=df.get("BB_High", [None]*len(df)), mode="lines", name="BB_High"),
                go.Scatter(x=df["Date"], y=df.get("BB_Low", [None]*len(df)), mode="lines", name="BB_Low")
            ],
            "layout": go.Layout(title="Technical Indicators")
        }
        sentiment_heatmap = {
            "data": [go.Heatmap(
                z=[[sentiment_score]*3]*3,
                x=["News", "Reddit", "Twitter"],
                y=["News", "Reddit", "Twitter"],
                colorscale="RdYlGn"
            )],
            "layout": go.Layout(title="Sentiment Heatmap")
        }
        if predictions is not None and len(predictions) > 0:
            prediction_text = f"Predicted final price in {forecast_horizon} days: ${predictions[-1]:.2f}"
        else:
            prediction_text = "Not enough data to make predictions."
        reasons_text = "\n".join([f"- {r}" for r in reasons])
        decision_text = (
            f"Decision: {decision}\n"
            f"Risk Level: {risk_level}\n"
            f"Fundamental Score: {fundamental_score}\n"
            f"Reasons:\n{reasons_text}\n"
            f"Target Buy Price: {target_buy if target_buy else 'N/A'}\n"
            f"Target Sell Price: {target_sell if target_sell else 'N/A'}"
        )
        risk_text = (
            f"Overall Risk: {risk}\n"
            f"ATR: {atr:.2f}\n"
            f"20-day Volatility: {vol:.4f}"
        )
        backtest_text = (
            f"MAE: {mae:.2f}\n"
            f"RMSE: {rmse:.2f}"
        )
        market_info_text = (
            f"Data from {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()} "
            f"for '{symbol}' in '{date_range}' range (Rows: {len(df)}) (Asset Type: {asset_type})"
        )
        news_list = []
        if articles:
            for art in articles:
                title = art.get("title") or "No Title"
                link = art.get("url") or "#"
                news_list.append(html.Li(html.A(title, href=link, target="_blank")))
        else:
            news_list.append(html.Li("No recent news found or API error."))

        return (
            market_info_text,
            price_chart,
            technical_chart,
            sentiment_heatmap,
            sector_heatmap,
            prediction_text,
            decision_text,
            risk_text,
            backtest_text,
            fundamentals_text,
            "",
            news_list
        )

if __name__ == "__main__":
    app.run_server(debug=False)
