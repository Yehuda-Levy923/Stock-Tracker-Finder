import yfinance as yf
import pandas as pd

def fetch_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df
