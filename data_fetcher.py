import yfinance as yf
import pandas as pd
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class DataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol and date range

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None

            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns for {symbol}")
                return None

            # Clean the data
            data = self._clean_data(data)

            self.logger.debug(f"Fetched {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_multiple_stocks(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[
        str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}

        for symbol in symbols:
            data = self.fetch_stock_data(symbol, start_date, end_date)
            if data is not None:
                results[symbol] = data

        return results

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the stock data

        Args:
            data: Raw stock data

        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN values
        data = data.dropna()

        # Ensure all prices are positive
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]

        # Sort by date
        data = data.sort_index()

        # Add some basic derived columns
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Spread'] = (data['High'] - data['Low']) / data['Close']

        return data

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different price fields
            for price_field in ['regularMarketPrice', 'currentPrice', 'price']:
                if price_field in info and info[price_field]:
                    return float(info[price_field])

            # Fallback to recent history
            recent_data = ticker.history(period="1d")
            if not recent_data.empty:
                return recent_data['Close'].iloc[-1]

            return None

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get some basic info
            info = ticker.info

            # Check if we get meaningful data
            if 'symbol' in info or 'shortName' in info:
                return True

            # Try getting recent data
            recent_data = ticker.history(period="5d")
            return not recent_data.empty

        except:
            return False

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get basic information about a stock

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock info or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            stock_info = {
                'symbol': symbol,
                'name': info.get('shortName', info.get('longName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', None),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None)
            }

            return stock_info

        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            return None