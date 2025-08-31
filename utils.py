import logging
import os
import pandas as pd
import requests
from typing import List, Optional
import yfinance as yf


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Setup handlers
    handlers = [logging.StreamHandler()]  # Console handler

    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        handlers.append(logging.FileHandler(f'logs/stock_analysis_{timestamp}.log'))

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)


def get_sp500_symbols() -> List[str]:
    """
    Get list of S&P 500 stock symbols

    Returns:
        List of stock symbols
    """
    try:
        # Try to get from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()

        # Clean symbols (remove dots, etc.)
        cleaned_symbols = []
        for symbol in symbols:
            # Replace dots with dashes for Yahoo Finance compatibility
            clean_symbol = str(symbol).replace('.', '-')
            cleaned_symbols.append(clean_symbol)

        logging.getLogger(__name__).info(f"Retrieved {len(cleaned_symbols)} S&P 500 symbols")
        return cleaned_symbols

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to fetch S&P 500 list from Wikipedia: {str(e)}")

        # Fallback to hardcoded list of major stocks
        fallback_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'XOM', 'JPM', 'PG', 'HD', 'CVX', 'MA', 'PFE',
            'BAC', 'ABBV', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS',
            'ABT', 'DHR', 'VZ', 'ADBE', 'CMCSA', 'NKE', 'CRM', 'TXN', 'NEE',
            'ACN', 'RTX', 'QCOM', 'HON', 'PM', 'UPS', 'SBUX', 'T', 'MDT',
            'IBM', 'CAT', 'GS', 'AMGN', 'SPGI', 'GE', 'AXP', 'BLK', 'DE'
        ]

        logging.getLogger(__name__).info(f"Using fallback list with {len(fallback_symbols)} symbols")
        return fallback_symbols


def get_nasdaq100_symbols() -> List[str]:
    """
    Get list of NASDAQ 100 stock symbols

    Returns:
        List of stock symbols
    """
    try:
        # Get NASDAQ 100 from Yahoo Finance
        nasdaq100_etf = yf.Ticker("QQQ")
        holdings = nasdaq100_etf.get_holdings()

        if holdings is not None and not holdings.empty:
            symbols = holdings.index.tolist()
            return symbols[:100]  # Top 100

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to fetch NASDAQ 100 list: {str(e)}")

    # Fallback NASDAQ 100 symbols
    fallback_nasdaq = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'ADBE',
        'NFLX', 'PYPL', 'INTC', 'CRM', 'CMCSA', 'PEP', 'COST', 'CSCO',
        'AVGO', 'TXN', 'QCOM', 'TMUS', 'INTU', 'AMD', 'HON', 'SBUX',
        'GILD', 'MDLZ', 'ISRG', 'BKNG', 'ADP', 'VRTX', 'FISV', 'CSX',
        'ATVI', 'REGN', 'ILMN', 'MU', 'AMAT', 'ADI', 'LRCX', 'MELI'
    ]

    return fallback_nasdaq


def validate_symbols(symbols: List[str]) -> List[str]:
    """
    Validate stock symbols and remove invalid ones

    Args:
        symbols: List of stock symbols to validate

    Returns:
        List of valid symbols
    """
    valid_symbols = []
    logger = logging.getLogger(__name__)

    logger.info(f"Validating {len(symbols)} symbols...")

    # Test in batches to avoid overwhelming the API
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]

        for symbol in batch:
            try:
                ticker = yf.Ticker(symbol)
                # Try to get some basic info
                info = ticker.info

                # Check if we got meaningful data
                if info and ('symbol' in info or 'shortName' in info or 'longName' in info):
                    valid_symbols.append(symbol)
                else:
                    # Try getting recent data as backup validation
                    recent_data = ticker.history(period="5d")
                    if not recent_data.empty:
                        valid_symbols.append(symbol)
                    else:
                        logger.debug(f"Invalid symbol: {symbol}")

            except Exception as e:
                logger.debug(f"Error validating {symbol}: {str(e)}")
                continue

    logger.info(f"Validated {len(valid_symbols)} out of {len(symbols)} symbols")
    return valid_symbols


def get_market_data_summary():
    """
    Get overall market summary for major indices

    Returns:
        Dictionary with market data
    """
    logger = logging.getLogger(__name__)

    try:
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'VIX': '^VIX',
            'Russell 2000': '^RUT'
        }

        market_data = {}

        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")

                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0

                    market_data[name] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'symbol': symbol
                    }

            except Exception as e:
                logger.debug(f"Error getting data for {name}: {str(e)}")
                continue

        return market_data

    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        return {}


def calculate_portfolio_metrics(results: List[dict]) -> dict:
    """
    Calculate portfolio-level metrics from analysis results

    Args:
        results: List of analysis results

    Returns:
        Dictionary with portfolio metrics
    """
    try:
        if not results:
            return {}

        # Basic counts
        total_stocks = len(results)
        buy_signals = len([r for r in results if r['signal'] == 'BUY'])
        sell_signals = len([r for r in results if r['signal'] == 'SELL'])
        hold_signals = len([r for r in results if r['signal'] == 'HOLD'])

        # Confidence metrics
        confidences = [r['confidence'] for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence = len([c for c in confidences if c > 0.7])

        # Pattern strength metrics
        pattern_strengths = [r.get('pattern_strength', 0) for r in results]
        avg_pattern_strength = sum(pattern_strengths) / len(pattern_strengths) if pattern_strengths else 0
        strong_patterns = len([p for p in pattern_strengths if p > 0.6])

        # RSI metrics
        rsi_values = [r.get('current_rsi', 50) for r in results]
        avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50
        oversold_count = len([r for r in rsi_values if r <= 30])
        overbought_count = len([r for r in rsi_values if r >= 70])

        # Risk-reward metrics
        risk_rewards = [r.get('risk_reward_ratio', 0) for r in results if r.get('risk_reward_ratio', 0) > 0]
        avg_risk_reward = sum(risk_rewards) / len(risk_rewards) if risk_rewards else 0
        good_risk_reward = len([rr for rr in risk_rewards if rr >= 2.0])

        # Cycle phase distribution
        phases = [r.get('cycle_phase', 'unknown') for r in results]
        phase_distribution = {}
        for phase in set(phases):
            phase_distribution[phase] = phases.count(phase)

        return {
            'total_stocks': total_stocks,
            'signal_distribution': {
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': hold_signals
            },
            'confidence_metrics': {
                'average': avg_confidence,
                'high_confidence_count': high_confidence,
                'high_confidence_pct': high_confidence / total_stocks if total_stocks > 0 else 0
            },
            'pattern_metrics': {
                'average_strength': avg_pattern_strength,
                'strong_patterns_count': strong_patterns,
                'strong_patterns_pct': strong_patterns / total_stocks if total_stocks > 0 else 0
            },
            'rsi_metrics': {
                'average_rsi': avg_rsi,
                'oversold_count': oversold_count,
                'overbought_count': overbought_count
            },
            'risk_metrics': {
                'average_risk_reward': avg_risk_reward,
                'good_risk_reward_count': good_risk_reward,
                'trades_with_targets': len(risk_rewards)
            },
            'phase_distribution': phase_distribution
        }

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}


def filter_results(results: List[dict],
                   min_confidence: float = 0.0,
                   signals: List[str] = None,
                   min_pattern_strength: float = 0.0,
                   rsi_range: tuple = None) -> List[dict]:
    """
    Filter results based on various criteria

    Args:
        results: List of analysis results
        min_confidence: Minimum confidence level
        signals: List of signals to include ('BUY', 'SELL', 'HOLD')
        min_pattern_strength: Minimum pattern strength
        rsi_range: Tuple of (min_rsi, max_rsi)

    Returns:
        Filtered list of results
    """
    filtered = results.copy()

    # Filter by confidence
    if min_confidence > 0:
        filtered = [r for r in filtered if r.get('confidence', 0) >= min_confidence]

    # Filter by signals
    if signals:
        filtered = [r for r in filtered if r.get('signal') in signals]

    # Filter by pattern strength
    if min_pattern_strength > 0:
        filtered = [r for r in filtered if r.get('pattern_strength', 0) >= min_pattern_strength]

    # Filter by RSI range
    if rsi_range:
        min_rsi, max_rsi = rsi_range
        filtered = [r for r in filtered if min_rsi <= r.get('current_rsi', 50) <= max_rsi]

    return filtered


def save_trading_signals_csv(results: List[dict], filename: str = None):
    """
    Save trading signals to CSV in a trading-friendly format

    Args:
        results: Analysis results
        filename: Output filename
    """
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'trading_signals_{timestamp}.csv'

    try:
        # Prepare data for CSV
        csv_data = []
        for r in results:
            csv_data.append({
                'Symbol': r['symbol'],
                'Signal': r['signal'],
                'Confidence': f"{r['confidence']:.1%}",
                'Current_Price': f"${r['current_price']:.2f}",
                'Target_Price': f"${r.get('target_price', 0):.2f}" if r.get('target_price') else '',
                'Stop_Loss': f"${r.get('stop_loss', 0):.2f}" if r.get('stop_loss') else '',
                'Risk_Reward_Ratio': f"{r.get('risk_reward_ratio', 0):.2f}" if r.get('risk_reward_ratio') else '',
                'Current_RSI': f"{r['current_rsi']:.1f}",
                'Pattern_Strength': f"{r['pattern_strength']:.2f}",
                'Cycle_Phase': r.get('cycle_phase', ''),
                'Trend': r.get('trend', ''),
                'Position_Size': r.get('position_size', ''),
                'Reasoning': r.get('reasoning', ''),
                'Date_Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)

        logger = logging.getLogger(__name__)
        logger.info(f"Trading signals saved to {filename}")
        print(f"Trading signals saved to {filename}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving trading signals: {str(e)}")


def load_watchlist(filename: str) -> List[str]:
    """
    Load stock symbols from a watchlist file

    Args:
        filename: Path to watchlist file (txt or csv)

    Returns:
        List of stock symbols
    """
    try:
        symbols = []

        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            # Assume first column contains symbols
            symbols = df.iloc[:, 0].astype(str).str.upper().tolist()
        else:
            # Assume text file with one symbol per line
            with open(filename, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]

        # Clean symbols
        cleaned_symbols = []
        for symbol in symbols:
            # Remove any non-alphanumeric characters except hyphens and dots
            clean_symbol = ''.join(c for c in symbol if c.isalnum() or c in '.-')
            if clean_symbol:
                cleaned_symbols.append(clean_symbol)

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(cleaned_symbols)} symbols from {filename}")

        return cleaned_symbols

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading watchlist from {filename}: {str(e)}")
        return []


def create_config_template(filename: str = 'config.py'):
    """
    Create a configuration template file

    Args:
        filename: Configuration file name
    """
    config_content = '''"""
Configuration file for Stock Pattern Detection System
Modify these settings according to your preferences
"""

# Data fetching settings
DATA_PERIOD_DAYS = 730  # ~2 years of data
MIN_DATA_POINTS = 200   # Minimum data points required

# Pattern detection settings
PATTERN_TOLERANCE = 0.15           # 15% tolerance for pattern matching
SHORT_CYCLE_DAYS = 90              # ~3 months
LONG_CYCLE_DAYS = 270              # ~9 months
MIN_PATTERN_STRENGTH = 0.3         # Minimum pattern strength threshold

# RSI settings
RSI_PERIOD = 14                    # RSI calculation period
RSI_OVERSOLD_THRESHOLD = 30        # RSI oversold level
RSI_OVERBOUGHT_THRESHOLD = 70      # RSI overbought level

# Signal generation settings
MIN_CONFIDENCE = 0.6               # Minimum confidence for signals
RSI_WEIGHT = 0.4                   # Weight for RSI in signal calculation
PATTERN_WEIGHT = 0.6               # Weight for pattern in signal calculation

# Risk management settings
MAX_RISK_PER_TRADE = 0.02          # 2% maximum risk per trade
PROFIT_TARGET_RATIO = 2.0          # 2:1 reward:risk ratio

# Visualization settings
FIGURE_SIZE = (15, 10)             # Default figure size
DPI = 100                          # Figure resolution
MAX_INDIVIDUAL_CHARTS = 10         # Maximum individual charts to create

# Logging settings
LOG_LEVEL = 'INFO'                 # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_TO_FILE = True                 # Whether to log to file

# Symbol lists (customize as needed)
CUSTOM_WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    # Add your preferred stocks here
]

# Major ETFs to include
MAJOR_ETFS = [
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 
    'VEA', 'VWO', 'GLD', 'SLV', 'TLT'
]
'''

    try:
        with open(filename, 'w') as f:
            f.write(config_content)

        print(f"Configuration template created: {filename}")
        print("Please review and modify the settings as needed.")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating config template: {str(e)}")


def check_dependencies():
    """
    Check if all required dependencies are installed

    Returns:
        Tuple of (all_installed: bool, missing_packages: List[str])
    """
    required_packages = [
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'requests'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    all_installed = len(missing_packages) == 0

    if not all_installed:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")

    return all_installed, missing_packages


def print_analysis_header():
    """Print a nice header for the analysis"""
    header = """
    ╔══════════════════════════════════════════════════════════╗
    ║              STOCK PATTERN DETECTION SYSTEM              ║
    ║                                                          ║
    ║         Sine Wave Pattern Analysis with RSI              ║
    ║              3-Month Cycles & 9-Month Inversions         ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(header)


def print_analysis_footer(total_time: float, total_stocks: int):
    """
    Print analysis completion summary

    Args:
        total_time: Total execution time in seconds
        total_stocks: Number of stocks analyzed
    """
    footer = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                   ANALYSIS COMPLETE                      ║
    ║                                                          ║
    ║    Total Time: {total_time:>6.1f} seconds                          ║
    ║    Stocks Analyzed: {total_stocks:>3d}                              ║
    ║    Average Time per Stock: {total_time / total_stocks if total_stocks > 0 else 0:>4.1f} seconds            ║
    ║                                                          ║
    ║         Check generated files for detailed results       ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(footer)


# Performance monitoring utilities
class Timer:
    """Simple timer context manager for performance monitoring"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        logger = logging.getLogger(__name__)
        logger.debug(f"{self.description} completed in {elapsed:.2f} seconds")


def memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A (psutil not installed)"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Check dependencies
    all_installed, missing = check_dependencies()
    print(f"All dependencies installed: {all_installed}")

    # Test S&P 500 symbol fetching
    symbols = get_sp500_symbols()
    print(f"Retrieved {len(symbols)} S&P 500 symbols")

    # Print analysis header
    print_analysis_header()

    print("Utility functions test completed.")