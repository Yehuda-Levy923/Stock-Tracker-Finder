import pandas as pd

def rsi_signals(rsi: pd.Series, lower: int = 30, upper: int = 70) -> pd.Series:
    signals = pd.Series(index=rsi.index, data="HOLD")
    signals[rsi < lower] = "BUY"
    signals[rsi > upper] = "SELL"
    return signals

def combined_signals(price: pd.Series, rsi: pd.Series, sma: pd.Series) -> pd.Series:
    """
    Strategy:
    - RSI < 30 → BUY (regardless of SMA, but stronger if price > SMA)
    - RSI > 70 → SELL (regardless of SMA, but stronger if price < SMA)
    - Otherwise HOLD
    """
    signals = pd.Series(index=price.index, data="HOLD")

    buy_condition = (rsi < 30)
    sell_condition = (rsi > 70)

    signals[buy_condition] = "BUY"
    signals[sell_condition] = "SELL"

    return signals

def latest_recommendation(signals: pd.Series) -> str:
    """
    Returns the most recent signal as recommendation.
    """
    if signals.dropna().empty:
        return "HOLD"
    return signals.iloc[-1]
