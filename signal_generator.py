import pandas as pd

def rsi_signals(rsi: pd.Series, lower: int = 30, upper: int = 70) -> pd.Series:
    signals = pd.Series(index=rsi.index, data="HOLD")
    signals[rsi < lower] = "BUY"
    signals[rsi > upper] = "SELL"
    return signals
