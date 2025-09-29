import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_analysis(price: pd.Series, rsi: pd.Series, sma: pd.Series, ema: pd.Series,
                  signals: pd.Series, symbol: str, save: bool = True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price + Moving Averages
    ax1.plot(price.index, price, label=f"{symbol} Price", color="blue")
    ax1.plot(sma.index, sma, label="SMA(20)", color="orange")
    ax1.plot(ema.index, ema, label="EMA(20)", color="purple")
    ax1.set_title(f"{symbol} Stock Price with SMA/EMA")
    ax1.legend()

    # RSI + Signals
    ax2.plot(rsi.index, rsi, label="RSI", color="darkgreen")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.axhline(70, color="red", linestyle="--")

    buy_signals = rsi[signals == "BUY"]
    sell_signals = rsi[signals == "SELL"]

    ax2.scatter(buy_signals.index, buy_signals, marker="^", color="green", label="BUY")
    ax2.scatter(sell_signals.index, sell_signals, marker="v", color="red", label="SELL")

    ax2.set_title("RSI with Buy/Sell Signals")
    ax2.legend()

    plt.tight_layout()

    if save:
        os.makedirs("charts", exist_ok=True)
        filepath = os.path.join("charts", f"{symbol}_analysis.png")
        plt.savefig(filepath)
        plt.close()
        print(f"✅ Chart saved: {filepath}")
    else:
        plt.show()
