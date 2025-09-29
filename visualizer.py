import matplotlib.pyplot as plt
import pandas as pd

def plot_price_and_rsi(price: pd.Series, rsi: pd.Series, signals: pd.Series, symbol: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)

    # Price chart
    ax1.plot(price.index, price, label=f"{symbol} Price", color="blue")
    ax1.set_title(f"{symbol} Stock Price")
    ax1.legend()

    # RSI chart
    ax2.plot(rsi.index, rsi, label="RSI", color="purple")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.axhline(70, color="red", linestyle="--")
    buy_signals = rsi[signals == "BUY"]
    sell_signals = rsi[signals == "SELL"]
    ax2.scatter(buy_signals.index, buy_signals, marker="^", color="green", label="BUY")
    ax2.scatter(sell_signals.index, sell_signals, marker="v", color="red", label="SELL")
    ax2.set_title("RSI with Buy/Sell Signals")
    ax2.legend()

    plt.tight_layout()
    plt.show()
