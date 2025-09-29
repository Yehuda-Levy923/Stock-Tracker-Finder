from data_fetcher import fetch_data
from indicators import rsi
from signal_generator import rsi_signals
from visualizer import plot_price_and_rsi

def main():
    symbol = "MSFT"
    data = fetch_data(symbol, period="6mo", interval="1d")

    price = data["Close"]
    rsi_values = rsi(price)
    signals = rsi_signals(rsi_values)

    plot_price_and_rsi(price, rsi_values, signals, symbol)

if __name__ == "__main__":
    main()
