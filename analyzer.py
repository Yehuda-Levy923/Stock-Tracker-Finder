# analyzer.py
import os
import requests
from io import StringIO
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from indicators import rsi, sma
from signal_generator import combined_signals, latest_recommendation

# -------------------------
# Get list of S&P 500 tickers
# -------------------------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/140.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch S&P 500 tickers: {response.status_code}")

    # Read tables from the HTML content
    tables = pd.read_html(StringIO(response.text))
    tickers = tables[0]['Symbol'].tolist()
    return tickers

# -------------------------
# Plot and save individual stock chart
# -------------------------
def save_chart(df: pd.DataFrame, symbol: str):
    os.makedirs("charts", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- PRICE + SMA ----
    ax1.plot(df.index, df["Close"], label="Close", color="blue")
    ax1.plot(df.index, df["SMA"], label="SMA(50)", color="orange")

    # BUY / SELL markers on price chart
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green", label="BUY", alpha=0.9)
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red", label="SELL", alpha=0.9)

    ax1.set_title(f"{symbol} Price & Signals")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # ---- RSI ----
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    file_path = os.path.join("charts", f"{symbol}.png")
    plt.savefig(file_path)
    plt.close()

# -------------------------
# Plot and save summary chart
# -------------------------
def save_summary_chart(results):
    os.makedirs("charts", exist_ok=True)

    df = pd.DataFrame(results, columns=["Symbol", "Signal"])
    counts = df["Signal"].value_counts()

    plt.figure(figsize=(6, 5))
    counts.plot(kind="bar", color=["green", "red", "gray"])
    plt.title("S&P 500 Signal Summary")
    plt.ylabel("Number of Stocks")
    plt.xticks(rotation=0)
    plt.tight_layout()

    file_path = os.path.join("charts", "sp500_summary.png")
    plt.savefig(file_path)
    plt.close()

# -------------------------
# Analyze a single stock
# -------------------------
def analyze_stock(symbol: str, period: str = "6mo", make_chart: bool = True):
    print(f"\n🔍 Analyzing {symbol}...")
    df = yf.download(symbol, period=period, interval="1d", progress=False)

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return None, None

    df["RSI"] = rsi(df["Close"])
    df["SMA"] = sma(df["Close"], 50)
    df["Signal"] = combined_signals(df["Close"], df["RSI"], df["SMA"])

    rec = latest_recommendation(df["Signal"])
    print(f"📊 Latest recommendation for {symbol}: {rec}")

    if make_chart:
        save_chart(df, symbol)

    return df, rec

# -------------------------
# Fast batch analysis for S&P 500
# -------------------------
def analyze_sp500_fast():
    tickers = get_sp500_tickers()
    print("\n📈 Downloading all S&P 500 stocks in batch...")
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", progress=False)

    results = []

    for symbol in tickers:
        try:
            df = data[symbol].dropna()
            if df.empty:
                continue

            df["RSI"] = rsi(df["Close"])
            df["SMA"] = sma(df["Close"], 50)
            df["Signal"] = combined_signals(df["Close"], df["RSI"], df["SMA"])
            rec = latest_recommendation(df["Signal"])
            results.append((symbol, rec))

            # Save chart only if BUY or SELL
            if rec in ["BUY", "SELL"]:
                save_chart(df, symbol)

        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")

    # Save the summary bar chart
    save_summary_chart(results)

    return results

# -------------------------
# Find all BUY signals in S&P 500
# -------------------------
def find_buys():
    results = analyze_sp500_fast()
    buys = [s for s in results if s[1] == "BUY"]

    print("\n✅ Recommended BUYs in the S&P 500:")
    if not buys:
        print("No BUY signals found today.")
    else:
        for symbol, rec in buys:
            print(f"- {symbol}")
