📈 Stock Tracker Finder

A simple stock analysis tool built in Python that fetches market data, calculates technical indicators, and generates trading signals with visualizations.

This is a clean restart of the project — starting from scratch for clarity and simplicity.

🚀 Features

Fetches historical stock data using yfinance

Calculates RSI (Relative Strength Index)

Generates BUY / SELL signals based on RSI thresholds

Visualizes stock price + RSI with clear markers

📂 Project Structure
stock-tracker-finder/
│
├── main.py              # Entry point
├── data_fetcher.py      # Data fetching from Yahoo Finance
├── indicators.py        # Technical indicators (RSI, etc.)
├── signal_generator.py  # Buy/sell signal logic
├── visualizer.py        # Plotting functions
│
├── requirements.txt     # Dependencies
└── README.md            # Project docs

🛠 Installation

Clone the repo:

git clone https://github.com/Yehuda-Levy923/Stock-Tracker-Finder.git
cd Stock-Tracker-Finder


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Run the analysis on a stock (default: Apple AAPL):

python main.py


You’ll see:

A stock price chart

An RSI chart with BUY/SELL markers

📊 Example Output

(Example chart for Microsoft)


<img width="1034" height="745" alt="image" src="https://github.com/user-attachments/assets/2fcb3c6a-bc12-4fa6-a4ad-59468bdc9b09" />

🔮 Next Steps

Add moving averages (SMA/EMA)

Add support & resistance detection

Support multiple stocks at once

Backtesting strategy performance
