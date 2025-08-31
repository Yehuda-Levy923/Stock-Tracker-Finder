import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import our custom modules
from data_fetcher import DataFetcher
from sine_wave_detector import SineWaveDetector
from rsi_calculator import RSICalculator
from signal_generator import SignalGenerator
from visualizer import Visualizer
from utils import setup_logging, get_sp500_symbols


def main():
    """Main function to run the stock pattern detection system"""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Stock Pattern Detection System")

    # Initialize components
    data_fetcher = DataFetcher()
    sine_detector = SineWaveDetector()
    rsi_calculator = RSICalculator()
    signal_generator = SignalGenerator()
    visualizer = Visualizer()

    # Get S&P 500 symbols + major ETFs
    symbols = get_sp500_symbols()
    major_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT']
    all_symbols = symbols + major_etfs

    logger.info(f"Analyzing {len(all_symbols)} symbols")

    # Results storage
    results = []
    failed_symbols = []

    # Process each symbol
    for i, symbol in enumerate(all_symbols):
        try:
            logger.info(f"Processing {symbol} ({i + 1}/{len(all_symbols)})")

            # Fetch data (2 years to capture multiple cycles)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # ~2 years

            data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)

            if data is None or len(data) < 200:  # Need sufficient data
                logger.warning(f"Insufficient data for {symbol}")
                failed_symbols.append(symbol)
                continue

            # Calculate RSI
            data = rsi_calculator.calculate_rsi(data)

            # Detect sine wave patterns
            pattern_result = sine_detector.detect_pattern(data)

            if pattern_result is None:
                logger.info(f"No pattern detected for {symbol}")
                continue

            # Generate trading signal
            signal_result = signal_generator.generate_signal(
                data, pattern_result, data.iloc[-1]  # Current data point
            )

            # Store results
            result = {
                'symbol': symbol,
                'current_price': data['Close'].iloc[-1],
                'current_rsi': data['RSI'].iloc[-1],
                'pattern_strength': pattern_result['strength'],
                'cycle_phase': pattern_result['current_phase'],
                'signal': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'target_price': signal_result.get('target_price'),
                'stop_loss': signal_result.get('stop_loss'),
                'data': data,
                'pattern_data': pattern_result
            }

            results.append(result)
            logger.info(f"{symbol}: {signal_result['signal']} (Confidence: {signal_result['confidence']:.2f})")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            continue

    logger.info(f"Analysis complete. Processed {len(results)} symbols successfully")
    logger.info(f"Failed symbols: {len(failed_symbols)}")

    if not results:
        logger.warning("No results to display")
        return

    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer.create_summary_dashboard(results)
    visualizer.create_individual_charts(results[:10])  # Top 10 for individual charts

    # Save results to CSV
    save_results_to_csv(results)

    # Print summary
    print_summary(results)


def save_results_to_csv(results):
    """Save results to CSV file"""
    df_results = pd.DataFrame([
        {
            'Symbol': r['symbol'],
            'Current_Price': r['current_price'],
            'Current_RSI': r['current_rsi'],
            'Pattern_Strength': r['pattern_strength'],
            'Cycle_Phase': r['cycle_phase'],
            'Signal': r['signal'],
            'Confidence': r['confidence'],
            'Target_Price': r.get('target_price'),
            'Stop_Loss': r.get('stop_loss')
        }
        for r in results
    ])

    filename = f"stock_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def print_summary(results):
    """Print analysis summary"""
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    sell_signals = [r for r in results if r['signal'] == 'SELL']
    hold_signals = [r for r in results if r['signal'] == 'HOLD']

    print("\n" + "=" * 50)
    print("STOCK PATTERN ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total symbols analyzed: {len(results)}")
    print(f"BUY signals: {len(buy_signals)}")
    print(f"SELL signals: {len(sell_signals)}")
    print(f"HOLD signals: {len(hold_signals)}")

    if buy_signals:
        print("\nTOP BUY SIGNALS:")
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        for signal in buy_signals[:5]:
            print(f"  {signal['symbol']}: ${signal['current_price']:.2f} "
                  f"(Confidence: {signal['confidence']:.2f})")

    if sell_signals:
        print("\nTOP SELL SIGNALS:")
        sell_signals.sort(key=lambda x: x['confidence'], reverse=True)
        for signal in sell_signals[:5]:
            print(f"  {signal['symbol']}: ${signal['current_price']:.2f} "
                  f"(Confidence: {signal['confidence']:.2f})")


if __name__ == "__main__":
    main()