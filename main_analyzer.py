import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import our custom modules
from data_fetcher import DataFetcher
from ground_ceiling_detector import GroundCeilingDetector  # Updated import
from rsi_calculator import RSICalculator
from signal_generator import SignalGenerator  # Updated import
from visualizer import Visualizer
from utils import setup_logging, get_sp500_symbols, print_analysis_header, print_analysis_footer


def main():
    """Main function to run the Ground-Ceiling pattern detection system"""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Print header
    print_analysis_header_ground_ceiling()

    logger.info("Starting Ground-Ceiling Pattern Detection System")

    # Initialize components
    data_fetcher = DataFetcher()
    pattern_detector = GroundCeilingDetector(lookback_days=365, min_touch_count=2, level_tolerance=0.05)
    rsi_calculator = RSICalculator()
    signal_generator = SignalGenerator(min_confidence=0.6)
    visualizer = Visualizer()

    # Get symbols to analyze
    symbols = get_symbols_to_analyze()
    logger.info(f"Analyzing {len(symbols)} symbols")

    # Results storage
    results = []
    failed_symbols = []
    start_time = time.time()

    # Process each symbol
    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"Processing {symbol} ({i + 1}/{len(symbols)})")
            print(f"Analyzing {symbol}... ({i + 1}/{len(symbols)})", end=" ")

            # Fetch data (1+ years to capture ground/ceiling patterns)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)  # ~13 months

            data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)

            if data is None or len(data) < 200:  # Need sufficient data
                logger.warning(f"Insufficient data for {symbol}")
                failed_symbols.append(symbol)
                print("❌ Insufficient data")
                continue

            # Calculate RSI
            data = rsi_calculator.calculate_rsi(data)

            # Detect Ground-Ceiling pattern
            pattern_result = pattern_detector.detect_pattern(data)

            if pattern_result is None:
                logger.info(f"No Ground-Ceiling pattern detected for {symbol}")
                print("⚪ No pattern")
                continue

            # Generate trading signal
            current_data = data.iloc[-1]
            signal_result = signal_generator.generate_signal(data, pattern_result, current_data)

            # Generate complete trade plan
            trade_plan = signal_generator.generate_trade_plan(data, pattern_result, account_size=50000)

            # Get pattern summary
            pattern_summary = signal_generator.get_pattern_summary(pattern_result)

            # Backtest the pattern (last 60 days)
            backtest_result = signal_generator.backtest_pattern(data, pattern_result, lookback_days=60)

            # Store comprehensive results
            result = {
                'symbol': symbol,
                'current_price': pattern_result['current_price'],
                'current_rsi': data['RSI'].iloc[-1],
                'ground_level': pattern_result['current_ground'],
                'ceiling_level': pattern_result['current_ceiling'],
                'position_in_range': pattern_result['position_in_range'],
                'upside_potential': pattern_result['upside_potential'],
                'downside_risk': pattern_result['downside_risk'],
                'pattern_strength': pattern_result['strength'],
                'signal': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'target_price': signal_result.get('target_price'),
                'stop_loss': signal_result.get('stop_loss'),
                'risk_reward_ratio': pattern_result['upside_potential'] / max(pattern_result['downside_risk'], 0.1),
                'transition_level': pattern_result.get('old_ceiling_became_ground'),
                'pattern_zone': signal_result.get('pattern_zone', 'unknown'),
                'entry_quality': 'good' if pattern_result['position_in_range'] <= 0.3 else 'poor',
                'trend': pattern_result.get('trend', 'neutral'),
                'recent_breakout': pattern_result.get('recent_breakout', {}),
                'ceiling_touches': pattern_result.get('ceiling_touches', 0),
                'ground_touches': pattern_result.get('ground_touches', 0),

                # Additional analysis results
                'data': data,
                'pattern_data': pattern_result,
                'signal_data': signal_result,
                'trade_plan': trade_plan,
                'pattern_summary': pattern_summary,
                'backtest_result': backtest_result,
                'reasoning': signal_result['reasoning']
            }

            results.append(result)

            # Print quick result
            signal_emoji = get_signal_emoji(signal_result['signal'])
            zone_emoji = get_zone_emoji(pattern_result['position_in_range'])
            print(f"{signal_emoji} {signal_result['signal']} ({signal_result['confidence']:.0%}) {zone_emoji}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            print("❌ Error")
            continue

    # Calculate total time
    total_time = time.time() - start_time

    logger.info(f"Analysis complete. Processed {len(results)} symbols successfully")
    logger.info(f"Failed symbols: {len(failed_symbols)}")

    if not results:
        logger.warning("No results to display")
        print("\n⚠️  No Ground-Ceiling patterns found in any analyzed stocks.")
        return

    # Sort results by quality (combination of signal confidence and position quality)
    results = sort_results_by_quality(results)

    # Generate outputs
    logger.info("Generating outputs...")

    # 1. Print detailed summary
    print_detailed_summary(results)

    # 2. Save results to files
    save_comprehensive_results(results)

    # 3. Generate visualizations
    create_visualizations(results, visualizer)

    # 4. Generate trade alerts
    generate_trade_alerts(results)

    # Print footer
    print_analysis_footer(total_time, len(results))


def print_analysis_header_ground_ceiling():
    """Print header specific to Ground-Ceiling analysis"""
    header = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              GROUND & CEILING PATTERN DETECTOR                   ║
    ║                                                                  ║
    ║    🏠 Ground (Support) & 🏛️ Ceiling (Resistance) Analysis        ║
    ║    📈 Detects when old ceiling becomes new ground               ║
    ║    💰 Buy signals based on upside vs downside potential         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(header)


def get_symbols_to_analyze():
    """Get list of symbols to analyze"""
    try:
        # Try to load custom watchlist first
        if os.path.exists('watchlist.txt'):
            with open('watchlist.txt', 'r') as f:
                custom_symbols = [line.strip().upper() for line in f if line.strip()]
            if custom_symbols:
                print(f"📋 Loaded {len(custom_symbols)} symbols from watchlist.txt")
                return custom_symbols
    except:
        pass

    # Get S&P 500 symbols + major ETFs
    symbols = get_sp500_symbols()
    major_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT', 'XLF', 'XLK', 'XLE']
    all_symbols = symbols + major_etfs

    # Limit to reasonable number for demo (remove this in production)
    if len(all_symbols) > 100:
        print(f"📊 Analyzing top 100 symbols (out of {len(all_symbols)} available)")
        return all_symbols[:100]

    return all_symbols


def get_signal_emoji(signal):
    """Get emoji for signal type"""
    emoji_map = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡'
    }
    return emoji_map.get(signal, '⚪')


def get_zone_emoji(position_in_range):
    """Get emoji for price zone"""
    if position_in_range <= 0.25:
        return '🏠'  # Near ground
    elif position_in_range <= 0.5:
        return '📊'  # Lower range
    elif position_in_range <= 0.75:
        return '📈'  # Upper range
    else:
        return '🏛️'  # Near ceiling


def sort_results_by_quality(results):
    """Sort results by overall quality score"""

    def quality_score(result):
        confidence = result['confidence']
        pattern_strength = result['pattern_strength']
        risk_reward = min(result['risk_reward_ratio'], 5)  # Cap at 5
        position_quality = 1 - abs(result['position_in_range'] - 0.15)  # Best near ground (15%)

        # Bonus for BUY signals
        signal_bonus = 1.2 if result['signal'] == 'BUY' else 0.8

        total_score = (confidence * 0.3 +
                       pattern_strength * 0.3 +
                       (risk_reward / 5) * 0.2 +
                       position_quality * 0.2) * signal_bonus

        return total_score

    return sorted(results, key=quality_score, reverse=True)


def print_detailed_summary(results):
    """Print detailed summary of results"""
    print("\n" + "=" * 80)
    print("📊 GROUND-CEILING PATTERN ANALYSIS RESULTS")
    print("=" * 80)

    # Overall statistics
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    sell_signals = [r for r in results if r['signal'] == 'SELL']
    hold_signals = [r for r in results if r['signal'] == 'HOLD']

    print(f"📈 Total patterns found: {len(results)}")
    print(f"🟢 BUY signals: {len(buy_signals)}")
    print(f"🔴 SELL signals: {len(sell_signals)}")
    print(f"🟡 HOLD signals: {len(hold_signals)}")
    print()

    # Top buy opportunities
    if buy_signals:
        print("🎯 TOP BUY OPPORTUNITIES:")
        print("-" * 80)
        print(
            f"{'Symbol':<8} {'Price':<8} {'Ground':<8} {'Ceiling':<8} {'Zone':<12} {'Upside':<8} {'Conf':<6} {'R:R':<6}")
        print("-" * 80)

        for signal in buy_signals[:10]:  # Top 10
            zone_desc = f"{signal['position_in_range']:.0%} range"
            print(f"{signal['symbol']:<8} "
                  f"${signal['current_price']:<7.2f} "
                  f"${signal['ground_level']:<7.2f} "
                  f"${signal['ceiling_level']:<7.2f} "
                  f"{zone_desc:<12} "
                  f"{signal['upside_potential']:<7.1f}% "
                  f"{signal['confidence']:<5.0%} "
                  f"{signal['risk_reward_ratio']:<5.1f}")
        print()

    # Pattern strength analysis
    strong_patterns = [r for r in results if r['pattern_strength'] > 0.7]
    if strong_patterns:
        print(f"💪 STRONGEST PATTERNS ({len(strong_patterns)} found):")
        print("-" * 60)
        for pattern in strong_patterns[:5]:
            transition = pattern.get('transition_level')
            transition_str = f" (transition: ${transition:.2f})" if transition else ""
            print(f"  {pattern['symbol']:<6} - Strength: {pattern['pattern_strength']:.2f}{transition_str}")
        print()

    # Zone analysis
    ground_zone = [r for r in results if r['position_in_range'] <= 0.3]
    ceiling_zone = [r for r in results if r['position_in_range'] >= 0.7]

    print(f"🏠 Near Ground Zone: {len(ground_zone)} stocks")
    print(f"🏛️ Near Ceiling Zone: {len(ceiling_zone)} stocks")
    print()


def save_comprehensive_results(results):
    """Save comprehensive results to multiple files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Main results CSV
    main_results = []
    for r in results:
        main_results.append({
            'Symbol': r['symbol'],
            'Signal': r['signal'],
            'Confidence': f"{r['confidence']:.1%}",
            'Current_Price': f"${r['current_price']:.2f}",
            'Ground_Level': f"${r['ground_level']:.2f}",
            'Ceiling_Level': f"${r['ceiling_level']:.2f}",
            'Position_In_Range': f"{r['position_in_range']:.1%}",
            'Zone': r['pattern_zone'],
            'Upside_Potential': f"{r['upside_potential']:.1f}%",
            'Downside_Risk': f"{r['downside_risk']:.1f}%",
            'Risk_Reward_Ratio': f"{r['risk_reward_ratio']:.2f}",
            'Pattern_Strength': f"{r['pattern_strength']:.2f}",
            'Transition_Level': f"${r['transition_level']:.2f}" if r['transition_level'] else 'N/A',
            'Target_Price': f"${r['target_price']:.2f}" if r['target_price'] else 'N/A',
            'Stop_Loss': f"${r['stop_loss']:.2f}" if r['stop_loss'] else 'N/A',
            'Trend': r['trend'],
            'Entry_Quality': r['entry_quality'],
            'Current_RSI': f"{r['current_rsi']:.1f}",
            'Reasoning': r['reasoning']
        })

    df_main = pd.DataFrame(main_results)
    main_filename = f"ground_ceiling_results_{timestamp}.csv"
    df_main.to_csv(main_filename, index=False)
    print(f"📄 Main results saved to: {main_filename}")

    # 2. Buy signals only (for easy trading)
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    if buy_signals:
        buy_results = []
        for r in buy_signals:
            buy_results.append({
                'Symbol': r['symbol'],
                'Current_Price': f"${r['current_price']:.2f}",
                'Entry_Quality': r['entry_quality'],
                'Ground_Support': f"${r['ground_level']:.2f}",
                'Ceiling_Target': f"${r['ceiling_level']:.2f}",
                'Stop_Loss': f"${r['stop_loss']:.2f}" if r['stop_loss'] else 'N/A',
                'Upside_Potential': f"{r['upside_potential']:.1f}%",
                'Risk_Reward': f"{r['risk_reward_ratio']:.2f}",
                'Confidence': f"{r['confidence']:.1%}",
                'Pattern_Strength': f"{r['pattern_strength']:.2f}",
                'Why_Buy': r['reasoning'][:100] + "..." if len(r['reasoning']) > 100 else r['reasoning']
            })

        df_buy = pd.DataFrame(buy_results)
        buy_filename = f"buy_signals_{timestamp}.csv"
        df_buy.to_csv(buy_filename, index=False)
        print(f"🎯 Buy signals saved to: {buy_filename}")

    # 3. Pattern summary report
    report_filename = f"pattern_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write("GROUND-CEILING PATTERN ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("SUMMARY:\n")
        f.write(f"Total patterns detected: {len(results)}\n")
        f.write(f"BUY signals: {len([r for r in results if r['signal'] == 'BUY'])}\n")
        f.write(f"SELL signals: {len([r for r in results if r['signal'] == 'SELL'])}\n")
        f.write(f"HOLD signals: {len([r for r in results if r['signal'] == 'HOLD'])}\n\n")

        # Top 5 patterns with full details
        f.write("TOP 5 PATTERNS:\n")
        f.write("-" * 30 + "\n")
        for i, r in enumerate(results[:5], 1):
            f.write(f"\n{i}. {r['symbol']} - {r['signal']} Signal\n")
            f.write(f"   Pattern Summary:\n")
            for line in r['pattern_summary'].split('\n'):
                f.write(f"   {line}\n")
            f.write(f"   Reasoning: {r['reasoning']}\n")

    print(f"📋 Detailed report saved to: {report_filename}")


def create_visualizations(results, visualizer):
    """Create visualizations for the results"""
    try:
        print("📊 Creating visualizations...")

        # Create summary dashboard
        visualizer.create_summary_dashboard(results)

        # Create individual charts for top 5 buy signals
        buy_signals = [r for r in results if r['signal'] == 'BUY'][:5]
        if buy_signals:
            print(f"📈 Creating individual charts for top {len(buy_signals)} buy signals...")
            visualizer.create_individual_charts(buy_signals)

    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")


def generate_trade_alerts(results):
    """Generate trading alerts for immediate action"""
    print("\n" + "=" * 60)
    print("🚨 IMMEDIATE TRADING ALERTS")
    print("=" * 60)

    # High-confidence buy signals near ground
    urgent_buys = [r for r in results if (
            r['signal'] == 'BUY' and
            r['confidence'] > 0.7 and
            r['position_in_range'] <= 0.3 and
            r['risk_reward_ratio'] > 2
    )]

    if urgent_buys:
        print("🔥 HIGH-PRIORITY BUY ALERTS:")
        print("-" * 40)
        for alert in urgent_buys[:3]:  # Top 3
            print(f"🎯 {alert['symbol']} at ${alert['current_price']:.2f}")
            print(f"   • Near ground (${alert['ground_level']:.2f}) in {alert['position_in_range']:.0%} of range")
            print(f"   • Upside: {alert['upside_potential']:.1f}% to ${alert['ceiling_level']:.2f}")
            print(f"   • Risk-Reward: {alert['risk_reward_ratio']:.1f}:1")
            print(f"   • Confidence: {alert['confidence']:.0%}")
            if alert['transition_level']:
                print(f"   • Key Level: ${alert['transition_level']:.2f} (old ceiling → new ground)")
            print()
    else:
        print("ℹ️  No high-priority buy alerts at this time")

    # Breakout alerts
    breakout_alerts = [r for r in results if (
            r['recent_breakout'].get('ceiling_breakout', False) or
            r['recent_breakout'].get('ground_breakdown', False)
    )]

    if breakout_alerts:
        print("⚡ BREAKOUT ALERTS:")
        print("-" * 20)
        for alert in breakout_alerts[:3]:
            breakout_type = "Ceiling Breakout" if alert['recent_breakout'].get(
                'ceiling_breakout') else "Ground Breakdown"
            emoji = "🚀" if "Ceiling" in breakout_type else "⚠️"
            print(f"{emoji} {alert['symbol']}: {breakout_type}")
            print(f"   Current: ${alert['current_price']:.2f} | Signal: {alert['signal']}")

    print("\n📝 Note: Always do your own research before making trading decisions!")


def create_watchlist_template():
    """Create a watchlist template file"""
    template_content = """# Ground-Ceiling Analysis Watchlist
# Add one symbol per line (without # symbol)
# Example symbols:

AAPL
MSFT  
GOOGL
AMZN
TSLA
NVDA
META
SPY
QQQ
"""

    try:
        with open('watchlist_template.txt', 'w') as f:
            f.write(template_content)
        print("📝 Created watchlist_template.txt - copy to watchlist.txt and customize")
    except Exception as e:
        print(f"❌ Error creating watchlist template: {str(e)}")


if __name__ == "__main__":
    # Check if watchlist exists, if not create template
    if not os.path.exists('watchlist.txt'):
        create_watchlist_template()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logging.getLogger(__name__).error(f"Fatal error in main: {str(e)}", exc_info=True)
    finally:
        print("\n👋 Analysis complete. Check generated files for detailed results.")