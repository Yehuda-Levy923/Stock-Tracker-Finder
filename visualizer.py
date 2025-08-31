import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """Class to create visualizations for stock pattern analysis"""

    def __init__(self, figsize: tuple = (15, 10), dpi: int = 100):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
            dpi: Figure DPI for quality
        """
        self.figsize = figsize
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)

        # Color scheme
        self.colors = {
            'BUY': '#00ff00',  # Green
            'SELL': '#ff0000',  # Red
            'HOLD': '#ffff00',  # Yellow
            'price': '#1f77b4',  # Blue
            'rsi': '#ff7f0e',  # Orange
            'pattern': '#2ca02c',  # Green
            'volume': '#d62728'  # Red
        }

    def create_summary_dashboard(self, results: List[Dict], save_path: str = None):
        """
        Create a summary dashboard with key metrics and signals

        Args:
            results: List of analysis results
            save_path: Path to save the figure
        """
        try:
            fig = plt.figure(figsize=(20, 12), dpi=self.dpi)

            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 1. Signal distribution pie chart
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_signal_distribution(results, ax1)

            # 2. Confidence histogram
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_confidence_histogram(results, ax2)

            # 3. Pattern strength vs RSI scatter
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_pattern_vs_rsi(results, ax3)

            # 4. Top signals table
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_top_signals_table(results, ax4)

            # 5. Performance metrics
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_performance_metrics(results, ax5)

            # 6. Cycle phase distribution
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_cycle_phase_distribution(results, ax6)

            # 7. Risk-Reward scatter
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_risk_reward_scatter(results, ax7)

            plt.suptitle('Stock Pattern Analysis Dashboard', fontsize=20, fontweight='bold')

            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig(f'analysis_dashboard_{timestamp}.png', dpi=self.dpi, bbox_inches='tight')

            plt.show()

        except Exception as e:
            self.logger.error(f"Error creating summary dashboard: {str(e)}")

    def create_individual_charts(self, results: List[Dict], max_charts: int = 10):
        """
        Create individual charts for top signals

        Args:
            results: List of analysis results
            max_charts: Maximum number of individual charts to create
        """
        # Sort by confidence and take top results
        top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:max_charts]

        for i, result in enumerate(top_results):
            try:
                self._create_individual_stock_chart(result, i + 1)
            except Exception as e:
                self.logger.error(f"Error creating chart for {result['symbol']}: {str(e)}")

    def _create_individual_stock_chart(self, result: Dict, chart_num: int):
        """Create detailed chart for individual stock"""
        symbol = result['symbol']
        data = result['data']
        pattern_data = result['pattern_data']

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle(f'{symbol} - Pattern Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Price with pattern overlay
        ax1 = axes[0]
        dates = data.index
        prices = data['Close']

        ax1.plot(dates, prices, color=self.colors['price'], linewidth=2, label='Price')

        # Overlay sine wave pattern if available
        if 'sine_params' in pattern_data and 'fitted_curve' in pattern_data['sine_params']:
            fitted_curve = pattern_data['sine_params']['fitted_curve']
            # Denormalize the fitted curve to price scale
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            fitted_prices = fitted_curve * price_std + price_mean

            ax1.plot(dates, fitted_prices, color=self.colors['pattern'],
                     linewidth=2, alpha=0.7, linestyle='--', label='Sine Pattern')

        # Mark current signal
        current_price = result['current_price']
        signal = result['signal']

        ax1.scatter(dates[-1], current_price, color=self.colors[signal],
                    s=200, marker='o', edgecolor='black', linewidth=2,
                    label=f'{signal} Signal', zorder=5)

        # Add target and stop loss if available
        if result.get('target_price'):
            ax1.axhline(y=result['target_price'], color='green',
                        linestyle=':', alpha=0.7, label='Target')
        if result.get('stop_loss'):
            ax1.axhline(y=result['stop_loss'], color='red',
                        linestyle=':', alpha=0.7, label='Stop Loss')

        ax1.set_title(f'Price Chart - {signal} Signal (Confidence: {result["confidence"]:.1%})')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: RSI with levels
        ax2 = axes[1]
        rsi = data['RSI']
        ax2.plot(dates, rsi, color=self.colors['rsi'], linewidth=2, label='RSI')

        # RSI levels
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

        # Current RSI point
        current_rsi = result['current_rsi']
        ax2.scatter(dates[-1], current_rsi, color=self.colors[signal],
                    s=100, marker='o', edgecolor='black', zorder=5)

        ax2.set_title(f'RSI Indicator (Current: {current_rsi:.1f})')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Volume
        ax3 = axes[2]
        volume = data['Volume']
        colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red'
                  for i in range(len(data))]
        ax3.bar(dates, volume, color=colors, alpha=0.7, width=1)

        ax3.set_title('Volume')
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save individual chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.show()

    def _plot_signal_distribution(self, results: List[Dict], ax):
        """Plot pie chart of signal distribution"""
        signals = [r['signal'] for r in results]
        signal_counts = pd.Series(signals).value_counts()

        colors = [self.colors.get(signal, '#gray') for signal in signal_counts.index]

        wedges, texts, autotexts = ax.pie(signal_counts.values, labels=signal_counts.index,
                                          colors=colors, autopct='%1.1f%%', startangle=90)

        ax.set_title('Signal Distribution')

        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    def _plot_confidence_histogram(self, results: List[Dict], ax):
        """Plot histogram of confidence levels"""
        confidences = [r['confidence'] for r in results]

        ax.hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.2f}')

        ax.set_title('Confidence Distribution')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_pattern_vs_rsi(self, results: List[Dict], ax):
        """Plot pattern strength vs RSI scatter"""
        pattern_strengths = [r['pattern_strength'] for r in results]
        rsi_values = [r['current_rsi'] for r in results]
        signals = [r['signal'] for r in results]

        for signal in ['BUY', 'SELL', 'HOLD']:
            mask = [s == signal for s in signals]
            if any(mask):
                x = [pattern_strengths[i] for i in range(len(mask)) if mask[i]]
                y = [rsi_values[i] for i in range(len(mask)) if mask[i]]
                ax.scatter(x, y, c=self.colors[signal], label=signal, alpha=0.7, s=50)

        ax.set_title('Pattern Strength vs RSI')
        ax.set_xlabel('Pattern Strength')
        ax.set_ylabel('RSI Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add RSI level lines
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)

    def _plot_top_signals_table(self, results: List[Dict], ax):
        """Plot table of top signals"""
        # Sort by confidence
        top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:10]

        table_data = []
        for r in top_results:
            table_data.append([
                r['symbol'],
                r['signal'],
                f"{r['confidence']:.1%}",
                f"${r['current_price']:.2f}",
                f"{r['current_rsi']:.1f}",
                r['pattern_phase'],
                f"${r.get('target_price', 0):.2f}" if r.get('target_price') else 'N/A'
            ])

        columns = ['Symbol', 'Signal', 'Confidence', 'Price', 'RSI', 'Phase', 'Target']

        table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color code the signal column
        for i, row in enumerate(table_data):
            signal = row[1]
            table[(i + 1, 1)].set_facecolor(self.colors.get(signal, 'white'))
            table[(i + 1, 1)].set_text_props(weight='bold')

        ax.axis('off')
        ax.set_title('Top 10 Signals by Confidence', fontweight='bold', pad=20)

    def _plot_performance_metrics(self, results: List[Dict], ax):
        """Plot key performance metrics"""
        metrics = {
            'Total Analyzed': len(results),
            'Buy Signals': len([r for r in results if r['signal'] == 'BUY']),
            'Sell Signals': len([r for r in results if r['signal'] == 'SELL']),
            'Hold Signals': len([r for r in results if r['signal'] == 'HOLD']),
            'Avg Confidence': f"{np.mean([r['confidence'] for r in results]):.1%}",
            'High Confidence': len([r for r in results if r['confidence'] > 0.7])
        }

        y_pos = np.arange(len(metrics))
        values = []
        labels = []

        for key, value in metrics.items():
            labels.append(key)
            if isinstance(value, str):
                # For percentage, extract numeric value for plotting
                if '%' in value:
                    values.append(float(value.strip('%')) / 100)
                else:
                    values.append(0)
            else:
                values.append(value)

        bars = ax.barh(y_pos, values, color='lightblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title('Analysis Metrics')
        ax.set_xlabel('Count / Percentage')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metrics.values())):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    str(value), va='center', fontweight='bold')

    def _plot_cycle_phase_distribution(self, results: List[Dict], ax):
        """Plot distribution of cycle phases"""
        phases = [r['cycle_phase'] for r in results]
        phase_counts = pd.Series(phases).value_counts()

        bars = ax.bar(phase_counts.index, phase_counts.values,
                      color=['orange', 'red', 'blue', 'green'], alpha=0.7)

        ax.set_title('Cycle Phase Distribution')
        ax.set_ylabel('Count')
        ax.set_xlabel('Cycle Phase')

        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

    def _plot_risk_reward_scatter(self, results: List[Dict], ax):
        """Plot risk-reward ratio scatter"""
        symbols = []
        risk_rewards = []
        confidences = []
        signals = []

        for r in results:
            if r.get('risk_reward_ratio') and r['risk_reward_ratio'] > 0:
                symbols.append(r['symbol'])
                risk_rewards.append(r['risk_reward_ratio'])
                confidences.append(r['confidence'])
                signals.append(r['signal'])

        if not risk_rewards:
            ax.text(0.5, 0.5, 'No Risk-Reward Data Available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk-Reward Analysis')
            return

        # Create scatter plot with confidence as size and signal as color
        for signal in ['BUY', 'SELL', 'HOLD']:
            mask = [s == signal for s in signals]
            if any(mask):
                x = [confidences[i] for i in range(len(mask)) if mask[i]]
                y = [risk_rewards[i] for i in range(len(mask)) if mask[i]]
                sizes = [c * 200 for c in x]  # Scale confidence for marker size
                ax.scatter(x, y, c=self.colors[signal], label=signal,
                           alpha=0.7, s=sizes, edgecolors='black', linewidth=0.5)

        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7,
                   label='2:1 Target')
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7,
                   label='1:1 Breakeven')

        ax.set_title('Risk-Reward vs Confidence')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Risk-Reward Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_pattern_analysis_chart(self, data: pd.DataFrame, pattern_result: Dict,
                                      symbol: str, save_path: str = None):
        """
        Create detailed pattern analysis chart for a single stock

        Args:
            data: Stock data with indicators
            pattern_result: Pattern analysis results
            symbol: Stock symbol
            save_path: Path to save the chart
        """
        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 16), dpi=self.dpi)
            fig.suptitle(f'{symbol} - Detailed Pattern Analysis', fontsize=16, fontweight='bold')

            dates = data.index

            # Plot 1: Price with pattern overlay and phases
            ax1 = axes[0]
            prices = data['Close']
            ax1.plot(dates, prices, color=self.colors['price'], linewidth=2, label='Close Price')

            # Overlay fitted sine wave if available
            if 'sine_params' in pattern_result and 'fitted_curve' in pattern_result['sine_params']:
                fitted_curve = pattern_result['sine_params']['fitted_curve']
                # Denormalize
                price_mean = np.mean(prices)
                price_std = np.std(prices)
                fitted_prices = fitted_curve * price_std + price_mean

                ax1.plot(dates, fitted_prices, color=self.colors['pattern'],
                         linewidth=3, alpha=0.8, linestyle='--', label='Sine Pattern')

            ax1.set_title(f'Price Pattern (Strength: {pattern_result.get("strength", 0):.2f})')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: RSI with divergence
            ax2 = axes[1]
            rsi = data['RSI']
            ax2.plot(dates, rsi, color=self.colors['rsi'], linewidth=2, label='RSI')

            # RSI levels
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

            # Highlight RSI signals if available
            if 'RSI_Signal' in data.columns:
                buy_signals = data[data['RSI_Signal'] == 'BUY'].index
                sell_signals = data[data['RSI_Signal'] == 'SELL'].index

                if len(buy_signals) > 0:
                    ax2.scatter(buy_signals, data.loc[buy_signals, 'RSI'],
                                color='green', marker='^', s=100, label='RSI Buy', zorder=5)
                if len(sell_signals) > 0:
                    ax2.scatter(sell_signals, data.loc[sell_signals, 'RSI'],
                                color='red', marker='v', s=100, label='RSI Sell', zorder=5)

            ax2.set_title('RSI Indicator with Signals')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Pattern cycle visualization
            ax3 = axes[2]
            if 'sine_params' in pattern_result:
                period = pattern_result['sine_params']['period']
                phase = pattern_result['sine_params']['phase']

                # Create idealized sine wave for reference
                x_ideal = np.linspace(0, len(data), 1000)
                y_ideal = np.sin(2 * np.pi / period * x_ideal + phase)

                ax3.plot(x_ideal, y_ideal, color='blue', linewidth=2, label='Ideal Sine Wave')
                ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

                # Mark current position
                current_pos = len(data) - 1
                current_sine_value = np.sin(2 * np.pi / period * current_pos + phase)
                ax3.scatter(current_pos, current_sine_value, color='red', s=200,
                            marker='o', label='Current Position', zorder=5)

                # Add phase labels
                quarter_period = period / 4
                phase_positions = [quarter_period, 2 * quarter_period, 3 * quarter_period, period]
                phase_labels = ['Peak', 'Falling', 'Trough', 'Rising']

                for pos, label in zip(phase_positions, phase_labels):
                    if pos <= len(data):
                        ax3.axvline(x=pos, color='orange', linestyle=':', alpha=0.7)
                        ax3.text(pos, 0.8, label, rotation=90, va='bottom', ha='right')

            ax3.set_title(f'Cycle Pattern (Period: {pattern_result.get("sine_params", {}).get("period", 0):.1f} days)')
            ax3.set_ylabel('Normalized Sine Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Volume with moving average
            ax4 = axes[3]
            volume = data['Volume']
            volume_ma = volume.rolling(window=20).mean()

            ax4.bar(dates, volume, alpha=0.5, color='gray', label='Volume')
            ax4.plot(dates, volume_ma, color='red', linewidth=2, label='20-day MA')

            ax4.set_title('Volume Analysis')
            ax4.set_ylabel('Volume')
            ax4.set_xlabel('Date')
            ax4.legend()

            # Format all x-axes
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig(f'{symbol}_detailed_analysis_{timestamp}.png',
                            dpi=self.dpi, bbox_inches='tight')

            plt.show()

        except Exception as e:
            self.logger.error(f"Error creating pattern analysis chart: {str(e)}")

    def save_results_summary(self, results: List[Dict], filename: str = None):
        """Save a text summary of results"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_summary_{timestamp}.txt'

        try:
            with open(filename, 'w') as f:
                f.write("STOCK PATTERN ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")

                # Overall statistics
                total_stocks = len(results)
                buy_signals = len([r for r in results if r['signal'] == 'BUY'])
                sell_signals = len([r for r in results if r['signal'] == 'SELL'])
                hold_signals = len([r for r in results if r['signal'] == 'HOLD'])

                f.write(f"Total stocks analyzed: {total_stocks}\n")
                f.write(f"BUY signals: {buy_signals} ({buy_signals / total_stocks:.1%})\n")
                f.write(f"SELL signals: {sell_signals} ({sell_signals / total_stocks:.1%})\n")
                f.write(f"HOLD signals: {hold_signals} ({hold_signals / total_stocks:.1%})\n\n")

                # Top buy signals
                buy_results = [r for r in results if r['signal'] == 'BUY']
                if buy_results:
                    f.write("TOP BUY SIGNALS:\n")
                    f.write("-" * 20 + "\n")
                    buy_results.sort(key=lambda x: x['confidence'], reverse=True)
                    for r in buy_results[:10]:
                        f.write(f"{r['symbol']:<6} | Price: ${r['current_price']:>8.2f} | ")
                        f.write(f"Confidence: {r['confidence']:>6.1%} | ")
                        f.write(f"RSI: {r['current_rsi']:>5.1f} | ")
                        f.write(f"Phase: {r['cycle_phase']:<8}\n")
                    f.write("\n")

                # Top sell signals
                sell_results = [r for r in results if r['signal'] == 'SELL']
                if sell_results:
                    f.write("TOP SELL SIGNALS:\n")
                    f.write("-" * 20 + "\n")
                    sell_results.sort(key=lambda x: x['confidence'], reverse=True)
                    for r in sell_results[:10]:
                        f.write(f"{r['symbol']:<6} | Price: ${r['current_price']:>8.2f} | ")
                        f.write(f"Confidence: {r['confidence']:>6.1%} | ")
                        f.write(f"RSI: {r['current_rsi']:>5.1f} | ")
                        f.write(f"Phase: {r['cycle_phase']:<8}\n")
                    f.write("\n")

                # Statistics
                confidences = [r['confidence'] for r in results]
                pattern_strengths = [r['pattern_strength'] for r in results]

                f.write("STATISTICS:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Average confidence: {np.mean(confidences):.1%}\n")
                f.write(f"Average pattern strength: {np.mean(pattern_strengths):.2f}\n")
                f.write(f"High confidence signals (>70%): {len([c for c in confidences if c > 0.7])}\n")

            print(f"Summary saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving summary: {str(e)}")