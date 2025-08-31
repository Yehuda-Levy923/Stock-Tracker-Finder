import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

class SignalGenerator:
    """Class to generate trading signals based on pattern analysis and RSI"""

    def __init__(self,
                 min_confidence: float = 0.6,
                 rsi_weight: float = 0.4,
                 pattern_weight: float = 0.6):
        """
        Initialize signal generator

        Args:
            min_confidence: Minimum confidence level for signals
            rsi_weight: Weight for RSI in signal calculation
            pattern_weight: Weight for pattern analysis in signal calculation
        """
        self.min_confidence = min_confidence
        self.rsi_weight = rsi_weight
        self.pattern_weight = pattern_weight
        self.logger = logging.getLogger(__name__)

        # Risk management parameters
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.profit_target_ratio = 2.0  # 2:1 reward:risk ratio

    def generate_signal(self, data: pd.DataFrame, pattern_result: Dict, current_data: pd.Series) -> Dict:
        """
        Generate trading signal based on pattern and RSI analysis

        Args:
            data: Historical price and indicator data
            pattern_result: Results from sine wave pattern detection
            current_data: Current data point

        Returns:
            Dictionary with signal information
        """
        try:
            current_price = current_data['Close']
            current_rsi = current_data.get('RSI', 50)

            # Analyze pattern signal
            pattern_signal = self._analyze_pattern_signal(pattern_result, current_price)

            # Analyze RSI signal
            rsi_signal = self._analyze_rsi_signal(current_rsi, data)

            # Combine signals
            combined_signal = self._combine_signals(pattern_signal, rsi_signal)

            # Calculate position sizing and risk management
            risk_management = self._calculate_risk_management(current_price, combined_signal, pattern_result)

            # Generate final signal with confidence
            final_signal = self._generate_final_signal(combined_signal, pattern_result, current_rsi)

            result = {
                'signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'pattern_signal': pattern_signal,
                'rsi_signal': rsi_signal,
                'current_price': current_price,
                'current_rsi': current_rsi,
                'target_price': risk_management.get('target_price'),
                'stop_loss': risk_management.get('stop_loss'),
                'position_size': risk_management.get('position_size', 1.0),
                'risk_reward_ratio': risk_management.get('risk_reward_ratio'),
                'reasoning': final_signal['reasoning'],
                'pattern_phase': pattern_result.get('current_phase', 'unknown'),
                'trend': pattern_result.get('trend', 'neutral')
            }

            self.logger.debug(f"Generated signal: {result['signal']} with confidence: {result['confidence']:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }

    def _analyze_pattern_signal(self, pattern_result: Dict, current_price: float) -> Dict:
        """Analyze signal from sine wave pattern"""
        if not pattern_result or pattern_result.get('strength', 0) < 0.3:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'reasoning': 'Pattern strength too weak'
            }

        current_phase = pattern_result.get('current_phase', 'unknown')
        pattern_strength = pattern_result.get('strength', 0)
        trend = pattern_result.get('trend', 'neutral')
        next_reversal = pattern_result.get('next_reversal', 'uncertain')

        # Signal logic based on pattern phase and trend
        if current_phase == 'trough' and trend in ['neutral', 'uptrend']:
            signal = 'BUY'
            reasoning = f"Near cycle trough with {trend} trend - expect reversal up"
            strength = pattern_strength * 0.9

        elif current_phase == 'rising' and trend == 'uptrend':
            signal = 'BUY'
            reasoning = f"In rising phase with uptrend - momentum continuing"
            strength = pattern_strength * 0.7

        elif current_phase == 'peak' and trend in ['neutral', 'downtrend']:
            signal = 'SELL'
            reasoning = f"Near cycle peak with {trend} trend - expect reversal down"
            strength = pattern_strength * 0.9

        elif current_phase == 'falling' and trend == 'downtrend':
            signal = 'SELL'
            reasoning = f"In falling phase with downtrend - momentum continuing"
            strength = pattern_strength * 0.7

        else:
            signal = 'HOLD'
            reasoning = f"Phase: {current_phase}, Trend: {trend} - mixed signals"
            strength = pattern_strength * 0.3

        return {
            'signal': signal,
            'strength': min(strength, 1.0),
            'reasoning': reasoning,
            'phase': current_phase,
            'trend': trend
        }

    def _analyze_rsi_signal(self, current_rsi: float, data: pd.DataFrame) -> Dict:
        """Analyze signal from RSI"""
        try:
            # RSI thresholds
            oversold_threshold = 30
            overbought_threshold = 70
            extreme_oversold = 20
            extreme_overbought = 80

            # Get recent RSI trend
            recent_rsi = data['RSI'].tail(10)
            rsi_momentum = recent_rsi.diff().mean()
            rsi_trend = 'rising' if rsi_momentum > 0 else 'falling'

            # Generate signal based on RSI level and momentum
            if current_rsi <= extreme_oversold:
                signal = 'BUY'
                strength = 0.9
                reasoning = f"RSI extremely oversold at {current_rsi:.1f} - strong buy signal"

            elif current_rsi <= oversold_threshold and rsi_trend == 'rising':
                signal = 'BUY'
                strength = 0.7
                reasoning = f"RSI oversold at {current_rsi:.1f} and rising - buy signal"

            elif current_rsi >= extreme_overbought:
                signal = 'SELL'
                strength = 0.9
                reasoning = f"RSI extremely overbought at {current_rsi:.1f} - strong sell signal"

            elif current_rsi >= overbought_threshold and rsi_trend == 'falling':
                signal = 'SELL'
                strength = 0.7
                reasoning = f"RSI overbought at {current_rsi:.1f} and falling - sell signal"

            elif 40 <= current_rsi <= 60:
                signal = 'HOLD'
                strength = 0.3
                reasoning = f"RSI neutral at {current_rsi:.1f} - no clear signal"

            else:
                # RSI in middle range but trending
                if current_rsi < 50 and rsi_trend == 'rising':
                    signal = 'BUY'
                    strength = 0.4
                    reasoning = f"RSI below 50 but rising - weak buy signal"
                elif current_rsi > 50 and rsi_trend == 'falling':
                    signal = 'SELL'
                    strength = 0.4
                    reasoning = f"RSI above 50 but falling - weak sell signal"
                else:
                    signal = 'HOLD'
                    strength = 0.2
                    reasoning = f"RSI at {current_rsi:.1f} - mixed signals"

            return {
                'signal': signal,
                'strength': strength,
                'reasoning': reasoning,
                'rsi_value': current_rsi,
                'rsi_trend': rsi_trend
            }

        except Exception as e:
            self.logger.error(f"Error analyzing RSI signal: {str(e)}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'reasoning': 'Error in RSI analysis'
            }

    def _combine_signals(self, pattern_signal: Dict, rsi_signal: Dict) -> Dict:
        """Combine pattern and RSI signals"""
        pattern_sig = pattern_signal.get('signal', 'HOLD')
        rsi_sig = rsi_signal.get('signal', 'HOLD')

        pattern_strength = pattern_signal.get('strength', 0)
        rsi_strength = rsi_signal.get('strength', 0)

        # Calculate weighted strength
        combined_strength = (
            self.pattern_weight * pattern_strength +
            self.rsi_weight * rsi_strength
        )

        # Signal combination logic
        if pattern_sig == rsi_sig:
            # Both signals agree
            combined_signal = pattern_sig
            confidence_multiplier = 1.2  # Boost confidence when signals agree
        elif pattern_sig == 'HOLD' or rsi_sig == 'HOLD':
            # One signal is neutral
            non_neutral_signal = pattern_sig if pattern_sig != 'HOLD' else rsi_sig
            combined_signal = non_neutral_signal
            confidence_multiplier = 0.8  # Reduce confidence when one is neutral
        else:
            # Conflicting signals
            # Use stronger signal, but reduce confidence significantly
            if pattern_strength > rsi_strength:
                combined_signal = pattern_sig
            else:
                combined_signal = rsi_sig
            confidence_multiplier = 0.5  # Low confidence for conflicting signals

        combined_confidence = min(combined_strength * confidence_multiplier, 1.0)

        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'pattern_signal': pattern_sig,
            'rsi_signal': rsi_sig,
            'agreement': pattern_sig == rsi_sig
        }

    def _calculate_risk_management(self, current_price: float, combined_signal: Dict, pattern_result: Dict) -> Dict:
        """Calculate position sizing and risk management levels"""
        try:
            signal = combined_signal.get('signal', 'HOLD')

            if signal == 'HOLD':
                return {}

            # Use pattern amplitude and volatility for risk calculation
            amplitude = pattern_result.get('sine_params', {}).get('amplitude', 0.1)

            # Estimate volatility-based stop loss (simplified)
            volatility_stop = current_price * 0.05  # 5% default
            pattern_stop = current_price * amplitude * 0.5  # Half the pattern amplitude

            if signal == 'BUY':
                stop_loss = current_price - max(volatility_stop, pattern_stop)
                target_price = current_price + (current_price - stop_loss) * self.profit_target_ratio
            else:  # SELL
                stop_loss = current_price + max(volatility_stop, pattern_stop)
                target_price = current_price - (stop_loss - current_price) * self.profit_target_ratio

            risk_per_share = abs(current_price - stop_loss)
            risk_reward_ratio = abs(target_price - current_price) / risk_per_share if risk_per_share > 0 else 0

            # Position sizing based on risk
            max_loss_per_trade = 1000  # Example: $1000 max loss per trade
            position_size = max_loss_per_trade / risk_per_share if risk_per_share > 0 else 1

            return {
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'position_size': round(position_size, 0),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'risk_per_share': round(risk_per_share, 2)
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk management: {str(e)}")
            return {}

    def _generate_final_signal(self, combined_signal: Dict, pattern_result: Dict, current_rsi: float) -> Dict:
        """Generate final signal with comprehensive reasoning"""
        signal = combined_signal.get('signal', 'HOLD')
        base_confidence = combined_signal.get('confidence', 0)

        # Apply confidence filters
        if base_confidence < self.min_confidence:
            signal = 'HOLD'
            reasoning = f"Signal confidence {base_confidence:.2f} below minimum threshold {self.min_confidence}"
        else:
            # Build comprehensive reasoning
            pattern_phase = pattern_result.get('current_phase', 'unknown')
            pattern_strength = pattern_result.get('strength', 0)

            reasoning_parts = []

            if combined_signal.get('agreement', False):
                reasoning_parts.append("Pattern and RSI signals agree")
            else:
                reasoning_parts.append("Mixed signals from pattern and RSI")

            reasoning_parts.append(f"Pattern phase: {pattern_phase}")
            reasoning_parts.append(f"Pattern strength: {pattern_strength:.2f}")
            reasoning_parts.append(f"RSI: {current_rsi:.1f}")

            reasoning = "; ".join(reasoning_parts)

        # Final confidence adjustment
        final_confidence = base_confidence

        # Boost confidence for strong patterns with confirming RSI
        if (pattern_result.get('strength', 0) > 0.7 and
            combined_signal.get('agreement', False)):
            final_confidence *= 1.1

        # Reduce confidence for weak trends
        if pattern_result.get('trend', '') == 'neutral':
            final_confidence *= 0.9

        final_confidence = min(final_confidence, 0.95)  # Cap at 95%

        return {
            'signal': signal,
            'confidence': final_confidence,
            'reasoning': reasoning
        }