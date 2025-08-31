import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List


class SignalGenerator:
    """Complete Signal generator for Ground-Ceiling pattern analysis"""

    def __init__(self,
                 min_confidence: float = 0.6,
                 rsi_weight: float = 0.3,
                 pattern_weight: float = 0.7):
        """
        Initialize signal generator for Ground-Ceiling patterns

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
        Generate trading signal based on Ground-Ceiling pattern and RSI analysis

        Args:
            data: Historical price and indicator data
            pattern_result: Results from ground-ceiling pattern detection
            current_data: Current data point

        Returns:
            Dictionary with signal information
        """
        try:
            current_price = current_data['Close']
            current_rsi = current_data.get('RSI', 50)

            # Analyze Ground-Ceiling pattern signal
            pattern_signal = self._analyze_ground_ceiling_signal(pattern_result, current_price, data)

            # Analyze RSI signal
            rsi_signal = self._analyze_rsi_signal(current_rsi, data)

            # Combine signals
            combined_signal = self._combine_signals(pattern_signal, rsi_signal, pattern_result)

            # Calculate risk management based on ground/ceiling levels
            risk_management = self._calculate_ground_ceiling_risk_management(
                current_price, combined_signal, pattern_result
            )

            # Generate final signal with reasoning
            final_signal = self._generate_final_signal(
                combined_signal, pattern_result, current_rsi, current_price
            )

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
                'ground_level': pattern_result.get('current_ground'),
                'ceiling_level': pattern_result.get('current_ceiling'),
                'position_in_range': pattern_result.get('position_in_range', 0.5),
                'upside_potential': pattern_result.get('upside_potential', 0),
                'downside_risk': pattern_result.get('downside_risk', 0),
                'pattern_zone': self._determine_zone(pattern_result.get('position_in_range', 0.5)),
                'recent_breakout': pattern_result.get('recent_breakout', {}),
                'transition_level': pattern_result.get('old_ceiling_became_ground')
            }

            self.logger.debug(
                f"Generated Ground-Ceiling signal: {result['signal']} with confidence: {result['confidence']:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }

    def _analyze_ground_ceiling_signal(self, pattern_result: Dict, current_price: float, data: pd.DataFrame) -> Dict:
        """Analyze signal from Ground-Ceiling pattern"""
        if not pattern_result or pattern_result.get('strength', 0) < 0.4:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'reasoning': 'Pattern strength too weak'
            }

        current_ground = pattern_result.get('current_ground', current_price)
        current_ceiling = pattern_result.get('current_ceiling', current_price)
        position_in_range = pattern_result.get('position_in_range', 0.5)
        upside_potential = pattern_result.get('upside_potential', 0)
        downside_risk = pattern_result.get('downside_risk', 0)
        trend = pattern_result.get('trend', 'neutral')
        pattern_strength = pattern_result.get('strength', 0)
        recent_breakout = pattern_result.get('recent_breakout', {})

        # Key algorithm implementation: Check upside vs downside
        signal = 'HOLD'
        strength = pattern_strength
        reasoning = ""

        # Main decision logic based on your algorithm
        if position_in_range <= 0.3:  # Near ground (bottom 30%)
            # Close to ground - check if it's a good buy
            if upside_potential > downside_risk * 1.5:  # Upside > 1.5x downside
                signal = 'BUY'
                strength = pattern_strength * 0.9
                reasoning = f"Near ground ({position_in_range:.1%} of range), upside {upside_potential:.1f}% vs downside {downside_risk:.1f}%"

                # Boost if trend is supportive
                if trend in ['neutral', 'uptrend']:
                    strength *= 1.1
                    reasoning += f", {trend} trend supportive"

            else:
                signal = 'HOLD'
                strength = pattern_strength * 0.4
                reasoning = f"Near ground but poor upside/downside ratio: {upside_potential:.1f}%/{downside_risk:.1f}%"

        elif position_in_range >= 0.7:  # Near ceiling (top 30%)
            # Close to ceiling - likely not a buy
            signal = 'SELL' if downside_risk > upside_potential else 'HOLD'
            strength = pattern_strength * 0.7 if signal == 'SELL' else pattern_strength * 0.3
            reasoning = f"Near ceiling ({position_in_range:.1%} of range), limited upside {upside_potential:.1f}%"

        elif 0.3 < position_in_range < 0.7:  # Middle range (30-70%)
            # In the middle - evaluate based on trend and upside/downside
            if upside_potential > downside_risk * 2:  # Strong upside advantage
                signal = 'BUY'
                strength = pattern_strength * 0.7
                reasoning = f"Mid-range ({position_in_range:.1%}), strong upside {upside_potential:.1f}% vs {downside_risk:.1f}%"
            elif downside_risk > upside_potential * 1.5:  # Strong downside risk
                signal = 'SELL'
                strength = pattern_strength * 0.6
                reasoning = f"Mid-range ({position_in_range:.1%}), high downside risk {downside_risk:.1f}%"
            else:
                signal = 'HOLD'
                strength = pattern_strength * 0.4
                reasoning = f"Mid-range ({position_in_range:.1%}), balanced risk/reward"

        # Check for breakout scenarios
        if recent_breakout.get('ceiling_breakout', False):
            signal = 'BUY'
            strength = min(pattern_strength * 1.2, 1.0)
            reasoning = f"Ceiling breakout detected - price above {current_ceiling:.2f}"

        elif recent_breakout.get('ground_breakdown', False):
            signal = 'SELL'
            strength = min(pattern_strength * 1.1, 1.0)
            reasoning = f"Ground breakdown detected - price below {current_ground:.2f}"

        # Special case: If we're very close to the transition level (old ceiling = new ground)
        transition_level = pattern_result.get('old_ceiling_became_ground')
        if transition_level and abs(current_price - transition_level) / transition_level < 0.05:
            if current_price > transition_level:
                signal = 'BUY'
                strength = min(pattern_strength * 1.15, 1.0)
                reasoning += f" + at key transition level {transition_level:.2f}"
            else:
                # Below transition level is concerning
                strength *= 0.8
                reasoning += f" - below key transition level {transition_level:.2f}"

        return {
            'signal': signal,
            'strength': min(strength, 1.0),
            'reasoning': reasoning,
            'position_in_range': position_in_range,
            'upside_potential': upside_potential,
            'downside_risk': downside_risk,
            'risk_reward_ratio': upside_potential / max(downside_risk, 1) if downside_risk > 0 else upside_potential
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
            recent_rsi = data['RSI'].tail(10) if len(data) >= 10 else data['RSI']
            rsi_momentum = recent_rsi.diff().mean()
            rsi_trend = 'rising' if rsi_momentum > 0 else 'falling'

            # Generate signal based on RSI level and momentum
            if current_rsi <= extreme_oversold:
                signal = 'BUY'
                strength = 0.95
                reasoning = f"RSI extremely oversold at {current_rsi:.1f}"

            elif current_rsi <= oversold_threshold and rsi_trend == 'rising':
                signal = 'BUY'
                strength = 0.8
                reasoning = f"RSI oversold at {current_rsi:.1f} and rising"

            elif current_rsi >= extreme_overbought:
                signal = 'SELL'
                strength = 0.95
                reasoning = f"RSI extremely overbought at {current_rsi:.1f}"

            elif current_rsi >= overbought_threshold and rsi_trend == 'falling':
                signal = 'SELL'
                strength = 0.8
                reasoning = f"RSI overbought at {current_rsi:.1f} and falling"

            elif 40 <= current_rsi <= 60:
                signal = 'HOLD'
                strength = 0.3
                reasoning = f"RSI neutral at {current_rsi:.1f}"

            else:
                # RSI in middle range but trending
                if current_rsi < 50 and rsi_trend == 'rising':
                    signal = 'BUY'
                    strength = 0.5
                    reasoning = f"RSI {current_rsi:.1f} rising from below 50"
                elif current_rsi > 50 and rsi_trend == 'falling':
                    signal = 'SELL'
                    strength = 0.5
                    reasoning = f"RSI {current_rsi:.1f} falling from above 50"
                else:
                    signal = 'HOLD'
                    strength = 0.2
                    reasoning = f"RSI {current_rsi:.1f} - mixed signals"

            return {
                'signal': signal,
                'strength': strength,
                'reasoning': reasoning,
                'rsi_value': current_rsi,
                'rsi_trend': rsi_trend,
                'rsi_momentum': rsi_momentum
            }

        except Exception as e:
            self.logger.error(f"Error analyzing RSI signal: {str(e)}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'reasoning': 'Error in RSI analysis'
            }

    def _combine_signals(self, pattern_signal: Dict, rsi_signal: Dict, pattern_result: Dict) -> Dict:
        """Combine Ground-Ceiling pattern and RSI signals"""
        pattern_sig = pattern_signal.get('signal', 'HOLD')
        rsi_sig = rsi_signal.get('signal', 'HOLD')

        pattern_strength = pattern_signal.get('strength', 0)
        rsi_strength = rsi_signal.get('strength', 0)

        # Get pattern quality for weighting
        pattern_quality = pattern_result.get('strength', 0.5)

        # For Ground-Ceiling pattern, give more weight to pattern when it's strong
        if pattern_quality > 0.7:
            pattern_weight = 0.8
            rsi_weight = 0.2
        elif pattern_quality > 0.5:
            pattern_weight = 0.7
            rsi_weight = 0.3
        else:
            pattern_weight = 0.6
            rsi_weight = 0.4

        # Calculate weighted strength
        combined_strength = (
                pattern_weight * pattern_strength +
                rsi_weight * rsi_strength
        )

        # Signal combination logic - prioritize pattern for Ground-Ceiling
        if pattern_sig == rsi_sig and pattern_sig != 'HOLD':
            # Both signals agree on direction
            combined_signal = pattern_sig
            confidence_multiplier = 1.3
            agreement_type = 'full_agreement'

        elif pattern_sig != 'HOLD' and pattern_quality > 0.6:
            # Strong pattern signal takes precedence
            combined_signal = pattern_sig
            confidence_multiplier = 1.0 if rsi_sig == 'HOLD' else 0.8
            agreement_type = 'pattern_leading'

        elif rsi_sig != 'HOLD' and pattern_sig == 'HOLD':
            # Only RSI has opinion
            combined_signal = rsi_sig
            confidence_multiplier = 0.6
            agreement_type = 'rsi_only'

        elif pattern_sig != 'HOLD' and rsi_sig != 'HOLD' and pattern_sig != rsi_sig:
            # Conflicting signals - pattern wins for Ground-Ceiling
            combined_signal = pattern_sig
            confidence_multiplier = 0.6
            agreement_type = 'conflict_pattern_wins'

        else:
            # Both HOLD or weak signals
            combined_signal = 'HOLD'
            confidence_multiplier = 0.5
            agreement_type = 'both_neutral'

        combined_confidence = min(combined_strength * confidence_multiplier, 1.0)

        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'pattern_signal': pattern_sig,
            'rsi_signal': rsi_sig,
            'agreement_type': agreement_type,
            'pattern_strength': pattern_strength,
            'rsi_strength': rsi_strength,
            'weights_used': {'pattern': pattern_weight, 'rsi': rsi_weight}
        }

    def _calculate_ground_ceiling_risk_management(self, current_price: float, combined_signal: Dict,
                                                  pattern_result: Dict) -> Dict:
        """Calculate risk management based on ground and ceiling levels"""
        try:
            signal = combined_signal.get('signal', 'HOLD')

            if signal == 'HOLD':
                return {}

            current_ground = pattern_result.get('current_ground', current_price * 0.9)
            current_ceiling = pattern_result.get('current_ceiling', current_price * 1.1)

            # Calculate stop loss and target based on ground/ceiling levels
            if signal == 'BUY':
                # Stop loss: slightly below ground level
                stop_loss = current_ground * 0.97  # 3% below ground for safety

                # Target: ceiling level (with small margin)
                target_price = current_ceiling * 0.98  # Slightly below ceiling

                # Don't target more than 25% gain to be realistic
                max_target = current_price * 1.25
                target_price = min(target_price, max_target)

            else:  # SELL
                # Stop loss: slightly above ceiling level
                stop_loss = current_ceiling * 1.03  # 3% above ceiling for safety

                # Target: ground level (with small margin)
                target_price = current_ground * 1.02  # Slightly above ground

                # Don't target more than 20% loss to be realistic
                min_target = current_price * 0.8
                target_price = max(target_price, min_target)

            # Calculate risk metrics
            risk_per_share = abs(current_price - stop_loss)
            potential_profit = abs(target_price - current_price)
            risk_reward_ratio = potential_profit / risk_per_share if risk_per_share > 0 else 0

            # Position sizing based on pattern confidence and ground/ceiling strength
            pattern_confidence = pattern_result.get('strength', 0.5)
            signal_confidence = combined_signal.get('confidence', 0.5)

            # Consider how close we are to ground (for buys) or ceiling (for sells)
            position_in_range = pattern_result.get('position_in_range', 0.5)

            if signal == 'BUY':
                # Bigger position when closer to ground
                position_bonus = max(0, (0.5 - position_in_range)) * 2  # 0 to 1 bonus
            else:
                # Bigger position when closer to ceiling
                position_bonus = max(0, (position_in_range - 0.5)) * 2  # 0 to 1 bonus

            base_position_multiplier = (pattern_confidence + signal_confidence + position_bonus) / 3

            # Risk-reward adjustment
            if risk_reward_ratio >= 2.0:
                risk_reward_multiplier = 1.2
            elif risk_reward_ratio >= 1.5:
                risk_reward_multiplier = 1.0
            else:
                risk_reward_multiplier = 0.7

            # Base position size (example: $1000 max loss)
            max_loss_per_trade = 1000
            base_position_size = max_loss_per_trade / risk_per_share if risk_per_share > 0 else 1

            final_position_size = base_position_size * base_position_multiplier * risk_reward_multiplier

            return {
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'position_size': round(final_position_size, 0),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'risk_per_share': round(risk_per_share, 2),
                'ground_level': round(current_ground, 2),
                'ceiling_level': round(current_ceiling, 2),
                'confidence_factor': round(base_position_multiplier, 2)
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk management: {str(e)}")
            return {}

    def _generate_final_signal(self, combined_signal: Dict, pattern_result: Dict,
                               current_rsi: float, current_price: float) -> Dict:
        """Generate final signal with comprehensive reasoning"""
        signal = combined_signal.get('signal', 'HOLD')
        base_confidence = combined_signal.get('confidence', 0)

        # Apply confidence filters
        if base_confidence < self.min_confidence and signal != 'HOLD':
            signal = 'HOLD'
            reasoning = f"Signal confidence {base_confidence:.2f} below minimum threshold {self.min_confidence}"
        else:
            # Build comprehensive reasoning
            current_ground = pattern_result.get('current_ground', 0)
            current_ceiling = pattern_result.get('current_ceiling', 0)
            position_in_range = pattern_result.get('position_in_range', 0.5)
            upside_potential = pattern_result.get('upside_potential', 0)
            downside_risk = pattern_result.get('downside_risk', 0)
            trend = pattern_result.get('trend', 'neutral')
            pattern_strength = pattern_result.get('strength', 0)
            agreement_type = combined_signal.get('agreement_type', 'unknown')

            transition_level = pattern_result.get('old_ceiling_became_ground')

            reasoning_parts = []

            # Ground-Ceiling pattern information
            reasoning_parts.append(
                f"Ground-Ceiling pattern: Ground ${current_ground:.2f}, Ceiling ${current_ceiling:.2f}")
            reasoning_parts.append(
                f"Position: {position_in_range:.1%} of range ({self._determine_zone(position_in_range)})")
            reasoning_parts.append(f"Upside: {upside_potential:.1f}%, Downside: {downside_risk:.1f}%")

            # Key transition information
            if transition_level:
                reasoning_parts.append(f"Old ceiling ${transition_level:.2f} became new ground")

            # Signal agreement
            if agreement_type == 'full_agreement':
                reasoning_parts.append("Pattern and RSI agree")
            elif agreement_type == 'pattern_leading':
                reasoning_parts.append("Strong pattern signal")
            elif agreement_type == 'conflict_pattern_wins':
                reasoning_parts.append("Pattern overrides RSI conflict")

            # RSI and trend
            reasoning_parts.append(f"RSI: {current_rsi:.1f}, Trend: {trend}")

            # Breakout information
            recent_breakout = pattern_result.get('recent_breakout', {})
            if recent_breakout.get('ceiling_breakout'):
                reasoning_parts.append("Recent ceiling breakout")
            elif recent_breakout.get('ground_breakdown'):
                reasoning_parts.append("Recent ground breakdown")

            reasoning = "; ".join(reasoning_parts)

        # Final confidence adjustments
        final_confidence = base_confidence

        # Boost confidence for strong patterns near optimal zones
        pattern_quality = pattern_result.get('strength', 0.5)
        position_in_range = pattern_result.get('position_in_range', 0.5)

        if pattern_quality > 0.7:
            if signal == 'BUY' and position_in_range <= 0.3:  # Buy near ground
                final_confidence *= 1.15
            elif signal == 'SELL' and position_in_range >= 0.7:  # Sell near ceiling
                final_confidence *= 1.1

        # Boost for good risk-reward ratio
        upside = pattern_result.get('upside_potential', 0)
        downside = pattern_result.get('downside_risk', 1)
        if upside / downside > 2:  # Good risk-reward
            final_confidence *= 1.1

        # Reduce confidence if price is in middle range with mixed signals
        if 0.4 <= position_in_range <= 0.6 and agreement_type != 'full_agreement':
            final_confidence *= 0.9

        # Cap at 95%
        final_confidence = min(final_confidence, 0.95)

        return {
            'signal': signal,
            'confidence': final_confidence,
            'reasoning': reasoning
        }

    def _determine_zone(self, position_in_range: float) -> str:
        """Determine which zone the price is in"""
        if position_in_range <= 0.25:
            return "near ground"
        elif position_in_range <= 0.4:
            return "lower range"
        elif position_in_range <= 0.6:
            return "middle range"
        elif position_in_range <= 0.8:
            return "upper range"
        else:
            return "near ceiling"

    def analyze_entry_timing(self, data: pd.DataFrame, pattern_result: Dict) -> Dict:
        """
        Analyze optimal entry timing based on ground-ceiling pattern
        """
        try:
            current_price = data['Close'].iloc[-1]
            current_ground = pattern_result.get('current_ground', current_price)
            current_ceiling = pattern_result.get('current_ceiling', current_price)
            position_in_range = pattern_result.get('position_in_range', 0.5)

            # Analyze recent price action near ground/ceiling levels
            recent_data = data.tail(20)  # Last 20 days
            recent_prices = recent_data['Close'].values

            # Check how many times we've tested the ground recently
            ground_tests = sum(1 for price in recent_prices
                               if abs(price - current_ground) / current_ground <= 0.03)

            # Check how many times we've tested the ceiling recently
            ceiling_tests = sum(1 for price in recent_prices
                                if abs(price - current_ceiling) / current_ceiling <= 0.03)

            # Determine entry quality
            if position_in_range <= 0.3 and ground_tests >= 2:
                entry_quality = 'excellent'
                entry_reason = f"Multiple ground tests ({ground_tests}), strong support confirmed"
            elif position_in_range <= 0.3 and ground_tests >= 1:
                entry_quality = 'good'
                entry_reason = f"Near ground with recent test, good entry point"
            elif position_in_range <= 0.5:
                entry_quality = 'fair'
                entry_reason = f"Below mid-range, reasonable entry"
            elif position_in_range >= 0.7:
                entry_quality = 'poor'
                entry_reason = f"Near ceiling, poor risk/reward"
            else:
                entry_quality = 'average'
                entry_reason = f"Mid-range entry, monitor for better levels"

            # Calculate optimal entry price
            if position_in_range > 0.4:
                optimal_entry = current_ground * 1.05  # 5% above ground
            else:
                optimal_entry = current_price  # Current price is good

            return {
                'entry_quality': entry_quality,
                'entry_reason': entry_reason,
                'optimal_entry_price': round(optimal_entry, 2),
                'current_vs_optimal': round(((current_price - optimal_entry) / optimal_entry) * 100, 1),
                'ground_tests': ground_tests,
                'ceiling_tests': ceiling_tests,
                'wait_for_better_entry': position_in_range > 0.6
            }

        except Exception as e:
            self.logger.error(f"Error analyzing entry timing: {str(e)}")
            return {
                'entry_quality': 'unknown',
                'entry_reason': 'Analysis error',
                'optimal_entry_price': current_price,
                'current_vs_optimal': 0,
                'ground_tests': 0,
                'ceiling_tests': 0,
                'wait_for_better_entry': False
            }

    def calculate_position_sizing(self, account_size: float, pattern_result: Dict,
                                  combined_signal: Dict) -> Dict:
        """Calculate position sizing based on ground-ceiling pattern strength"""
        try:
            current_price = pattern_result.get('current_price', 100)
            current_ground = pattern_result.get('current_ground', current_price * 0.9)
            pattern_strength = pattern_result.get('strength', 0.5)
            signal_confidence = combined_signal.get('confidence', 0.5)
            position_in_range = pattern_result.get('position_in_range', 0.5)

            # Base risk per trade (2% of account)
            base_risk_amount = account_size * 0.02

            # Calculate stop loss distance
            stop_loss_distance = abs(current_price - (current_ground * 0.97))

            # Base position size
            base_shares = int(base_risk_amount / stop_loss_distance) if stop_loss_distance > 0 else 0

            # Adjust based on pattern strength
            strength_multiplier = 0.5 + (pattern_strength * 1.0)  # 0.5 to 1.5

            # Adjust based on signal confidence
            confidence_multiplier = 0.7 + (signal_confidence * 0.6)  # 0.7 to 1.3

            # Adjust based on position in range (bigger size near ground)
            if position_in_range <= 0.3:
                range_multiplier = 1.2  # 20% bigger near ground
            elif position_in_range <= 0.5:
                range_multiplier = 1.0
            else:
                range_multiplier = 0.8  # 20% smaller away from ground

            # Final position size
            final_shares = int(base_shares * strength_multiplier * confidence_multiplier * range_multiplier)

            # Position value and risk metrics
            position_value = final_shares * current_price
            max_loss = final_shares * stop_loss_distance
            account_risk_pct = (max_loss / account_size) * 100 if account_size > 0 else 0

            return {
                'recommended_shares': final_shares,
                'position_value': round(position_value, 2),
                'max_loss': round(max_loss, 2),
                'account_risk_pct': round(account_risk_pct, 2),
                'strength_multiplier': round(strength_multiplier, 2),
                'confidence_multiplier': round(confidence_multiplier, 2),
                'range_multiplier': round(range_multiplier, 2),
                'stop_loss_price': round(current_ground * 0.97, 2),
                'position_too_large': account_risk_pct > 3.0,
                'position_very_small': final_shares < 10
            }

        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {str(e)}")
            return {
                'recommended_shares': 0,
                'position_value': 0,
                'max_loss': 0,
                'account_risk_pct': 0,
                'error': str(e)
            }

    def generate_trade_plan(self, data: pd.DataFrame, pattern_result: Dict,
                            account_size: float = 10000) -> Dict:
        """
        Generate a complete trade plan based on ground-ceiling analysis
        """
        try:
            current_data = data.iloc[-1]
            signal_result = self.generate_signal(data, pattern_result, current_data)

            if signal_result['signal'] != 'BUY':
                return {
                    'trade_recommended': False,
                    'reason': f"Signal is {signal_result['signal']}, not a buy",
                    'signal_result': signal_result
                }

            # Get detailed analysis
            entry_analysis = self.analyze_entry_timing(data, pattern_result)
            position_analysis = self.calculate_position_sizing(account_size, pattern_result,
                                                               {'confidence': signal_result['confidence']})

            current_price = pattern_result.get('current_price', 0)
            current_ground = pattern_result.get('current_ground', 0)
            current_ceiling = pattern_result.get('current_ceiling', 0)

            # Generate complete trade plan
            trade_plan = {
                'trade_recommended': True,
                'symbol': 'STOCK',  # To be filled in by calling function
                'signal': signal_result['signal'],
                'confidence': signal_result['confidence'],
                'entry_quality': entry_analysis['entry_quality'],

                # Entry details
                'current_price': current_price,
                'optimal_entry': entry_analysis['optimal_entry_price'],
                'entry_timing': 'immediate' if not entry_analysis['wait_for_better_entry'] else 'wait_for_better_level',

                # Levels
                'ground_level': current_ground,
                'ceiling_level': current_ceiling,
                'stop_loss': round(current_ground * 0.97, 2),
                'target_1': round(current_ceiling * 0.95, 2),  # Conservative target
                'target_2': round(current_ceiling * 0.98, 2),  # Aggressive target

                # Position sizing
                'recommended_shares': position_analysis['recommended_shares'],
                'position_value': position_analysis['position_value'],
                'max_risk': position_analysis['max_loss'],
                'risk_pct': position_analysis['account_risk_pct'],

                # Risk/Reward
                'upside_potential': pattern_result.get('upside_potential', 0),
                'downside_risk': pattern_result.get('downside_risk', 0),
                'risk_reward_ratio': signal_result.get('risk_reward_ratio', 0),

                # Pattern details
                'pattern_strength': pattern_result.get('strength', 0),
                'position_in_range': pattern_result.get('position_in_range', 0),
                'old_ceiling_became_ground': pattern_result.get('old_ceiling_became_ground'),
                'transition_level': pattern_result.get('old_ceiling_became_ground'),

                # Reasoning
                'reasoning': signal_result['reasoning'],
                'entry_reason': entry_analysis['entry_reason'],

                # Warnings
                'warnings': self._generate_trade_warnings(pattern_result, signal_result, entry_analysis,
                                                          position_analysis)
            }

            return trade_plan

        except Exception as e:
            self.logger.error(f"Error generating trade plan: {str(e)}")
            return {
                'trade_recommended': False,
                'reason': f"Error generating trade plan: {str(e)}",
                'error': str(e)
            }

    def _generate_trade_warnings(self, pattern_result: Dict, signal_result: Dict,
                                 entry_analysis: Dict, position_analysis: Dict) -> List[str]:
        """Generate warnings for the trade plan"""
        warnings = []

        try:
            # Pattern warnings
            pattern_strength = pattern_result.get('strength', 0)
            if pattern_strength < 0.6:
                warnings.append(f"Weak pattern strength ({pattern_strength:.2f})")

            # Entry timing warnings
            if entry_analysis.get('entry_quality') == 'poor':
                warnings.append("Poor entry timing - near ceiling")
            elif entry_analysis.get('wait_for_better_entry'):
                warnings.append("Consider waiting for price to move closer to ground")

            # Position size warnings
            if position_analysis.get('position_too_large'):
                warnings.append(f"Position risk ({position_analysis.get('account_risk_pct', 0):.1f}%) exceeds 3%")
            elif position_analysis.get('position_very_small'):
                warnings.append("Position size very small - consider larger account or different stock")

            # Confidence warnings
            confidence = signal_result.get('confidence', 0)
            if confidence < 0.7:
                warnings.append(f"Low signal confidence ({confidence:.2f})")

            # Risk-reward warnings
            risk_reward = signal_result.get('risk_reward_ratio', 0)
            if risk_reward < 1.5:
                warnings.append(f"Poor risk-reward ratio ({risk_reward:.2f})")

            # Breakout warnings
            recent_breakout = pattern_result.get('recent_breakout', {})
            if recent_breakout.get('ground_breakdown'):
                warnings.append("Recent ground breakdown detected - pattern may be failing")

            return warnings

        except Exception as e:
            return [f"Error generating warnings: {str(e)}"]

    def backtest_pattern(self, data: pd.DataFrame, pattern_result: Dict,
                         lookback_days: int = 60) -> Dict:
        """
        Backtest the ground-ceiling pattern over recent history
        """
        try:
            if len(data) < lookback_days + 20:
                return {'error': 'Insufficient data for backtesting'}

            # Use historical data for backtesting
            backtest_data = data.iloc[:-lookback_days]  # Exclude recent data
            test_data = data.iloc[-lookback_days:]  # Recent data to test on

            current_ground = pattern_result.get('current_ground', 0)
            current_ceiling = pattern_result.get('current_ceiling', 0)

            # Simulate trades
            trades = []
            position = None

            for i in range(len(test_data)):
                current_price = test_data['Close'].iloc[i]
                current_date = test_data.index[i]

                # Calculate position in range
                if current_ceiling > current_ground:
                    position_in_range = (current_price - current_ground) / (current_ceiling - current_ground)
                else:
                    position_in_range = 0.5

                # Entry logic (simplified)
                if position is None and position_in_range <= 0.3:
                    # Enter long position near ground
                    position = {
                        'entry_price': current_price,
                        'entry_date': current_date,
                        'stop_loss': current_ground * 0.97,
                        'target': current_ceiling * 0.95,
                        'type': 'BUY'
                    }

                # Exit logic
                elif position is not None:
                    exit_reason = None
                    exit_price = current_price

                    # Check stop loss
                    if current_price <= position['stop_loss']:
                        exit_reason = 'stop_loss'
                    # Check target
                    elif current_price >= position['target']:
                        exit_reason = 'target_hit'
                    # Check ceiling approach
                    elif position_in_range >= 0.9:
                        exit_reason = 'near_ceiling'

                    if exit_reason:
                        # Record trade
                        pnl = exit_price - position['entry_price']
                        pnl_pct = (pnl / position['entry_price']) * 100

                        trades.append({
                            'entry_date': position['entry_date'],
                            'entry_price': position['entry_price'],
                            'exit_date': current_date,
                            'exit_price': exit_price,
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'days_held': (current_date - position['entry_date']).days
                        })

                        position = None

            # Calculate backtest metrics
            if not trades:
                return {
                    'total_trades': 0,
                    'no_trades_reason': 'No entry signals generated during backtest period'
                }

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / len(trades)
            win_rate = len(winning_trades) / len(trades) * 100

            avg_winner = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loser = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

            profit_factor = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')

            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(win_rate, 1),
                'total_pnl': round(total_pnl, 2),
                'avg_pnl_per_trade': round(avg_pnl, 2),
                'avg_winner': round(avg_winner, 2),
                'avg_loser': round(avg_loser, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_hold_days': round(sum(t['days_held'] for t in trades) / len(trades), 1),
                'trades_detail': trades
            }

        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            return {'error': f'Backtesting failed: {str(e)}'}

    def get_pattern_summary(self, pattern_result: Dict) -> str:
        """
        Generate a human-readable summary of the ground-ceiling pattern
        """
        try:
            if not pattern_result:
                return "No pattern detected"

            current_price = pattern_result.get('current_price', 0)
            current_ground = pattern_result.get('current_ground', 0)
            current_ceiling = pattern_result.get('current_ceiling', 0)
            position_in_range = pattern_result.get('position_in_range', 0.5)
            upside_potential = pattern_result.get('upside_potential', 0)
            downside_risk = pattern_result.get('downside_risk', 0)
            pattern_strength = pattern_result.get('strength', 0)
            transition_level = pattern_result.get('old_ceiling_became_ground')

            zone = self._determine_zone(position_in_range)

            summary_parts = []

            # Basic pattern info
            summary_parts.append(f"Ground-Ceiling Pattern Detected (Strength: {pattern_strength:.2f})")
            summary_parts.append(f"Current Price: ${current_price:.2f}")
            summary_parts.append(f"Trading Range: ${current_ground:.2f} (ground) to ${current_ceiling:.2f} (ceiling)")
            summary_parts.append(f"Position: {position_in_range:.1%} of range ({zone})")

            # Key transition info
            if transition_level:
                summary_parts.append(f"Key Level: Old ceiling ${transition_level:.2f} became new ground")

            # Risk/Reward
            summary_parts.append(f"Upside Potential: {upside_potential:.1f}% to ceiling")
            summary_parts.append(f"Downside Risk: {downside_risk:.1f}% to ground")

            # Recommendation hint
            if position_in_range <= 0.3 and upside_potential > downside_risk * 1.5:
                summary_parts.append("⭐ POTENTIAL BUY ZONE - Near ground with good upside")
            elif position_in_range >= 0.7:
                summary_parts.append("⚠️ CAUTION ZONE - Near ceiling, limited upside")
            else:
                summary_parts.append("📊 MONITORING ZONE - Mid-range, evaluate risk/reward")

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Error generating summary: {str(e)}"