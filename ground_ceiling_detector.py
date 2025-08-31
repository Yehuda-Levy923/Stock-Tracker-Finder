import numpy as np
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
import logging
from typing import Dict, Optional, Tuple, List


class GroundCeilingDetector:
    """Class to detect ground (support) and ceiling (resistance) patterns with level transitions"""

    def __init__(self, lookback_days: int = 365, min_touch_count: int = 2, level_tolerance: float = 0.05):
        """
        Initialize the ground and ceiling detector

        Args:
            lookback_days: How far back to look for patterns (default ~1 year)
            min_touch_count: Minimum times a level must be touched to be valid
            level_tolerance: Tolerance for price level matching (5% default)
        """
        self.lookback_days = lookback_days
        self.min_touch_count = min_touch_count
        self.level_tolerance = level_tolerance
        self.logger = logging.getLogger(__name__)

        # Pattern parameters
        self.min_data_points = 200  # Minimum data points needed
        self.peak_distance = 10  # Minimum distance between peaks/troughs
        self.prominence_factor = 0.02  # Peak prominence as % of price range

    def detect_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect ground and ceiling pattern with level transitions

        Args:
            data: DataFrame with stock price data

        Returns:
            Dictionary with pattern information or None if no pattern found
        """
        if len(data) < self.min_data_points:
            self.logger.warning("Insufficient data points for pattern detection")
            return None

        try:
            # Use the specified lookback period
            lookback_data = data.tail(min(self.lookback_days, len(data)))
            prices = lookback_data['Close'].values
            highs = lookback_data['High'].values
            lows = lookback_data['Low'].values
            dates = lookback_data.index

            # Find all significant peaks and troughs
            peaks_info = self._find_significant_peaks(highs, dates)
            troughs_info = self._find_significant_troughs(lows, dates)

            if len(peaks_info) < 2 or len(troughs_info) < 2:
                self.logger.debug("Not enough peaks/troughs found")
                return None

            # Identify ceiling and ground levels
            ceiling_analysis = self._analyze_ceiling_levels(peaks_info, prices)
            ground_analysis = self._analyze_ground_levels(troughs_info, prices)

            # Check for the key pattern: old ceiling becomes new ground
            transition_analysis = self._detect_ceiling_to_ground_transition(
                ceiling_analysis, ground_analysis, peaks_info, troughs_info
            )

            if not transition_analysis['found_transition']:
                self.logger.debug("No ceiling-to-ground transition found")
                return None

            # Get current levels
            current_price = prices[-1]
            current_ground = ground_analysis['current_ground']
            current_ceiling = ceiling_analysis['current_ceiling']

            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                ceiling_analysis, ground_analysis, transition_analysis, current_price
            )

            if pattern_strength < 0.4:  # Minimum threshold
                return None

            # Determine current position and signal
            position_analysis = self._analyze_current_position(
                current_price, current_ground, current_ceiling, prices
            )

            # Calculate upside/downside potential
            upside_downside = self._calculate_upside_downside(
                current_price, current_ground, current_ceiling
            )

            # Determine trend
            trend = self._calculate_trend(prices)

            result = {
                'strength': pattern_strength,
                'current_ground': current_ground,
                'current_ceiling': current_ceiling,
                'old_ceiling_became_ground': transition_analysis['transition_level'],
                'transition_date': transition_analysis['transition_date'],
                'current_price': current_price,
                'position_in_range': position_analysis['position_pct'],
                'upside_potential': upside_downside['upside_pct'],
                'downside_risk': upside_downside['downside_pct'],
                'upside_target': current_ceiling,
                'downside_target': current_ground,
                'trend': trend,
                'pattern_type': 'ground_ceiling',
                'ceiling_touches': ceiling_analysis['touch_count'],
                'ground_touches': ground_analysis['touch_count'],
                'ceiling_strength': ceiling_analysis['strength'],
                'ground_strength': ground_analysis['strength'],
                'transition_strength': transition_analysis['strength'],
                'confidence': min(pattern_strength * 100, 95),
                'all_ceilings': ceiling_analysis['all_levels'],
                'all_grounds': ground_analysis['all_levels'],
                'recent_breakout': self._check_recent_breakout(prices, current_ceiling, current_ground)
            }

            self.logger.debug(f"Ground-Ceiling pattern detected - Ground: {current_ground:.2f}, "
                              f"Ceiling: {current_ceiling:.2f}, Strength: {pattern_strength:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in ground-ceiling detection: {str(e)}")
            return None

    def _find_significant_peaks(self, prices: np.ndarray, dates) -> List[Dict]:
        """Find significant peaks in the price data"""
        try:
            price_range = np.max(prices) - np.min(prices)
            min_prominence = price_range * self.prominence_factor

            peaks, properties = find_peaks(
                prices,
                distance=self.peak_distance,
                prominence=min_prominence
            )

            peaks_info = []
            for i, peak_idx in enumerate(peaks):
                peaks_info.append({
                    'index': peak_idx,
                    'price': prices[peak_idx],
                    'date': dates[peak_idx],
                    'prominence': properties['prominences'][i]
                })

            # Sort by date
            peaks_info.sort(key=lambda x: x['date'])

            return peaks_info

        except Exception as e:
            self.logger.debug(f"Error finding peaks: {str(e)}")
            return []

    def _find_significant_troughs(self, prices: np.ndarray, dates) -> List[Dict]:
        """Find significant troughs in the price data"""
        try:
            # Find troughs by finding peaks in inverted data
            price_range = np.max(prices) - np.min(prices)
            min_prominence = price_range * self.prominence_factor

            troughs, properties = find_peaks(
                -prices,  # Invert to find troughs
                distance=self.peak_distance,
                prominence=min_prominence
            )

            troughs_info = []
            for i, trough_idx in enumerate(troughs):
                troughs_info.append({
                    'index': trough_idx,
                    'price': prices[trough_idx],
                    'date': dates[trough_idx],
                    'prominence': properties['prominences'][i]
                })

            # Sort by date
            troughs_info.sort(key=lambda x: x['date'])

            return troughs_info

        except Exception as e:
            self.logger.debug(f"Error finding troughs: {str(e)}")
            return []

    def _analyze_ceiling_levels(self, peaks_info: List[Dict], prices: np.ndarray) -> Dict:
        """Analyze ceiling (resistance) levels"""
        try:
            if not peaks_info:
                return {'current_ceiling': np.max(prices), 'strength': 0, 'touch_count': 0, 'all_levels': []}

            # Group peaks by similar price levels
            ceiling_levels = self._group_similar_levels([p['price'] for p in peaks_info])

            # Find the most significant ceiling level
            best_ceiling = None
            best_strength = 0

            for level, touches in ceiling_levels.items():
                if len(touches) >= self.min_touch_count:
                    # Calculate strength based on touch count and recency
                    touch_count = len(touches)
                    recency_bonus = 1.0

                    # Bonus for recent touches
                    recent_touches = [t for t in touches if t >= len(prices) - 60]  # Last 60 days
                    if recent_touches:
                        recency_bonus = 1.2

                    strength = touch_count * recency_bonus

                    if strength > best_strength:
                        best_ceiling = level
                        best_strength = strength

            # If no valid ceiling found, use highest peak
            if best_ceiling is None:
                best_ceiling = max([p['price'] for p in peaks_info])
                touch_count = 1
                strength = 0.3
            else:
                touch_count = len(ceiling_levels[best_ceiling])
                strength = min(best_strength / 10, 1.0)  # Normalize

            return {
                'current_ceiling': best_ceiling,
                'strength': strength,
                'touch_count': touch_count,
                'all_levels': list(ceiling_levels.keys())
            }

        except Exception as e:
            self.logger.debug(f"Error analyzing ceiling levels: {str(e)}")
            return {'current_ceiling': np.max(prices), 'strength': 0, 'touch_count': 0, 'all_levels': []}

    def _analyze_ground_levels(self, troughs_info: List[Dict], prices: np.ndarray) -> Dict:
        """Analyze ground (support) levels"""
        try:
            if not troughs_info:
                return {'current_ground': np.min(prices), 'strength': 0, 'touch_count': 0, 'all_levels': []}

            # Group troughs by similar price levels
            ground_levels = self._group_similar_levels([t['price'] for t in troughs_info])

            # Find the most significant ground level
            best_ground = None
            best_strength = 0

            for level, touches in ground_levels.items():
                if len(touches) >= self.min_touch_count:
                    # Calculate strength based on touch count and recency
                    touch_count = len(touches)
                    recency_bonus = 1.0

                    # Bonus for recent touches
                    recent_touches = [t for t in touches if t >= len(prices) - 60]  # Last 60 days
                    if recent_touches:
                        recency_bonus = 1.2

                    strength = touch_count * recency_bonus

                    if strength > best_strength:
                        best_ground = level
                        best_strength = strength

            # If no valid ground found, use lowest trough
            if best_ground is None:
                best_ground = min([t['price'] for t in troughs_info])
                touch_count = 1
                strength = 0.3
            else:
                touch_count = len(ground_levels[best_ground])
                strength = min(best_strength / 10, 1.0)  # Normalize

            return {
                'current_ground': best_ground,
                'strength': strength,
                'touch_count': touch_count,
                'all_levels': list(ground_levels.keys())
            }

        except Exception as e:
            self.logger.debug(f"Error analyzing ground levels: {str(e)}")
            return {'current_ground': np.min(prices), 'strength': 0, 'touch_count': 0, 'all_levels': []}

    def _group_similar_levels(self, levels: List[float]) -> Dict[float, List[int]]:
        """Group similar price levels together"""
        if not levels:
            return {}

        grouped = {}
        levels_with_index = [(price, idx) for idx, price in enumerate(levels)]
        levels_with_index.sort()

        for price, idx in levels_with_index:
            # Find if this price is similar to any existing group
            found_group = False
            for group_price in list(grouped.keys()):
                if abs(price - group_price) / group_price <= self.level_tolerance:
                    grouped[group_price].append(idx)
                    found_group = True
                    break

            if not found_group:
                grouped[price] = [idx]

        return grouped

    def _detect_ceiling_to_ground_transition(self, ceiling_analysis: Dict, ground_analysis: Dict,
                                             peaks_info: List[Dict], troughs_info: List[Dict]) -> Dict:
        """Detect if an old ceiling became a new ground (key pattern requirement)"""
        try:
            current_ceiling = ceiling_analysis['current_ceiling']
            current_ground = ground_analysis['current_ground']

            # Look for old ceiling levels that might have become the new ground
            all_ceiling_levels = ceiling_analysis['all_levels']

            transition_found = False
            transition_level = None
            transition_date = None
            transition_strength = 0

            # Check if any old ceiling is close to current ground
            for old_ceiling in all_ceiling_levels:
                if old_ceiling != current_ceiling:  # Not the current ceiling
                    # Is this old ceiling close to our current ground?
                    if abs(old_ceiling - current_ground) / current_ground <= self.level_tolerance * 1.5:

                        # Find when this level transitioned from ceiling to ground
                        ceiling_touches = self._find_level_touches(peaks_info, old_ceiling, 'peak')
                        ground_touches = self._find_level_touches(troughs_info, old_ceiling, 'trough')

                        # Check timing: ceiling touches should be before ground touches
                        if ceiling_touches and ground_touches:
                            last_ceiling_touch = max([t['date'] for t in ceiling_touches])
                            first_ground_touch = min([t['date'] for t in ground_touches])

                            if first_ground_touch > last_ceiling_touch:
                                transition_found = True
                                transition_level = old_ceiling
                                transition_date = first_ground_touch
                                transition_strength = len(ceiling_touches) + len(ground_touches)
                                break

            # Alternative: Check if current ground was previously a ceiling
            if not transition_found:
                ceiling_touches = self._find_level_touches(peaks_info, current_ground, 'peak')
                if ceiling_touches:
                    # This ground level was previously a ceiling
                    transition_found = True
                    transition_level = current_ground
                    transition_date = ceiling_touches[-1]['date']  # Last time it acted as ceiling
                    transition_strength = len(ceiling_touches)

            return {
                'found_transition': transition_found,
                'transition_level': transition_level,
                'transition_date': transition_date,
                'strength': min(transition_strength / 5, 1.0)  # Normalize
            }

        except Exception as e:
            self.logger.debug(f"Error detecting transition: {str(e)}")
            return {'found_transition': False, 'transition_level': None, 'transition_date': None, 'strength': 0}

    def _find_level_touches(self, points_info: List[Dict], target_level: float, point_type: str) -> List[Dict]:
        """Find touches of a specific price level"""
        touches = []
        for point in points_info:
            if abs(point['price'] - target_level) / target_level <= self.level_tolerance:
                touches.append(point)
        return touches

    def _calculate_pattern_strength(self, ceiling_analysis: Dict, ground_analysis: Dict,
                                    transition_analysis: Dict, current_price: float) -> float:
        """Calculate overall pattern strength"""
        try:
            # Base strength from ceiling and ground strength
            ceiling_strength = ceiling_analysis['strength']
            ground_strength = ground_analysis['strength']
            base_strength = (ceiling_strength + ground_strength) / 2

            # Bonus for finding the transition pattern
            transition_bonus = transition_analysis['strength'] * 0.4

            # Bonus for clear range (ceiling well above ground)
            current_ceiling = ceiling_analysis['current_ceiling']
            current_ground = ground_analysis['current_ground']

            if current_ceiling > current_ground:
                range_ratio = (current_ceiling - current_ground) / current_ground
                range_bonus = min(range_ratio / 0.5, 0.3)  # Max 30% bonus for good range
            else:
                range_bonus = 0

            # Penalty if current price is outside the range
            if current_price > current_ceiling * 1.1 or current_price < current_ground * 0.9:
                outside_penalty = 0.2
            else:
                outside_penalty = 0

            total_strength = base_strength + transition_bonus + range_bonus - outside_penalty

            return max(0, min(total_strength, 1.0))

        except Exception as e:
            self.logger.debug(f"Error calculating pattern strength: {str(e)}")
            return 0

    def _analyze_current_position(self, current_price: float, ground: float, ceiling: float,
                                  prices: np.ndarray) -> Dict:
        """Analyze where current price sits relative to ground and ceiling"""
        try:
            if ceiling <= ground:
                return {'position_pct': 0.5, 'zone': 'undefined'}

            # Calculate position as percentage between ground and ceiling
            position_pct = (current_price - ground) / (ceiling - ground)
            position_pct = max(0, min(1, position_pct))  # Clamp between 0 and 1

            # Determine zone
            if position_pct <= 0.25:
                zone = 'near_ground'
            elif position_pct <= 0.4:
                zone = 'lower_range'
            elif position_pct <= 0.6:
                zone = 'middle_range'
            elif position_pct <= 0.8:
                zone = 'upper_range'
            else:
                zone = 'near_ceiling'

            return {
                'position_pct': position_pct,
                'zone': zone
            }

        except Exception as e:
            self.logger.debug(f"Error analyzing position: {str(e)}")
            return {'position_pct': 0.5, 'zone': 'undefined'}

    def _calculate_upside_downside(self, current_price: float, ground: float, ceiling: float) -> Dict:
        """Calculate upside potential and downside risk"""
        try:
            upside_pct = ((ceiling - current_price) / current_price) * 100 if current_price > 0 else 0
            downside_pct = ((current_price - ground) / current_price) * 100 if current_price > 0 else 0

            # Ensure reasonable bounds
            upside_pct = max(-50, min(200, upside_pct))  # Cap between -50% and +200%
            downside_pct = max(-50, min(100, downside_pct))  # Cap between -50% and +100%

            return {
                'upside_pct': round(upside_pct, 1),
                'downside_pct': round(downside_pct, 1),
                'risk_reward_ratio': round(upside_pct / downside_pct, 2) if downside_pct > 0 else 0
            }

        except Exception as e:
            self.logger.debug(f"Error calculating upside/downside: {str(e)}")
            return {'upside_pct': 0, 'downside_pct': 0, 'risk_reward_ratio': 0}

    def _calculate_trend(self, prices: np.ndarray) -> str:
        """Calculate overall trend direction"""
        if len(prices) < 20:
            return "neutral"

        # Compare recent average to older average
        recent_avg = np.mean(prices[-20:])
        older_avg = np.mean(prices[-40:-20]) if len(prices) >= 40 else np.mean(prices[:-20])

        change_pct = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0

        if change_pct > 0.05:  # 5% threshold
            return "uptrend"
        elif change_pct < -0.05:
            return "downtrend"
        else:
            return "neutral"

    def _check_recent_breakout(self, prices: np.ndarray, ceiling: float, ground: float) -> Dict:
        """Check for recent breakouts above ceiling or below ground"""
        try:
            recent_prices = prices[-20:]  # Last 20 days
            current_price = prices[-1]

            # Check for ceiling breakout
            ceiling_breakout = False
            if current_price > ceiling * 1.02:  # 2% above ceiling
                days_above = sum(1 for p in recent_prices if p > ceiling)
                if days_above >= 3:  # At least 3 days above
                    ceiling_breakout = True

            # Check for ground breakdown
            ground_breakdown = False
            if current_price < ground * 0.98:  # 2% below ground
                days_below = sum(1 for p in recent_prices if p < ground)
                if days_below >= 3:  # At least 3 days below
                    ground_breakdown = True

            return {
                'ceiling_breakout': ceiling_breakout,
                'ground_breakdown': ground_breakdown,
                'breakout_strength': max(
                    (current_price - ceiling) / ceiling if ceiling_breakout else 0,
                    (ground - current_price) / ground if ground_breakdown else 0
                )
            }

        except Exception as e:
            self.logger.debug(f"Error checking breakouts: {str(e)}")
            return {'ceiling_breakout': False, 'ground_breakdown': False, 'breakout_strength': 0}

    def validate_pattern(self, data: pd.DataFrame, pattern_result: Dict) -> Dict:
        """
        Validate the detected ground-ceiling pattern

        Args:
            data: Original data
            pattern_result: Pattern detection result

        Returns:
            Validation results
        """
        try:
            if not pattern_result or pattern_result.get('strength', 0) < 0.4:
                return {'valid': False, 'reason': 'Pattern strength too low'}

            current_ground = pattern_result['current_ground']
            current_ceiling = pattern_result['current_ceiling']
            current_price = pattern_result['current_price']

            # Check 1: Ceiling should be above ground
            if current_ceiling <= current_ground:
                return {'valid': False, 'reason': 'Ceiling not above ground'}

            # Check 2: Reasonable range size
            range_pct = (current_ceiling - current_ground) / current_ground
            if range_pct < 0.1:  # Less than 10% range
                return {'valid': False, 'reason': f'Range too small: {range_pct:.1%}'}

            # Check 3: Transition pattern found
            if not pattern_result.get('old_ceiling_became_ground'):
                return {'valid': False, 'reason': 'No ceiling-to-ground transition found'}

            # Check 4: Sufficient touches
            ceiling_touches = pattern_result.get('ceiling_touches', 0)
            ground_touches = pattern_result.get('ground_touches', 0)
            if ceiling_touches < 2 or ground_touches < 2:
                return {'valid': False, 'reason': 'Insufficient level touches'}

            return {
                'valid': True,
                'quality_score': pattern_result['strength'],
                'range_pct': range_pct,
                'ceiling_touches': ceiling_touches,
                'ground_touches': ground_touches,
                'position_in_range': pattern_result['position_in_range']
            }

        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}