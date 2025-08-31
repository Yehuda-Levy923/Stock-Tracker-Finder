import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import logging
from typing import Dict, Optional, Tuple, List


class SineWaveDetector:
    """Class to detect sine wave patterns in stock price data"""

    def __init__(self, tolerance: float = 0.15):
        """
        Initialize the sine wave detector

        Args:
            tolerance: Tolerance level for pattern matching (default 15%)
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)

        # Pattern parameters
        self.short_cycle_days = 90  # ~3 months
        self.long_cycle_days = 270  # ~9 months (3 * short_cycle)
        self.min_data_points = 200  # Minimum data points needed

    def detect_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect sine wave pattern in price data

        Args:
            data: DataFrame with stock price data

        Returns:
            Dictionary with pattern information or None if no pattern found
        """
        if len(data) < self.min_data_points:
            self.logger.warning("Insufficient data points for pattern detection")
            return None

        try:
            # Use closing prices for pattern detection
            prices = data['Close'].values
            dates = pd.to_datetime(data.index)

            # Normalize prices to better detect patterns
            normalized_prices = self._normalize_prices(prices)

            # Detect the primary sine wave pattern
            sine_fit = self._fit_sine_wave(normalized_prices)

            if sine_fit is None:
                return None

            # Check for the 3-month cycle pattern
            short_cycle_strength = self._analyze_short_cycle(normalized_prices)

            # Check for the 9-month inversion pattern
            long_cycle_strength = self._analyze_long_cycle(normalized_prices)

            # Calculate overall pattern strength
            pattern_strength = self._calculate_pattern_strength(
                sine_fit, short_cycle_strength, long_cycle_strength
            )

            if pattern_strength < 0.3:  # Minimum threshold
                return None

            # Determine current cycle phase
            current_phase = self._determine_current_phase(normalized_prices, sine_fit)

            # Calculate trend direction
            trend = self._calculate_trend(prices)

            result = {
                'strength': pattern_strength,
                'short_cycle_strength': short_cycle_strength,
                'long_cycle_strength': long_cycle_strength,
                'current_phase': current_phase,
                'trend': trend,
                'sine_params': sine_fit,
                'cycle_info': self._get_cycle_info(normalized_prices, sine_fit),
                'next_reversal': self._predict_next_reversal(current_phase),
                'confidence': min(pattern_strength * 100, 95)  # Cap at 95%
            }

            self.logger.debug(f"Pattern detected with strength: {pattern_strength:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return None

    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize prices using z-score"""
        return (prices - np.mean(prices)) / np.std(prices)

    def _fit_sine_wave(self, normalized_prices: np.ndarray) -> Optional[Dict]:
        """
        Fit a sine wave to the normalized price data

        Args:
            normalized_prices: Normalized price array

        Returns:
            Dictionary with sine wave parameters or None
        """
        try:
            x = np.arange(len(normalized_prices))

            # Define sine function: A * sin(2π/T * x + φ) + C
            def sine_func(x, amplitude, period, phase, offset):
                return amplitude * np.sin(2 * np.pi / period * x + phase) + offset

            # Initial parameter guesses
            initial_amplitude = np.std(normalized_prices)
            initial_period = self.short_cycle_days
            initial_phase = 0
            initial_offset = np.mean(normalized_prices)

            # Fit the sine wave
            params, covariance = curve_fit(
                sine_func, x, normalized_prices,
                p0=[initial_amplitude, initial_period, initial_phase, initial_offset],
                bounds=([0.1, 30, -2 * np.pi, -2], [3, 500, 2 * np.pi, 2]),
                maxfev=5000
            )

            amplitude, period, phase, offset = params

            # Calculate R-squared for goodness of fit
            y_pred = sine_func(x, *params)
            ss_res = np.sum((normalized_prices - y_pred) ** 2)
            ss_tot = np.sum((normalized_prices - np.mean(normalized_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Only accept if R-squared is reasonable
            if r_squared < 0.2:
                return None

            return {
                'amplitude': amplitude,
                'period': period,
                'phase': phase,
                'offset': offset,
                'r_squared': r_squared,
                'fitted_curve': y_pred
            }

        except Exception as e:
            self.logger.debug(f"Sine wave fitting failed: {str(e)}")
            return None

    def _analyze_short_cycle(self, prices: np.ndarray) -> float:
        """Analyze the strength of the 3-month cycle"""
        try:
            # Use FFT to analyze frequency components
            fft = np.fft.fft(prices)
            freqs = np.fft.fftfreq(len(prices))

            # Find the frequency corresponding to ~90 days
            target_freq = 1 / self.short_cycle_days
            freq_tolerance = target_freq * 0.3  # 30% tolerance

            # Find power at frequencies near the target
            freq_mask = np.abs(freqs - target_freq) < freq_tolerance
            if not np.any(freq_mask):
                return 0.0

            # Calculate relative power at target frequency
            target_power = np.mean(np.abs(fft[freq_mask]))
            total_power = np.mean(np.abs(fft[freqs != 0]))  # Exclude DC component

            return min(target_power / total_power, 1.0) if total_power > 0 else 0.0

        except:
            return 0.0

    def _analyze_long_cycle(self, prices: np.ndarray) -> float:
        """Analyze the strength of the 9-month inversion cycle"""
        try:
            if len(prices) < self.long_cycle_days:
                return 0.0

            # Split data into 3-month segments
            segment_size = self.short_cycle_days
            segments = []

            for i in range(0, len(prices) - segment_size, segment_size):
                segment = prices[i:i + segment_size]
                if len(segment) == segment_size:
                    segments.append(segment)

            if len(segments) < 3:  # Need at least 3 segments for inversion analysis
                return 0.0

            # Analyze correlation between segments separated by 9 months
            correlations = []
            for i in range(len(segments) - 3):  # 3 segments = ~9 months
                correlation = np.corrcoef(segments[i], segments[i + 3])[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))  # Inversion shows as negative correlation

            if not correlations:
                return 0.0

            # Strong negative correlation indicates inversion pattern
            return np.mean(correlations)

        except:
            return 0.0

    def _calculate_pattern_strength(self, sine_fit: Dict, short_strength: float, long_strength: float) -> float:
        """Calculate overall pattern strength"""
        if sine_fit is None:
            return 0.0

        # Weight the components
        sine_weight = 0.5
        short_weight = 0.3
        long_weight = 0.2

        sine_strength = sine_fit['r_squared']

        total_strength = (
                sine_weight * sine_strength +
                short_weight * short_strength +
                long_weight * long_strength
        )

        return min(total_strength, 1.0)

    def _determine_current_phase(self, prices: np.ndarray, sine_fit: Dict) -> str:
        """Determine current phase of the cycle"""
        if sine_fit is None:
            return "unknown"

        # Get the current position in the sine wave
        current_x = len(prices) - 1
        current_phase_radians = (2 * np.pi / sine_fit['period'] * current_x + sine_fit['phase']) % (2 * np.pi)

        # Convert to degrees for easier interpretation
        current_phase_degrees = current_phase_radians * 180 / np.pi

        # Determine phase
        if 0 <= current_phase_degrees < 90:
            return "rising"
        elif 90 <= current_phase_degrees < 180:
            return "peak"
        elif 180 <= current_phase_degrees < 270:
            return "falling"
        else:
            return "trough"

    def _calculate_trend(self, prices: np.ndarray) -> str:
        """Calculate overall trend direction"""
        if len(prices) < 20:
            return "neutral"

        # Compare recent average to older average
        recent_avg = np.mean(prices[-20:])
        older_avg = np.mean(prices[-40:-20]) if len(prices) >= 40 else np.mean(prices[:-20])

        change_pct = (recent_avg - older_avg) / older_avg

        if change_pct > 0.05:  # 5% threshold
            return "uptrend"
        elif change_pct < -0.05:
            return "downtrend"
        else:
            return "neutral"

    def _get_cycle_info(self, prices: np.ndarray, sine_fit: Dict) -> Dict:
        """Get detailed cycle information"""
        if sine_fit is None:
            return {}

        return {
            'period_days': sine_fit['period'],
            'amplitude_normalized': sine_fit['amplitude'],
            'current_cycle_position': (len(prices) % sine_fit['period']) / sine_fit['period']
        }

    def _predict_next_reversal(self, current_phase: str) -> str:
        """Predict when the next reversal might occur"""
        phase_transitions = {
            "rising": "peak approaching",
            "peak": "decline expected",
            "falling": "trough approaching",
            "trough": "rise expected",
            "unknown": "uncertain"
        }

        return phase_transitions.get(current_phase, "uncertain")