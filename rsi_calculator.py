import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple


class RSICalculator:
    """Class to calculate RSI and related momentum indicators"""

    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, overbought_threshold: float = 70):
        """
        Initialize RSI calculator

        Args:
            rsi_period: Period for RSI calculation (default 14)
            oversold_threshold: RSI level considered oversold (default 30)
            overbought_threshold: RSI level considered overbought (default 70)
        """
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.logger = logging.getLogger(__name__)

    def calculate_rsi(self, data: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate RSI for the given data

        Args:
            data: DataFrame with price data
            price_column: Column to use for RSI calculation

        Returns:
            DataFrame with RSI added
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()

            # Calculate price changes
            delta = df[price_column].diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate initial average gain and loss
            avg_gain = gains.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
            avg_loss = losses.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()

            # Use Wilder's smoothing method for subsequent calculations
            for i in range(self.rsi_period, len(df)):
                avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (self.rsi_period - 1) + gains.iloc[i]) / self.rsi_period
                avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (self.rsi_period - 1) + losses.iloc[i]) / self.rsi_period

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Handle division by zero
            rsi = rsi.fillna(50)  # Neutral RSI when no losses

            df['RSI'] = rsi

            # Add RSI-based signals
            df['RSI_Oversold'] = (rsi <= self.oversold_threshold)
            df['RSI_Overbought'] = (rsi >= self.overbought_threshold)
            df['RSI_Signal'] = self._generate_rsi_signals(rsi)

            # Add RSI momentum indicators
            df = self._add_rsi_momentum_indicators(df)

            self.logger.debug(f"RSI calculated successfully. Current RSI: {rsi.iloc[-1]:.2f}")

            return df

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return data

    def _generate_rsi_signals(self, rsi: pd.Series) -> pd.Series:
        """
        Generate basic RSI trading signals

        Args:
            rsi: RSI values

        Returns:
            Series with RSI signals ('BUY', 'SELL', 'HOLD')
        """
        signals = pd.Series('HOLD', index=rsi.index)

        # Buy signals: RSI crossing above oversold level
        oversold_cross = (rsi > self.oversold_threshold) & (rsi.shift(1) <= self.oversold_threshold)
        signals[oversold_cross] = 'BUY'

        # Sell signals: RSI crossing below overbought level
        overbought_cross = (rsi < self.overbought_threshold) & (rsi.shift(1) >= self.overbought_threshold)
        signals[overbought_cross] = 'SELL'

        return signals

    def _add_rsi_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional RSI-based momentum indicators

        Args:
            df: DataFrame with RSI

        Returns:
            DataFrame with additional indicators
        """
        rsi = df['RSI']

        # RSI momentum (rate of change of RSI)
        df['RSI_Momentum'] = rsi.diff()

        # RSI divergence detection (simplified)
        df['RSI_Divergence'] = self._detect_rsi_divergence(df)

        # RSI trend (is RSI trending up or down?)
        rsi_sma_short = rsi.rolling(window=5).mean()
        rsi_sma_long = rsi.rolling(window=10).mean()
        df['RSI_Trend'] = np.where(rsi_sma_short > rsi_sma_long, 'UP', 'DOWN')

        # RSI strength indicator
        df['RSI_Strength'] = self._calculate_rsi_strength(rsi)

        return df

    def _detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Detect bullish/bearish divergence between price and RSI

        Args:
            df: DataFrame with price and RSI data
            lookback: Number of periods to look back for divergence

        Returns:
            Series indicating divergence ('BULLISH', 'BEARISH', 'NONE')
        """
        try:
            price = df['Close']
            rsi = df['RSI']
            divergence = pd.Series('NONE', index=df.index)

            for i in range(lookback, len(df)):
                # Get recent data
                recent_price = price.iloc[i - lookback:i + 1]
                recent_rsi = rsi.iloc[i - lookback:i + 1]

                # Find local highs and lows
                price_high_idx = recent_price.idxmax()
                price_low_idx = recent_price.idxmin()
                rsi_high_idx = recent_rsi.idxmax()
                rsi_low_idx = recent_rsi.idxmin()

                # Check for bullish divergence (price makes lower low, RSI makes higher low)
                if (price_low_idx == recent_price.index[-1] and
                        rsi_low_idx != recent_rsi.index[-1] and
                        recent_price.iloc[-1] < recent_price.iloc[0] and
                        recent_rsi.iloc[-1] > recent_rsi.iloc[0]):
                    divergence.iloc[i] = 'BULLISH'

                # Check for bearish divergence (price makes higher high, RSI makes lower high)
                elif (price_high_idx == recent_price.index[-1] and
                      rsi_high_idx != recent_rsi.index[-1] and
                      recent_price.iloc[-1] > recent_price.iloc[0] and
                      recent_rsi.iloc[-1] < recent_rsi.iloc[0]):
                    divergence.iloc[i] = 'BEARISH'

            return divergence

        except Exception as e:
            self.logger.debug(f"Error detecting RSI divergence: {str(e)}")
            return pd.Series('NONE', index=df.index)

    def _calculate_rsi_strength(self, rsi: pd.Series) -> pd.Series:
        """
        Calculate RSI strength indicator

        Args:
            rsi: RSI values

        Returns:
            Series with RSI strength ('VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK')
        """
        strength = pd.Series('MODERATE', index=rsi.index)

        # Very strong conditions
        very_strong_bull = (rsi > 80) | ((rsi > 70) & (rsi.diff() > 5))
        very_strong_bear = (rsi < 20) | ((rsi < 30) & (rsi.diff() < -5))

        strength[very_strong_bull] = 'VERY_STRONG_BULL'
        strength[very_strong_bear] = 'VERY_STRONG_BEAR'

        # Strong conditions
        strong_bull = (rsi > 70) & (rsi <= 80) & (rsi.diff() > 0)
        strong_bear = (rsi < 30) & (rsi >= 20) & (rsi.diff() < 0)

        strength[strong_bull] = 'STRONG_BULL'
        strength[strong_bear] = 'STRONG_BEAR'

        # Weak conditions
        weak = (rsi > 45) & (rsi < 55)
        strength[weak] = 'WEAK'

        return strength

    def get_rsi_analysis(self, data: pd.DataFrame) -> dict:
        """
        Get comprehensive RSI analysis for the current data

        Args:
            data: DataFrame with RSI data

        Returns:
            Dictionary with RSI analysis
        """
        if 'RSI' not in data.columns:
            return {'error': 'RSI not calculated'}

        current_rsi = data['RSI'].iloc[-1]
        rsi_series = data['RSI'].dropna()

        # Calculate percentile ranking
        rsi_percentile = (rsi_series < current_rsi).mean() * 100

        # Recent RSI trend
        recent_rsi = rsi_series.tail(10)
        rsi_trend = 'RISING' if recent_rsi.iloc[-1] > recent_rsi.iloc[0] else 'FALLING'
        rsi_momentum = recent_rsi.diff().mean()

        # Time since last extreme
        last_oversold = data[data['RSI'] <= self.oversold_threshold].index
        last_overbought = data[data['RSI'] >= self.overbought_threshold].index

        days_since_oversold = len(data) - data.index.get_loc(last_oversold[-1]) - 1 if len(last_oversold) > 0 else None
        days_since_overbought = len(data) - data.index.get_loc(last_overbought[-1]) - 1 if len(
            last_overbought) > 0 else None

        analysis = {
            'current_rsi': current_rsi,
            'rsi_level': self._classify_rsi_level(current_rsi),
            'rsi_percentile': rsi_percentile,
            'trend': rsi_trend,
            'momentum': rsi_momentum,
            'days_since_oversold': days_since_oversold,
            'days_since_overbought': days_since_overbought,
            'divergence': data['RSI_Divergence'].iloc[-1] if 'RSI_Divergence' in data.columns else 'NONE',
            'strength': data['RSI_Strength'].iloc[-1] if 'RSI_Strength' in data.columns else 'MODERATE',
            'signal': data['RSI_Signal'].iloc[-1] if 'RSI_Signal' in data.columns else 'HOLD'
        }

        return analysis

    def _classify_rsi_level(self, rsi_value: float) -> str:
        """Classify RSI level into categories"""
        if rsi_value >= 80:
            return 'EXTREMELY_OVERBOUGHT'
        elif rsi_value >= self.overbought_threshold:
            return 'OVERBOUGHT'
        elif rsi_value >= 60:
            return 'BULLISH'
        elif rsi_value >= 40:
            return 'NEUTRAL'
        elif rsi_value >= self.oversold_threshold:
            return 'BEARISH'
        elif rsi_value >= 20:
            return 'OVERSOLD'
        else:
            return 'EXTREMELY_OVERSOLD'

    def calculate_stochastic_rsi(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic RSI for additional momentum analysis

        Args:
            data: DataFrame with RSI calculated
            k_period: Period for %K calculation
            d_period: Period for %D smoothing

        Returns:
            DataFrame with Stochastic RSI added
        """
        try:
            if 'RSI' not in data.columns:
                self.logger.error("RSI must be calculated before Stochastic RSI")
                return data

            df = data.copy()
            rsi = df['RSI']

            # Calculate Stochastic RSI
            rsi_low = rsi.rolling(window=k_period).min()
            rsi_high = rsi.rolling(window=k_period).max()

            stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
            stoch_rsi = stoch_rsi.fillna(50)  # Handle division by zero

            # Calculate %D (smoothed version)
            stoch_rsi_d = stoch_rsi.rolling(window=d_period).mean()

            df['StochRSI_K'] = stoch_rsi
            df['StochRSI_D'] = stoch_rsi_d

            # Generate Stochastic RSI signals
            df['StochRSI_Signal'] = self._generate_stoch_rsi_signals(stoch_rsi, stoch_rsi_d)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating Stochastic RSI: {str(e)}")
            return data

    def _generate_stoch_rsi_signals(self, stoch_k: pd.Series, stoch_d: pd.Series) -> pd.Series:
        """Generate trading signals from Stochastic RSI"""
        signals = pd.Series('HOLD', index=stoch_k.index)

        # Buy signal: %K crosses above %D while both are oversold
        buy_condition = (
                (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) &
                (stoch_k < 20) & (stoch_d < 20)
        )
        signals[buy_condition] = 'BUY'

        # Sell signal: %K crosses below %D while both are overbought
        sell_condition = (
                (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) &
                (stoch_k > 80) & (stoch_d > 80)
        )
        signals[sell_condition] = 'SELL'

        return signals
