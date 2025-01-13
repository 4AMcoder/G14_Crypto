import pandas as pd
import numpy as np
from typing import Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.logger import get_logger

class MeanReversionStrategy:
    def __init__(
        self,
        base_lookback: int = 20,
        std_dev_threshold: float = 2.0,
        rsi_lookback: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        volume_lookback: int = 20,
        volume_threshold: float = 1.5,
        stop_loss_factor: float = 2.0,
        take_profit_factor: float = 3.5,
        zscore_threshold: float = 2.0,
        mean_period: int = 20,
        timeframe_adjustments: dict = None
    ):
        self.base_lookback = base_lookback
        self.std_dev_threshold = std_dev_threshold
        self.rsi_lookback = rsi_lookback
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_lookback = volume_lookback
        self.volume_threshold = volume_threshold
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.zscore_threshold = zscore_threshold
        self.mean_period = mean_period
        self.logger = get_logger("trading_bot.strategy")
        
        # Default timeframe adjustments if none provided
        self.timeframe_adjustments = timeframe_adjustments or {
            '1min': 0.5,
            '5min': 0.7,
            '15min': 0.8,
            '30min': 0.9,
            '60min': 1.0,
            '240min': 1.2,
            '720min': 1.5,
            '1440min': 1.8,
        }
        
        # Data size adjustments
        self.data_size_adjustments = {
            1000: 1.0,
            2500: 1.2,
            5000: 1.4,
            10000: 1.6,
            20000: 1.8
        }

    def calculate_zscore(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return (data - mean) / std

    def calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate RSI"""
        if len(data) < window:
            raise ValueError(f"Input series length ({len(data)}) must be >= window size ({window})")

        delta = data.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        avg_gain = gains.ewm(com=window-1, min_periods=window).mean()
        avg_loss = losses.ewm(com=window-1, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, data: pd.Series, window: int, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({
            'tr1': tr1, 
            'tr2': tr2, 
            'tr3': tr3
        }).max(axis=1)
        
        return tr.ewm(span=self.base_lookback, min_periods=self.base_lookback).mean()

    def calculate_stochastic(self, data: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(window).min()
        high_max = data['high'].rolling(window).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        k = k.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        
        return k, d

    def calculate_mean_reversion_score(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate composite mean reversion score"""
        # Z-score of price
        zscore = self.calculate_zscore(data['close'], window)
        
        # Distance from moving average
        ma = data['close'].rolling(window).mean()
        ma_distance = (data['close'] - ma) / ma
        
        # RSI normalized to -1 to 1 scale
        rsi = self.calculate_rsi(data['close'], window)
        rsi_norm = (rsi - 50) / 50
        
        # Combine signals into score (-1 to 1 scale)
        score = -(zscore / self.zscore_threshold * 0.4 + 
                 ma_distance * 0.4 + 
                 rsi_norm * 0.2)
        
        return score.clip(-1, 1)

    def _get_adjusted_lookback(self, timeframe: str, data_length: int) -> int:
        """Calculate adjusted lookback period based on timeframe and data size"""
        base_multiplier = self.timeframe_adjustments.get(timeframe, 1.0)
        
        size_multiplier = 1.0
        for size, mult in sorted(self.data_size_adjustments.items()):
            if data_length > size:
                size_multiplier = mult
        
        adjusted_lookback = int(self.base_lookback * base_multiplier * size_multiplier)
        
        max_lookback = int(data_length * 0.2)
        min_lookback = 5
        
        return min(max_lookback, max(min_lookback, adjusted_lookback))

    def generate_signals(self, data: pd.DataFrame, timeframe: str = '60min') -> pd.DataFrame:
        """Generate mean reversion trading signals"""
        if len(data) < self.base_lookback:
            raise ValueError(f"Data length ({len(data)}) is shorter than base lookback period ({self.base_lookback})")

        signals = data.copy()
        lookback = self._get_adjusted_lookback(timeframe, len(data))
        
        # Calculate indicators
        signals['atr'] = self.calculate_atr(signals)
        signals['rsi'] = self.calculate_rsi(signals['close'], self.rsi_lookback)
        signals['zscore'] = self.calculate_zscore(signals['close'], lookback)
        signals['mr_score'] = self.calculate_mean_reversion_score(signals, lookback)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(signals['close'], lookback)
        signals['bb_upper'] = bb_upper
        signals['bb_middle'] = bb_middle
        signals['bb_lower'] = bb_lower
        
        # Calculate Stochastic
        signals['stoch_k'], signals['stoch_d'] = self.calculate_stochastic(signals)
        
        # Volume analysis
        signals['volume_ma'] = signals['volume'].rolling(window=self.volume_lookback).mean()
        signals['volume_ratio'] = signals['volume'] / signals['volume_ma']
        
        # Mean reversion signals
        signals['buy_signal'] = (
            (signals['mr_score'] > 0.5) &                     # Strong mean reversion signal
            (signals['close'] < signals['bb_lower']) &        # Price below lower band
            (signals['rsi'] < self.rsi_oversold) &           # Oversold RSI
            (signals['stoch_k'] < 20) &                      # Oversold Stochastic
            (signals['volume_ratio'] > self.volume_threshold) # High volume
        )

        signals['sell_signal'] = (
            (signals['mr_score'] < -0.5) &                   # Strong mean reversion signal
            (signals['close'] > signals['bb_upper']) &       # Price above upper band
            (signals['rsi'] > self.rsi_overbought) &        # Overbought RSI
            (signals['stoch_k'] > 80) &                     # Overbought Stochastic
            (signals['volume_ratio'] > self.volume_threshold) # High volume
        )

        # Calculate stop-loss and take-profit using ATR
        signals['stop_loss'] = np.where(
            signals['buy_signal'],
            signals['close'] - signals['atr'] * self.stop_loss_factor,
            np.where(
                signals['sell_signal'],
                signals['close'] + signals['atr'] * self.stop_loss_factor,
                np.nan
            )
        )

        signals['take_profit'] = np.where(
            signals['buy_signal'],
            signals['close'] + signals['atr'] * self.take_profit_factor,
            np.where(
                signals['sell_signal'],
                signals['close'] - signals['atr'] * self.take_profit_factor,
                np.nan
            )
        )

        return signals

    def plot_signals(self, signals: pd.DataFrame) -> None:
        """Plot trading signals with indicators"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price with Bollinger Bands", "RSI and Volume"),
            row_width=[0.3, 0.7],
            specs=[[{"secondary_y": False}],
                  [{"secondary_y": True}]]
        )

        fig.add_trace(go.Candlestick(
            x=signals.index,
            open=signals["open"],
            high=signals["high"],
            low=signals["low"],
            close=signals["close"],
            name="Price"
        ), row=1, col=1)


        for band, color in [
            ("bb_upper", "rgba(173, 204, 255, 0.7)"), # TODO: change these colours aswell very faint
            ("bb_middle", "rgba(98, 128, 255, 0.7)"),
            ("bb_lower", "rgba(173, 204, 255, 0.7)")
        ]:
            fig.add_trace(go.Scatter(
                x=signals.index,
                y=signals[band],
                mode="lines",
                line=dict(color=color, width=1),
                name=f"{band.replace('_', ' ').title()}"
            ), row=1, col=1)

        buy_points = signals[signals["buy_signal"]]
        sell_points = signals[signals["sell_signal"]]

        fig.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["low"] * 0.999,
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color="green"),
            name="Buy Signal"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["high"] * 1.001,
            mode="markers",
            marker=dict(symbol="triangle-down", size=15, color="red"),
            name="Sell Signal"
        ), row=1, col=1)

        colors = ['green' if close >= open else 'red' 
                for open, close in zip(signals['open'], signals['close'])]
        
        fig.add_trace(go.Bar(
            x=signals.index,
            y=signals["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7
        ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["rsi"],
            mode="lines",
            line=dict(color="purple", width=1),
            name="RSI"
        ), row=2, col=1, secondary_y=True)

        fig.update_layout(
            title="Mean Reversion Analysis Dashboard",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
        fig.update_xaxes(title="Time", row=2, col=1)

        fig.add_hline(y=self.rsi_overbought, line_dash="dash", line_color="green", row=2, col=1, secondary_y=True)
        fig.add_hline(y=self.rsi_oversold, line_dash="dash", line_color="red", row=2, col=1, secondary_y=True)

        fig.show()