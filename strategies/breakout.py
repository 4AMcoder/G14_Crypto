import pandas as pd
import numpy as np
from typing import Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BreakoutStrategy:
    def __init__(
        self,
        base_lookback: int = 20,
        buffer: float = 0.001,
        stop_loss_factor: float = 1.0,
        take_profit_factor: float = 2.0,
        rsi_lookback: int = 20,  
        volume_lookback: int = 20, 
        rc_threshold: float = 0.5, 
        volume_multiplier: float = 0.1,
        pivot_window: int = 15,
        breakout_threshold: float = 0.0015,
        bb_std: float = 2.0,
        timeframe_adjustments: dict = None
    ):
        self.base_lookback = base_lookback
        self.buffer = buffer
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.rsi_lookback = rsi_lookback
        self.volume_lookback = volume_lookback
        self.rc_threshold = rc_threshold
        self.volume_multiplier = volume_multiplier
        self.pivot_window = pivot_window
        self.breakout_threshold = breakout_threshold
        self.bb_std = bb_std
        
        # Default timeframe adjustments if none provided
        self.timeframe_adjustments = timeframe_adjustments or {
            '1min': 0.5,    # Shorter lookback for very short timeframes
            '5min': 0.7,
            '15min': 0.8,
            '30min': 0.9,
            '60min': 1.0,   # Base lookback (1H)
            '240min': 1.2,  # 4H
            '720min': 1.5,  # 12H
            '1440min': 1.8, # 24H (daily)
        }
        
        # Data size adjustments
        self.data_size_adjustments = {
            1000: 1.0,    # Base multiplier
            2500: 1.2,    # Medium datasets
            5000: 1.4,    # Larger datasets
            10000: 1.6,   # Very large datasets
            20000: 1.8    # Massive datasets
        }

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
        
        atr = tr.ewm(span=self.base_lookback, min_periods=self.base_lookback).mean()
        return atr

    def calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate RSI using exponential moving average method"""
        if len(data) < window:
            raise ValueError(f"Input series length ({len(data)}) must be >= window size ({window})")

        delta = data.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        avg_gain = gains.ewm(com=window-1, min_periods=window).mean()
        avg_loss = losses.ewm(com=window-1, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def calculate_robert_carver(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Robert Carver signal using EMA"""
        close_prices = data["close"]
        rolling_mean = close_prices.ewm(span=self.base_lookback, min_periods=self.base_lookback).mean()
        rolling_std = close_prices.rolling(window=self.base_lookback, min_periods=self.base_lookback).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.finfo(float).eps)
        
        return (close_prices - rolling_mean) / rolling_std

    def calculate_bollinger_bands(self, data: pd.Series, lookback: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = data.ewm(span=lookback, min_periods=lookback).mean()
        rolling_std = data.rolling(window=lookback, min_periods=lookback).std()
        
        upper_band = rolling_mean + (rolling_std * self.bb_std)
        lower_band = rolling_mean - (rolling_std * self.bb_std)
        
        return upper_band, rolling_mean, lower_band

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
        return vwap

    def calculate_position_size(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Calculate dynamic position size based on volatility"""
        atr = self.calculate_atr(data)
        avg_atr = atr.rolling(window=lookback).mean()
        
        # Normalize ATR to get relative volatility
        rel_vol = atr / avg_atr
        
        # Scale position size inversely with volatility (higher vol = smaller position)
        pos_size = 1 / rel_vol
        
        # Normalize between 0.1 and 1.0
        pos_size = pos_size.clip(0.1, 1.0)
        
        return pos_size

    def _get_adjusted_lookback(self, timeframe: str, data_length: int) -> int:
        """Calculate adjusted lookback period based on timeframe and data size"""
        # Get base multiplier for timeframe
        base_multiplier = self.timeframe_adjustments.get(timeframe, 1.0)
        
        # Get data size multiplier
        size_multiplier = 1.0
        for size, mult in sorted(self.data_size_adjustments.items()):
            if data_length > size:
                size_multiplier = mult
        
        # Calculate final lookback
        adjusted_lookback = int(self.base_lookback * base_multiplier * size_multiplier)
        
        # Ensure lookback is reasonable
        max_lookback = int(data_length * 0.2)  # Don't use more than 20% of data
        min_lookback = 5
        
        return min(max_lookback, max(min_lookback, adjusted_lookback))

    def generate_signals(self, data: pd.DataFrame, timeframe: str = '60min') -> pd.DataFrame:
        """Generate trading signals"""
        if len(data) < self.base_lookback:
            raise ValueError(f"Data length ({len(data)}) is shorter than base lookback period ({self.base_lookback})")

        signals = data.copy()
        lookback = self._get_adjusted_lookback(timeframe, len(data))
        
        # Calculate indicators
        signals["atr"] = self.calculate_atr(signals)
        signals["rsi"] = self.calculate_rsi(signals["close"], lookback)
        signals["rc_signal"] = self.calculate_robert_carver(signals)
        signals["vwap"] = self.calculate_vwap(signals)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(signals['close'], lookback)
        signals['bb_upper'] = bb_upper
        signals['bb_middle'] = bb_middle
        signals['bb_lower'] = bb_lower
        
        # Volume analysis
        signals["volume_ma"] = signals["volume"].ewm(span=lookback, min_periods=lookback).mean()
        signals["volume_std"] = signals["volume"].rolling(lookback, min_periods=lookback).std()
        signals["volume_trend"] = signals["volume"] / signals["volume_ma"]
        signals["strong_volume"] = signals["volume"] > signals["volume_ma"] * (1 + self.volume_multiplier)
        
        # Calculate position size
        signals["position_size"] = self.calculate_position_size(signals, lookback)
        
        # Generate signals
        signals["buy_signal"] = (
            (signals["close"] > signals["bb_upper"]) &
            (signals["close"] > signals["vwap"]) &
            (signals["rsi"] > 30) & (signals["rsi"] < 75) &
            (signals["rc_signal"] > -self.rc_threshold) &
            signals["strong_volume"]
        )

        signals["sell_signal"] = (
            (signals["close"] < signals["bb_lower"]) &
            (signals["close"] < signals["vwap"]) &
            (signals["rsi"] < 70) & (signals["rsi"] > 25) &
            (signals["rc_signal"] < self.rc_threshold) &
            signals["strong_volume"]
        )

        # Calculate stop-loss and take-profit levels
        signals["stop_loss"] = np.where(
            signals["buy_signal"],
            signals["close"] - signals["atr"] * self.stop_loss_factor,
            np.where(
                signals["sell_signal"],
                signals["close"] + signals["atr"] * self.stop_loss_factor,
                np.nan
            )
        )

        signals["take_profit"] = np.where(
            signals["buy_signal"],
            signals["close"] + signals["atr"] * self.take_profit_factor,
            np.where(
                signals["sell_signal"],
                signals["close"] - signals["atr"] * self.take_profit_factor,
                np.nan
            )
        )

        return signals

    def plot_signals(self, signals: pd.DataFrame) -> None:
        """Plot trading signals with Bollinger Bands and combined volume/RSI subplot"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price with Bollinger Bands", "Volume and RSI"),
            row_width=[0.3, 0.7],
            specs=[[{"secondary_y": False}],
                  [{"secondary_y": True}]]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=signals.index,
            open=signals["open"],
            high=signals["high"],
            low=signals["low"],
            close=signals["close"],
            name="Price"
        ), row=1, col=1)

        # Bollinger Bands
        for band, color in [
            ("bb_upper", "rgba(173, 204, 255, 0.7)"),
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

        # Buy/Sell signals
        buy_points = signals[signals["buy_signal"]]
        sell_points = signals[signals["sell_signal"]]

        fig.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["low"] * 0.999,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="green"),
            name="Buy Signal"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["high"] * 1.001,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Sell Signal"
        ), row=1, col=1)

        # Volume
        colors = ['green' if close >= open else 'red' 
                for open, close in zip(signals['open'], signals['close'])]
        
        fig.add_trace(go.Bar(
            x=signals.index,
            y=signals["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.7
        ), row=2, col=1, secondary_y=False)

        # RSI
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["rsi"],
            mode="lines",
            line=dict(color="purple", width=1),
            name="RSI"
        ), row=2, col=1, secondary_y=True)

        # Update layout
        fig.update_layout(
            title="Trading Analysis Dashboard",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        # Update axes
        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
        fig.update_xaxes(title="Time", row=2, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1, secondary_y=True)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1, secondary_y=True)

        fig.show()