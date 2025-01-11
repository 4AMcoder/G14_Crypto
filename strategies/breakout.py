import pandas as pd
import numpy as np
from typing import Optional
import mplfinance as mpf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BreakoutStrategy:
    def __init__(
        self,
        lookback: int = 10,
        buffer: float = 0.001,
        stop_loss_factor: float = 1.0,
        take_profit_factor: float = 2.0,
        rsi_lookback: int = 7,  
        volume_lookback: int = 5, 
        rc_threshold: float = 0.5, 
        volume_multiplier: float = 0.1,
        pivot_window: int = 3, 
    ):
        self.lookback = lookback
        self.buffer = buffer
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.rsi_lookback = rsi_lookback
        self.volume_lookback = volume_lookback
        self.rc_threshold = rc_threshold
        self.volume_multiplier = volume_multiplier
        self.pivot_window = pivot_window

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=self.lookback, min_periods=1).mean()  # Added min_periods

    def calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate RSI using simple method to avoid NaN values"""
        delta = data.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        # avoid div 0 when prices totally flat
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)

        rsi = 100 - (100 / (1 + rs))
        return rsi


    def calculate_robert_carver(self, data: pd.DataFrame) -> pd.Series:
        """Enhanced Robert Carver signal using EMA"""
        close_prices = data["close"]
        rolling_mean = close_prices.ewm(span=self.lookback, min_periods=1).mean()
        rolling_std = close_prices.rolling(window=self.lookback, min_periods=1).std()

        # avoid div 0 when prices totally flat
        rolling_std = rolling_std.replace(0, np.finfo(float).eps)

        rc = (close_prices - rolling_mean) / rolling_std 
        return rc 

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with rolling window"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        rolling_typical_volume = (typical_price * data["volume"]).rolling(window=self.lookback, min_periods=self.lookback)
        rolling_volume = data["volume"].rolling(window=self.lookback, min_periods=self.lookback)

        vwap = rolling_typical_volume.sum() / rolling_volume.sum()

        return vwap

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        return (np.sign(data['close'].diff().fillna(0)) * data['volume']).cumsum()

    def detect_pivots(self, data: pd.DataFrame) -> pd.DataFrame:
        def is_pivot(candle_idx: int, window: int) -> int:
            if candle_idx - window < 0 or candle_idx + window >= len(data):
                return 0

            pivot_high, pivot_low = True, True
            for i in range(candle_idx - window, candle_idx + window + 1):
                if data.iloc[candle_idx].low > data.iloc[i].low:
                    pivot_low = False
                if data.iloc[candle_idx].high < data.iloc[i].high:
                    pivot_high = False

            if pivot_high and pivot_low:
                return 3
            elif pivot_high:
                return 1
            elif pivot_low:
                return 2
            return 0

        data["pivot"] = [is_pivot(i, self.pivot_window) for i in range(len(data))]
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.lookback:
            raise ValueError(f"Data length ({len(data)}) is shorter than lookback period ({self.lookback})")

        signals = data.copy()

        # Technical Indicators with NaN handling
        signals["atr"] = self.calculate_atr(signals)
        signals["upper_channel"] = signals["high"].rolling(window=self.lookback, min_periods=1).max()
        signals["lower_channel"] = signals["low"].rolling(window=self.lookback, min_periods=1).min()
        signals["rsi"] = self.calculate_rsi(signals["close"], self.rsi_lookback)
        signals["rc_signal"] = self.calculate_robert_carver(signals)
        signals["vwap"] = self.calculate_vwap(signals)
        signals["obv"] = self.calculate_obv(signals)

        # Volume Analysis
        signals["volume_ma"] = signals["volume"].ewm(span=self.volume_lookback, min_periods=1).mean()
        signals["volume_std"] = signals["volume"].rolling(self.volume_lookback, min_periods=1).std()
        signals["volume_trend"] = signals["volume"] / signals["volume_ma"]
        signals["volume_confirm"] = (signals["volume"] > signals["volume_ma"])  # Simplified volume confirmation

        # Trend Analysis
        signals["short_ma"] = signals["close"].ewm(span=self.lookback//2, min_periods=1).mean()
        signals["long_ma"] = signals["close"].ewm(span=self.lookback, min_periods=1).mean()
        signals["trend_alignment"] = (signals["short_ma"] > signals["long_ma"]).astype(int)

        # Pivot Detection
        signals = self.detect_pivots(signals)

        # Breach Detection with less strict conditions
        signals["donchian_breach_upper"] = (
            (signals["close"] > signals["upper_channel"]) &
            (signals["volume"] > signals["volume_ma"])
        )
        signals["donchian_breach_lower"] = (
            (signals["close"] < signals["lower_channel"]) &
            (signals["volume"] > signals["volume_ma"])
        )

        # Buy and Sell Signals with relaxed conditions
        signals["buy_signal"] = (
            signals["donchian_breach_upper"] &
            (signals["close"] > signals["vwap"]) &
            (signals["rsi"] > 30) &  # More permissive RSI
            (signals["rc_signal"] > -self.rc_threshold) &  # More permissive RC signal
            (signals["volume"] > signals["volume_ma"])  # Simplified volume condition
        )

        signals["sell_signal"] = (
            signals["donchian_breach_lower"] &
            (signals["close"] < signals["vwap"]) &
            (signals["rsi"] < 70) &  # More permissive RSI
            (signals["rc_signal"] < self.rc_threshold) &  # More permissive RC signal
            (signals["volume"] > signals["volume_ma"])  # Simplified volume condition
        )

        # Stop-loss and Take-profit using ATR
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
        """
        Plot candlestick chart with RSI, volume, Donchian channels, and buy/sell signals.
        Chart 1: Candlesticks with Donchian channels and buy/sell signal vertical lines
        Chart 2: Volume bars (colored by price direction) with RSI line (dual y-axes)
        
        Parameters:
        - signals (pd.DataFrame): DataFrame containing trading data and signals
        """
        # Ensure required fields
        required_columns = {"open", "high", "low", "close", "volume", "buy_signal", 
                        "sell_signal", "upper_channel", "lower_channel", "rsi"}
        if not required_columns.issubset(signals.columns):
            raise ValueError(f"Missing columns in signals DataFrame. Required: {required_columns}")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Candlestick Chart with Signals", "Volume and RSI"),
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
            name="Candlestick"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["upper_channel"],
            mode="lines",
            line=dict(color="green", width=1),
            name="Upper Channel"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["lower_channel"],
            mode="lines",
            line=dict(color="red", width=1),
            name="Lower Channel"
        ), row=1, col=1)

        for i, row in signals.iterrows():
            if row["buy_signal"]:
                fig.add_shape(
                    type="line",
                    x0=row.name, x1=row.name,
                    y0=signals["low"].min(), y1=signals["high"].max(),
                    line=dict(color="green", dash="dot"),
                    row=1, col=1
                )
            if row["sell_signal"]:
                fig.add_shape(
                    type="line",
                    x0=row.name, x1=row.name,
                    y0=signals["low"].min(), y1=signals["high"].max(),
                    line=dict(color="red", dash="dot"),
                    row=1, col=1
                )

        colors = ['green' if close >= open else 'red' 
                for open, close in zip(signals['open'], signals['close'])]

        fig.add_trace(go.Bar(
            x=signals.index,
            y=signals["volume"],
            name="Volume",
            marker_color=colors,
        ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["rsi"],
            mode="lines",
            line=dict(color="blue", width=1),
            name="RSI"
        ), row=2, col=1, secondary_y=True)

        fig.update_layout(
            title="Trading Analysis Dashboard",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
        fig.update_xaxes(title="Time", row=2, col=1)

        fig.show()