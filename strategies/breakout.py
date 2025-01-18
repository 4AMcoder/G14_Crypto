import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple
from utils.logger import get_logger

class BreakoutStrategy:
    def __init__(
        self,
        lookback: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        volume_multiplier: float = 0.1,
        stop_loss_factor: float = 2.0,
        take_profit_factor: float = 3.0,
        rc_threshold: float = 0.5
    ):
        self.lookback = lookback
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.volume_multiplier = volume_multiplier
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.rc_threshold = rc_threshold
        self.logger = get_logger("trading_bot.strategy")

    def calculate_robert_carver(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Robert Carver signal using EMA"""
        close_prices = data["close"]
        rolling_mean = close_prices.ewm(span=self.lookback, min_periods=self.lookback).mean()
        rolling_std = close_prices.rolling(window=self.lookback, min_periods=self.lookback).std()
        
        rolling_std = rolling_std.replace(0, np.finfo(float).eps)
        return (close_prices - rolling_mean) / rolling_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout trading signals"""
        if len(data) < self.lookback:
            raise ValueError(f"Data length ({len(data)}) is shorter than lookback period ({self.lookback})")

        signals = data.copy()
        
        # Calculate indicators using pandas_ta
        signals['atr'] = data.ta.atr(length=self.lookback)
        signals['rsi'] = data.ta.rsi(length=self.rsi_period)
        signals['rc_signal'] = self.calculate_robert_carver(signals)
        signals['vwap'] = data.ta.vwap()
        
        # Calculate Bollinger Bands
        bbands = data.ta.bbands(length=self.lookback, std=self.bb_std)
        signals['bb_upper'] = bbands[f'BBU_{self.lookback}_{self.bb_std}']
        signals['bb_middle'] = bbands[f'BBM_{self.lookback}_{self.bb_std}']
        signals['bb_lower'] = bbands[f'BBL_{self.lookback}_{self.bb_std}']
        
        # Volume analysis
        signals['volume_ma'] = data.ta.sma(close=data['volume'], length=self.lookback)
        signals['volume_std'] = data.ta.stdev(close=data['volume'], length=self.lookback)
        signals['volume_ratio'] = signals['volume'] / signals['volume_ma']
        signals['strong_volume'] = signals['volume'] > signals['volume_ma'] * (1 + self.volume_multiplier)
        
        # Generate signals
        signals['bb_breach_up'] = (
            (signals['high'] > signals['bb_upper']) &
            (signals['close'] > signals['bb_middle'])
        )
        
        signals['bb_breach_down'] = (
            (signals['low'] < signals['bb_lower']) &
            (signals['close'] < signals['bb_middle'])
        )

        signals['buy_signal'] = (
            signals['bb_breach_up'] &
            ((signals['close'] > signals['vwap']) | (signals['close'] > signals['bb_middle'])) &
            ((signals['rsi'] > 40) & (signals['rsi'] < 80)) &
            ((signals['rc_signal'] > -self.rc_threshold) | signals['strong_volume'])
        )

        signals['sell_signal'] = (
            signals['bb_breach_down'] &
            ((signals['close'] < signals['vwap']) | (signals['close'] < signals['bb_middle'])) &
            ((signals['rsi'] < 65) & (signals['rsi'] > 20)) &
            ((signals['rc_signal'] < self.rc_threshold) | signals['strong_volume'])
        )

        # Calculate stop-loss and take-profit levels
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
            subplot_titles=("Price with Bollinger Bands", "Volume and RSI"),
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
            ("bb_upper", "rgba(0, 100, 255, 0.3)"),
            ("bb_middle", "rgba(0, 50, 255, 0.3)"),
            ("bb_lower", "rgba(0, 100, 255, 0.3)")
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
            opacity=1.0
        ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals["rsi"],
            mode="lines",
            line=dict(color="purple", width=1),
            name="RSI"
        ), row=2, col=1, secondary_y=True)

        fig.update_layout(
            title="Breakout Analysis Dashboard",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
        fig.update_xaxes(title="Time", row=2, col=1)

        # Add RSI levels
        fig.add_hline(y=75, line_dash="dash", line_color="red", row=2, col=1, secondary_y=True)
        fig.add_hline(y=35, line_dash="dash", line_color="green", row=2, col=1, secondary_y=True)

        fig.show()