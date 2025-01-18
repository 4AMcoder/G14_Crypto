import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple
from utils.logger import get_logger

class MeanReversionStrategy:
    def __init__(
        self,
        lookback: int = 20,
        std_dev_threshold: float = 2.0,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        volume_threshold: float = 1.5,
        stop_loss_factor: float = 2.0,
        take_profit_factor: float = 3.5,
        zscore_threshold: float = 2.0
    ):
        self.lookback = lookback
        self.std_dev_threshold = std_dev_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_threshold = volume_threshold
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.zscore_threshold = zscore_threshold
        self.logger = get_logger("trading_bot.strategy")

    def calculate_zscore(self, data: pd.Series) -> pd.Series:
        """Calculate rolling z-score"""
        return pd.Series(
            (data - data.rolling(window=self.lookback).mean()) / 
            data.rolling(window=self.lookback).std(),
            index=data.index
        )

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator manually"""
        # Calculate %K
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=d_period).mean()
        
        return k, d

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion trading signals"""
        if len(data) < self.lookback:
            raise ValueError(f"Data length ({len(data)}) is shorter than lookback period ({self.lookback})")

        signals = data.copy()
        
        # Calculate indicators using pandas_ta this time. cuts down on manually calculating myself
        signals['atr'] = data.ta.atr(length=self.lookback)
        signals['rsi'] = data.ta.rsi(length=self.rsi_period)
        signals['zscore'] = self.calculate_zscore(signals['close'])
        
        # Calculate Bollinger Bands
        bbands = data.ta.bbands(length=self.lookback, std=self.std_dev_threshold)
        signals['bb_upper'] = bbands[f'BBU_{self.lookback}_{self.std_dev_threshold}']
        signals['bb_middle'] = bbands[f'BBM_{self.lookback}_{self.std_dev_threshold}']
        signals['bb_lower'] = bbands[f'BBL_{self.lookback}_{self.std_dev_threshold}']
        
        # Calculate Stochastic oscillator
        signals['stoch_k'], signals['stoch_d'] = self.calculate_stochastic(data, k_period=self.lookback)
        
        # Volume
        signals['volume_ma'] = data.ta.sma(close=data['volume'], length=self.lookback)
        signals['volume_ratio'] = signals['volume'] / signals['volume_ma']
        
        # Mean reversion signals
        signals['buy_signal'] = (
            (signals['zscore'] < -self.zscore_threshold) &  #
            (signals['close'] < signals['bb_lower']) &     
            (signals['rsi'] < self.rsi_oversold) &        
            (signals['stoch_k'] < 20) &                   
            (signals['volume_ratio'] > self.volume_threshold) 
        )

        signals['sell_signal'] = (
            (signals['zscore'] > self.zscore_threshold) &   
            (signals['close'] > signals['bb_upper']) &     
            (signals['rsi'] > self.rsi_overbought) &      
            (signals['stoch_k'] > 80) &                
            (signals['volume_ratio'] > self.volume_threshold) 
        )

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
            title="Mean Reversion Analysis Dashboard",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
        fig.update_xaxes(title="Time", row=2, col=1)

        fig.add_hline(y=self.rsi_overbought, line_dash="dash", line_color="red", 
                     row=2, col=1, secondary_y=True)
        fig.add_hline(y=self.rsi_oversold, line_dash="dash", line_color="green", 
                     row=2, col=1, secondary_y=True)

        fig.show()