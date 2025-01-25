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
        lookback: int = 24,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        volume_multiplier: float = 1.5,
        stop_loss_factor: float = 2.0,
        take_profit_factor: float = 3.0,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30
    ):
        self.lookback = lookback
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.volume_multiplier = volume_multiplier
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        
        # Core indicators
        signals['atr'] = data.ta.atr(length=self.lookback)
        signals['rsi'] = data.ta.rsi(length=self.rsi_period)
        
        # Enhanced trend detection
        signals['sma_20'] = data.ta.sma(length=20)
        signals['sma_50'] = data.ta.sma(length=50)
        signals['trend'] = (signals['sma_20'] > signals['sma_50']).astype(int) * 2 - 1
        
        # Standard Bollinger Bands
        bbands = data.ta.bbands(length=self.lookback, std=self.bb_std)
        signals['bb_upper'] = bbands[f'BBU_{self.lookback}_{self.bb_std}']
        signals['bb_lower'] = bbands[f'BBL_{self.lookback}_{self.bb_std}']
        
        # Volume analysis
        signals['volume_ma'] = data.ta.sma(close=data['volume'], length=self.lookback)
        signals['volume_std'] = data.ta.stdev(close=data['volume'], length=self.lookback)
        signals['volume_zscore'] = (signals['volume'] - signals['volume_ma']) / signals['volume_std']
        
        # Price momentum
        signals['momentum'] = data.ta.mom(length=self.lookback)
        signals['momentum_ma'] = signals['momentum'].rolling(window=5).mean()
        
        # Breakout strength calculation
        signals['high_channel'] = signals['high'].rolling(window=self.lookback).max()
        signals['low_channel'] = signals['low'].rolling(window=self.lookback).min()
        
        # Calculate breakout strength
        signals['breakout_strength'] = pd.Series(0, index=signals.index)
        above_high = signals['close'] > signals['high_channel']
        below_low = signals['close'] < signals['low_channel']
        
        signals.loc[above_high, 'breakout_strength'] = (
            (signals.loc[above_high, 'close'] - signals.loc[above_high, 'high_channel']) / 
            signals.loc[above_high, 'atr']
        )
        signals.loc[below_low, 'breakout_strength'] = (
            (signals.loc[below_low, 'low_channel'] - signals.loc[below_low, 'close']) / 
            signals.loc[below_low, 'atr']
        )
        
        signals['buy_signal'] = (
            (signals['close'] > signals['bb_upper']) &      
            (
                (signals['trend'] > 0) |                   
                (signals['rsi'] < self.rsi_oversold)
            ) &
            (
                (signals['momentum_ma'] > 0.01) |             
                (signals['volume_zscore'] > 1.2)
            )
        )
        
        signals['sell_signal'] = (
            (signals['close'] < signals['bb_lower']) &  
            (
                (signals['trend'] < 0) |
                (signals['rsi'] > self.rsi_overbought)
            ) &
            (
                (signals['momentum_ma'] < -0.01) |
                (signals['volume_zscore'] > 1.2)
            )
        )
        
        # Calculate volatility and momentum factors
        # volatility_factor = signals['atr'] / signals['close']
        # momentum_factor = abs(signals['momentum_ma']) / signals['close']
        
        # Dynamic stop loss and take profit
        signals['stop_loss'] = pd.Series(np.nan, index=signals.index)
        signals['take_profit'] = pd.Series(np.nan, index=signals.index)
        
        # Set stop loss and take profit for buy signals
        buy_mask = signals['buy_signal']
        signals.loc[buy_mask, 'stop_loss'] = signals.loc[buy_mask, 'close'] * (
            1 - self.stop_loss_factor * (
                signals.loc[buy_mask, 'atr'] / signals.loc[buy_mask, 'close'] +
                abs(signals.loc[buy_mask, 'momentum_ma']) / signals.loc[buy_mask, 'close']
            )
        )
        signals.loc[buy_mask, 'take_profit'] = signals.loc[buy_mask, 'close'] * (
            1 + self.take_profit_factor * (
                signals.loc[buy_mask, 'atr'] / signals.loc[buy_mask, 'close'] +
                abs(signals.loc[buy_mask, 'momentum_ma']) / signals.loc[buy_mask, 'close']
            )
        )

        # Same for sell signals
        sell_mask = signals['sell_signal']
        signals.loc[sell_mask, 'stop_loss'] = signals.loc[sell_mask, 'close'] * (
            1 + self.stop_loss_factor * (
                signals.loc[sell_mask, 'atr'] / signals.loc[sell_mask, 'close'] +
                abs(signals.loc[sell_mask, 'momentum_ma']) / signals.loc[sell_mask, 'close']
            )
        )
        signals.loc[sell_mask, 'take_profit'] = signals.loc[sell_mask, 'close'] * (
            1 - self.take_profit_factor * (
                signals.loc[sell_mask, 'atr'] / signals.loc[sell_mask, 'close'] +
                abs(signals.loc[sell_mask, 'momentum_ma']) / signals.loc[sell_mask, 'close']
            )
        )
        
        return signals

    def plot_signals(self, signals: pd.DataFrame) -> None:
        """Plot trading signals with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price with Bollinger Bands", "Momentum", "Volume & RSI"),
            row_heights=[0.5, 0.25, 0.25],
            specs=[[{"secondary_y": False}],
                  [{"secondary_y": False}],
                  [{"secondary_y": True}]]
        )

        # Price and BB plot
        fig.add_trace(go.Candlestick(
            x=signals.index,
            open=signals["open"],
            high=signals["high"],
            low=signals["low"],
            close=signals["close"],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['bb_upper'],
            mode="lines",
            line=dict(color="rgba(0, 100, 255, 0.3)"),
            name="BB Upper"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['bb_lower'],
            mode="lines",
            line=dict(color="rgba(0, 100, 255, 0.3)"),
            name="BB Lower",
            fill="tonexty"
        ), row=1, col=1)

        # Add buy/sell signals
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

        # Momentum plot
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['momentum_ma'],
            mode="lines",
            line=dict(color="blue"),
            name="Momentum MA"
        ), row=2, col=1)

        # Add zero line for momentum
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Volume analysis with RSI overlay
        fig.add_trace(go.Bar(
            x=signals.index,
            y=signals["volume"],
            marker_color=np.where(signals['volume_zscore'] > 1.0, 
                                '#FF0000',  # Solid red
                                '#808080'),  # Solid gray
            name="Volume"
        ), row=3, col=1, secondary_y=False)
        
        # Update y-axis range for volume to ensure bars are visible
        max_vol = signals['volume'].max()
        min_vol = signals['volume'].min()
        volume_range_padding = (max_vol - min_vol) * 0.1  # 10% padding
        fig.update_yaxes(range=[min_vol - volume_range_padding, max_vol + volume_range_padding], 
                        row=3, col=1, secondary_y=False)

        # RSI on secondary y-axis
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['rsi'],
            mode="lines",
            line=dict(color="purple"),
            name="RSI"
        ), row=3, col=1, secondary_y=True)

        # Add RSI levels on secondary y-axis
        fig.add_hline(y=self.rsi_overbought, line_dash="dash", line_color="red", 
                     row=3, col=1, secondary_y=True)
        fig.add_hline(y=self.rsi_oversold, line_dash="dash", line_color="green", 
                     row=3, col=1, secondary_y=True)

        fig.update_layout(
            title="Breakout Analysis Dashboard",
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            yaxis3=dict(
                showgrid=True,
                gridcolor='LightGrey',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='LightGrey',
                zerolinewidth=1
            )
        )

        # Update y-axes titles and ranges
        fig.update_yaxes(title="Price", row=1, col=1)
        fig.update_yaxes(title="Momentum", row=2, col=1)
        fig.update_yaxes(title="Volume", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title="RSI", row=3, col=1, secondary_y=True, range=[0, 100])
        fig.update_xaxes(title="Time", row=3, col=1)

        fig.show()