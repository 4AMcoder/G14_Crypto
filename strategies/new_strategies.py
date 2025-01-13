from backtesting import Strategy
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional


class BreakoutStrategy(Strategy):
    n_lookback = 20
    buffer = 0.001
    rsi_lookback = 14
    volume_lookback = 20
    rc_threshold = 0.5
    volume_multiplier = 0.1
    pivot_window = 15
    breakout_threshold = 0.0015
    bb_std = 2.0
    stop_loss_factor = 1.0
    take_profit_factor = 2.0
    size_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_middle = self.I(lambda: pd.Series(close).ewm(span=self.n_lookback).mean())
        bb_std = self.I(lambda: pd.Series(close).rolling(self.n_lookback).std())
        self.bb_upper = self.I(lambda: self.bb_middle + (bb_std * self.bb_std))
        self.bb_lower = self.I(lambda: self.bb_middle - (bb_std * self.bb_std))

        # RSI
        delta = pd.Series(close).diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        avg_gain = gains.ewm(com=self.rsi_lookback-1, min_periods=self.rsi_lookback).mean()
        avg_loss = losses.ewm(com=self.rsi_lookback-1, min_periods=self.rsi_lookback).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        self.rsi = self.I(lambda: 100 - (100 / (1 + rs)), overlay=False, name='RSI')

        # Volume analysis
        self.volume_ma = self.I(lambda: pd.Series(volume).ewm(span=self.volume_lookback).mean())
        self.strong_volume = volume > self.volume_ma * (1 + self.volume_multiplier)

        # Robert Carver signal
        rolling_mean = pd.Series(close).ewm(span=self.n_lookback).mean()
        rolling_std = pd.Series(close).rolling(self.n_lookback).std()
        self.rc_signal = self.I(lambda: (close - rolling_mean) / rolling_std.replace(0, np.finfo(float).eps))

        # ATR for position sizing
        tr1 = high - low
        tr2 = abs(high - pd.Series(close).shift(1))
        tr3 = abs(low - pd.Series(close).shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        self.atr = self.I(lambda: pd.Series(tr).ewm(span=self.n_lookback).mean())

    def next(self):
        price = self.data.Close[-1]
        
        # Buy conditions
        buy_signal = (
            self.data.High[-1] > self.bb_upper[-1] and
            self.data.Close[-1] > self.bb_middle[-1] and
            self.rsi[-1] > 30 and self.rsi[-1] < 80 and
            (self.rc_signal[-1] > -self.rc_threshold or self.strong_volume[-1])
        )

        # Sell conditions
        sell_signal = (
            self.data.Low[-1] < self.bb_lower[-1] and
            self.data.Close[-1] < self.bb_middle[-1] and
            self.rsi[-1] < 70 and self.rsi[-1] > 20 and
            (self.rc_signal[-1] < self.rc_threshold or self.strong_volume[-1])
        )

        if buy_signal and not self.position:
            stop_loss = price - self.atr[-1] * self.stop_loss_factor
            take_profit = price + self.atr[-1] * self.take_profit_factor
            self.buy(size=self.size_pct, sl=stop_loss, tp=take_profit)

        elif sell_signal and not self.position:
            stop_loss = price + self.atr[-1] * self.stop_loss_factor
            take_profit = price - self.atr[-1] * self.take_profit_factor
            self.sell(size=self.size_pct, sl=stop_loss, tp=take_profit)


class MeanReversionStrategy(Strategy):
    n_lookback = 10          
    std_dev_threshold = 1.5  
    rsi_lookback = 10
    rsi_overbought = 75     
    rsi_oversold = 25        
    volume_lookback = 10     
    volume_threshold = 1.2   
    stop_loss_factor = 1.5   
    take_profit_factor = 2.0 
    zscore_threshold = 1.0 
    size_pct = 0.02 

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_middle = self.I(lambda: pd.Series(close).rolling(self.n_lookback).mean())
        bb_std = self.I(lambda: pd.Series(close).rolling(self.n_lookback).std())
        self.bb_upper = self.I(lambda: self.bb_middle + (bb_std * self.std_dev_threshold))
        self.bb_lower = self.I(lambda: self.bb_middle - (bb_std * self.std_dev_threshold))

        # RSI
        delta = pd.Series(close).diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        avg_gain = gains.ewm(com=self.rsi_lookback-1, min_periods=self.rsi_lookback).mean()
        avg_loss = losses.ewm(com=self.rsi_lookback-1, min_periods=self.rsi_lookback).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        self.rsi = self.I(lambda: 100 - (100 / (1 + rs)), overlay=False, name='RSI')

        # Z-score
        close_series = pd.Series(close)
        mean = close_series.rolling(self.n_lookback).mean()
        std = close_series.rolling(self.n_lookback).std()
        self.zscore = self.I(lambda: (close - mean) / std.replace(0, np.finfo(float).eps))

        # Volume analysis
        self.volume_ma = self.I(lambda: pd.Series(volume).rolling(self.volume_lookback).mean())
        self.volume_ratio = volume / self.volume_ma

        # ATR for position sizing
        tr1 = high - low
        tr2 = abs(high - pd.Series(close).shift(1))
        tr3 = abs(low - pd.Series(close).shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        self.atr = self.I(lambda: pd.Series(tr).ewm(span=self.n_lookback).mean())

    def next(self):
        price = self.data.Close[-1]

        zscore_score = -self.zscore[-1] / self.zscore_threshold
        rsi_score = (50 - self.rsi[-1]) / 50
        bb_score = (self.bb_middle[-1] - price) / (self.bb_upper[-1] - self.bb_middle[-1])
        volume_score = (self.volume_ratio[-1] - 1) / self.volume_threshold

        long_score = (
            zscore_score * 0.3 +  
            rsi_score * 0.3 +     
            bb_score * 0.3 +     
            volume_score * 0.1 
        )

        short_score = -long_score # making this opposite for short
        threshold = 0.4  # hardcoded this in for now

        if not self.position:
            if long_score > threshold:
                stop_loss = price - self.atr[-1] * self.stop_loss_factor
                take_profit = price + self.atr[-1] * self.take_profit_factor
                self.buy(size=self.size_pct, sl=stop_loss, tp=take_profit)
                
            elif short_score > threshold:
                stop_loss = price + self.atr[-1] * self.stop_loss_factor
                take_profit = price - self.atr[-1] * self.take_profit_factor
                self.sell(size=self.size_pct, sl=stop_loss, tp=take_profit)