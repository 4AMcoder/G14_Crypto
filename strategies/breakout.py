import numpy as np
import pandas as pd
from typing import Optional


class BreakoutStrategy:
    def __init__(
        self,
        lookback: int = 20,
        buffer: float = 0.0001,
        stop_loss_factor: float = 1.0,
        take_profit_factor: float = 2.0,
        rsi_lookback: int = 14,
        volume_lookback: int = 20,
        rc_threshold: float = 1.0,
        volume_multiplier: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.buffer = buffer
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.rsi_lookback = rsi_lookback
        self.volume_lookback = volume_lookback
        self.rc_threshold = rc_threshold
        self.volume_multiplier = volume_multiplier

    def calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_robert_carver(self, data: pd.DataFrame) -> pd.Series:
        rolling_mean = data["close"].rolling(self.lookback).mean()
        rolling_std = data["close"].rolling(self.lookback).std()
        return (data["close"] - rolling_mean) / rolling_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()

        # Donchian Channels
        signals["upper_channel"] = signals["high"].rolling(self.lookback).max()
        signals["lower_channel"] = signals["low"].rolling(self.lookback).min()

        # RSI
        signals["rsi"] = self.calculate_rsi(signals["close"], self.rsi_lookback)

        # RC Signal
        signals["rc_signal"] = self.calculate_robert_carver(signals)

        # Volume
        signals["volume_mean"] = signals["volume"].rolling(self.volume_lookback).mean()
        signals["volume_std"] = signals["volume"].rolling(self.volume_lookback).std()
        signals["volume_confirm"] = (
            signals["volume"] > (signals["volume_mean"] + self.volume_multiplier * signals["volume_std"])
        )

        # Donchian breaches
        signals["donchian_breach_upper"] = signals["close"] > signals["upper_channel"]
        signals["donchian_breach_lower"] = signals["close"] < signals["lower_channel"]

        # Buy signal
        signals["buy_signal"] = (
            (signals["donchian_breach_upper"])
            & (signals["rsi"] > 50)
            & (signals["rc_signal"] > self.rc_threshold)
            & (signals["volume_confirm"])
        )

        # Sell signal
        signals["sell_signal"] = (
            (signals["donchian_breach_lower"])
            & (signals["rsi"] < 50)
            & (signals["rc_signal"] < -self.rc_threshold)
            & (signals["volume_confirm"])
        )

        # Combined signal for easier debugging
        signals["triggered_signal"] = signals["buy_signal"] | signals["sell_signal"]

        # Stop-loss and take-profit levels
        recent_volatility = signals["high"] - signals["low"]
        signals["stop_loss"] = signals["close"] - recent_volatility * self.stop_loss_factor
        signals["take_profit"] = signals["close"] + recent_volatility * self.take_profit_factor

        # Drop NaNs so final signals are valid
        signals.dropna(inplace=True)

        return signals
