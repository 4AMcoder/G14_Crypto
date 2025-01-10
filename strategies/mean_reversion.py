import pandas as pd
import numpy as np

class MeanReversionStrategy:
    def __init__(self, lookback=20, threshold=0.02):
        """
        Initialize the mean reversion strategy.

        Parameters:
        - lookback (int): Number of periods for the moving average.
        - threshold (float): Deviation threshold for buy/sell signals.
        """
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data):
        """
        Generate buy/sell signals based on mean reversion logic.

        Parameters:
        - data (pd.DataFrame): Historical price data with a 'close' column.

        Returns:
        - signals (pd.DataFrame): Data with buy/sell signals.
        """
        signals = data.copy()
        signals['moving_avg'] = signals['close'].rolling(self.lookback).mean()

        # Calculate price deviation from the moving average
        signals['deviation'] = (signals['close'] - signals['moving_avg']) / signals['moving_avg']

        # Generate signals
        signals['buy_signal'] = signals['deviation'] < -self.threshold
        signals['sell_signal'] = signals['deviation'] > self.threshold

        return signals
