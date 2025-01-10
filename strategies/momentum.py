import pandas as pd
import numpy as np

class MomentumStrategy:
    def __init__(self, lookback=10):
        """
        Initialize the momentum strategy.

        Parameters:
        - lookback (int): Number of periods for calculating momentum.
        """
        self.lookback = lookback

    def generate_signals(self, data):
        """
        Generate buy/sell signals based on momentum logic.

        Parameters:
        - data (pd.DataFrame): Historical price data with a 'close' column.

        Returns:
        - signals (pd.DataFrame): Data with buy/sell signals.
        """
        signals = data.copy()

        # Calculate Rate of Change (ROC)
        signals['roc'] = (signals['close'] - signals['close'].shift(self.lookback)) / signals['close'].shift(self.lookback)

        # Generate signals
        signals['buy_signal'] = signals['roc'] > 0
        signals['sell_signal'] = signals['roc'] < 0

        return signals
