import pytest
import pandas as pd
import numpy as np
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.breakout import BreakoutStrategy
import mplfinance as mpf


@pytest.fixture
def sample_data():
    """
    Create a sample DataFrame with historical price data for testing.
    """
    dates = pd.date_range(start="2023-01-01", periods=100, freq="T")
    data = pd.DataFrame({
        "timestamp": dates,
        "close": np.sin(np.linspace(0, 10, 100)) + 100  # Simulated price data
    })
    return data.set_index("timestamp")

def test_mean_reversion_strategy(sample_data):
    """
    Test the Mean Reversion Strategy signal generation.
    """
    strategy = MeanReversionStrategy(lookback=5, threshold=0.02)
    signals = strategy.generate_signals(sample_data)

    # Check that the signals DataFrame contains the necessary columns
    assert 'buy_signal' in signals.columns
    assert 'sell_signal' in signals.columns

    # Validate that at least some signals are generated
    assert signals['buy_signal'].sum() > 0
    assert signals['sell_signal'].sum() > 0

def test_momentum_strategy(sample_data):
    """
    Test the Momentum Strategy signal generation.
    """
    strategy = MomentumStrategy(lookback=5)
    signals = strategy.generate_signals(sample_data)

    # Check that the signals DataFrame contains the necessary columns
    assert 'buy_signal' in signals.columns
    assert 'sell_signal' in signals.columns

    # Validate that at least some signals are generated
    assert signals['buy_signal'].sum() > 0
    assert signals['sell_signal'].sum() > 0

@pytest.fixture
def synthetic_data():
    """
    Create an elongated synthetic dataset with clearer breakout and crash conditions.
    """
    data = {
        "timestamp": [
            1672531200, 1672531260, 1672531320, 1672531380, 1672531440,  # Calm period
            1672531500, 1672531560, 1672531620, 1672531680, 1672531740,  # Calm period continues
            1672531800, 1672531860, 1672531920, 1672531980, 1672532040,  # Breakout spike
            1672532100, 1672532160, 1672532220, 1672532280, 1672532340,  # Breakout spike continues
            1672532400, 1672532460, 1672532520, 1672532580, 1672532640,  # Steep crash
            1672532700, 1672532760, 1672532820, 1672532880, 1672532940,  # Steep crash continues
            1672533000, 1672533060, 1672533120, 1672533180, 1672533240,  # Calm recovery
        ],
        "open": [
            95, 96, 98, 95, 98,  # Calm
            99, 91, 95, 95, 95,  # Calm continues
            110, 130, 145, 160, 180,  # Spike up
            175, 165, 155, 145, 140,  # Spike peak
            120, 100, 85, 70, 60,  # Steep crash
            65, 75, 85, 95, 100,  # Recovery
            100, 100, 100, 100, 100,  # Calm
        ],
        "high": [
            100, 100, 100, 100, 100,  # Calm
            100, 100, 100, 100, 100,  # Calm continues
            115, 135, 150, 165, 185,  # Spike up
            180, 170, 160, 150, 145,  # Spike peak
            125, 105, 90, 75, 65,  # Steep crash
            70, 80, 90, 100, 105,  # Recovery
            100, 100, 100, 100, 100,  # Calm
        ],
        "low": [
            90, 90, 90, 90, 90,  # Calm
            90, 90, 90, 90, 90,  # Calm continues
            105, 125, 140, 155, 175,  # Spike up
            170, 160, 150, 140, 135,  # Spike peak
            115, 95, 80, 65, 55,  # Steep crash
            60, 70, 80, 90, 95,  # Recovery
            100, 100, 100, 100, 100,  # Calm
        ],
        "close": [
            96, 98, 95, 98, 99,  # Calm
            91, 95, 95, 95, 95,  # Calm continues
            110, 130, 145, 170, 210,  # Spike up
            195, 170, 155, 145, 140,  # Spike peak
            120, 100, 85, 70, 40,  # Steep crash
            60, 70, 80, 90, 93,  # Recovery
            95, 103, 105, 100, 97,  # Calm
        ],
        "volume": [
            1000, 1000, 1000, 1000, 1000,  # Calm
            1000, 1000, 1000, 1000, 1000,  # Calm continues
            2000, 4000, 8000, 8500, 7000,  # Spike up
            7500, 6000, 4500, 4000, 3500,  # Spike peak
            3000, 2500, 2000, 1500, 1200,  # Steep crash
            1300, 1400, 1500, 1600, 1700,  # Recovery
            1000, 1000, 1000, 1000, 1000,  # Calm
        ],
    }


    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)

    return df


def test_breakout_signals(synthetic_data):
    """
    Test the comprehensive breakout strategy with elongated synthetic data.
    """

    strategy = BreakoutStrategy(lookback=5, rsi_lookback=5, volume_lookback=5, rc_threshold=1.35)

    signals = strategy.generate_signals(synthetic_data)

    result_df = signals[["close", "rsi", "rc_signal", "donchian_breach_upper", "donchian_breach_lower", "buy_signal", "sell_signal"]]
    result_df.to_csv("result.csv") # manual check

    expected_buy_timestamps = [pd.Timestamp('2023-01-01 00:13:00')]  
    expected_sell_timestamps = [] # TODO: no sell signal. will parametrize this later for more scenarios

    buy_timestamps = signals[signals["buy_signal"]].index.tolist()
    sell_timestamps = signals[signals["sell_signal"]].index.tolist()

    # Assert buy/sell signals
    assert buy_timestamps == expected_buy_timestamps, f"Unexpected buy signals: {buy_timestamps}"
    assert sell_timestamps == expected_sell_timestamps, f"Unexpected sell signals: {sell_timestamps}"

    # Check volume confirmation
    assert all(signals.loc[buy_timestamps, "volume_confirm"]), "Buy signals lack volume confirmation."
    assert all(signals.loc[sell_timestamps, "volume_confirm"]), "Sell signals lack volume confirmation."


def test_no_signals_on_stable_data():
    """
    Test that no buy/sell signals are generated on stable data.
    """
    # Stable dataset with no breakouts
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="T"),
        "high": [100] * 20,
        "low": [90] * 20,
        "close": [95] * 20,
    }
    stable_data = pd.DataFrame(data).set_index("timestamp")

    breakout = BreakoutStrategy(lookback=5, buffer=0.01, stop_loss_factor=1.0, take_profit_factor=2.0)
    signals = breakout.generate_signals(stable_data)

    # Ensure no signals are generated
    assert signals["buy_signal"].sum() == 0, "Buy signals generated on stable data!"
    assert signals["sell_signal"].sum() == 0, "Sell signals generated on stable data!"


def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Parameters:
    - series: A pandas Series of price data (e.g., close prices).
    - window: The lookback window for RSI calculation.

    Returns:
    - pd.Series: RSI values.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)  # Positive price changes
    loss = -delta.clip(upper=0)  # Neg price changes

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss # Relative strength
    rsi = 100 - (100 / (1 + rs))

    return rsi

def plot_synthetic_data_with_metrics():
    # Synthetic Data
    data = {
        "timestamp": [
            1672531200, 1672531260, 1672531320, 1672531380, 1672531440,  # Calm period
            1672531500, 1672531560, 1672531620, 1672531680, 1672531740,  # Calm period continues
            1672531800, 1672531860, 1672531920, 1672531980, 1672532040,  # Breakout spike
            1672532100, 1672532160, 1672532220, 1672532280, 1672532340,  # Breakout spike continues
            1672532400, 1672532460, 1672532520, 1672532580, 1672532640,  # Steep crash
            1672532700, 1672532760, 1672532820, 1672532880, 1672532940,  # Steep crash continues
            1672533000, 1672533060, 1672533120, 1672533180, 1672533240,  # Calm recovery
        ],
        "open": [
            95, 96, 98, 95, 98,
            99, 91, 95, 95, 95, 
            110, 130, 145, 160, 180,  
            175, 165, 155, 145, 140,  
            120, 100, 85, 70, 60, 
            65, 75, 85, 95, 100, 
            100, 100, 100, 100, 100, 
        ],
        "high": [
            100, 100, 100, 100, 100, 
            100, 100, 100, 100, 100, 
            115, 135, 150, 165, 185, 
            180, 170, 160, 150, 145,  
            125, 105, 90, 75, 65, 
            70, 80, 90, 100, 105, 
            100, 100, 100, 100, 100, 
        ],
        "low": [
            90, 90, 90, 90, 90, 
            90, 90, 90, 90, 90,  
            105, 125, 140, 155, 175, 
            170, 160, 150, 140, 135,
            115, 95, 80, 65, 55,  
            60, 70, 80, 90, 95,  
            100, 100, 100, 100, 100,  
        ],
        "close": [
            96, 98, 95, 98, 99,
            91, 95, 95, 95, 95,  
            110, 130, 145, 170, 210, 
            195, 170, 155, 145, 140, 
            120, 100, 85, 70, 40,  
            60, 70, 80, 90, 93, 
            95, 103, 105, 100, 97,  
        ],
        "volume": [
            1000, 1000, 1000, 1000, 1000,  
            1000, 1000, 1000, 1000, 1000,  
            2000, 4000, 8000, 8500, 7000,  
            7500, 6000, 4500, 4000, 3500,  
            3000, 2500, 2000, 1500, 1200,  
            1300, 1400, 1500, 1600, 1700,  
            1000, 1000, 1000, 1000, 1000, 
        ],
    }

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)

    strategy = BreakoutStrategy(lookback=5, rsi_lookback=5, volume_lookback=5, rc_threshold=1.35)

    df = strategy.generate_signals(df)
    print(df)

    add_plots = [
        mpf.make_addplot(df["rsi"], panel=1, ylabel="RSI", color="blue"),
        mpf.make_addplot(df["upper_channel"], color="green"),
        mpf.make_addplot(df["lower_channel"], color="red"),
    ]

    mpf.plot(
        df,
        type="candle",
        title="Synthetic Data with RSI and Donchian Channels",
        style="charles",
        volume=True,
        addplot=add_plots,
        ylabel="Price",
        xlabel="Time",
    )

    mpf.show()


if __name__ == "__main__":
    plot_synthetic_data_with_metrics()