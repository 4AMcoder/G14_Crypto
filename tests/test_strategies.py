import pytest
import pandas as pd
from strategies.breakout import BreakoutStrategy


@pytest.fixture
def synthetic_data():
    """
    Create synthetic data to include both bullish breakout and bearish breakdown scenarios.
    """
    data = {
    "timestamp": [
        1672531200, 1672531260, 1672531320, 1672531380, 1672531440,
        1672531500, 1672531560, 1672531620, 1672531680, 1672531740,
        1672531800, 1672531860, 1672531920, 1672531980, 1672532040,
        1672532100, 1672532160, 1672532220, 1672532280, 1672532340,
        1672532400, 1672532460, 1672532520, 1672532580, 1672532640,
        1672532700, 1672532760, 1672532820, 1672532880, 1672532940,
        1672533000, 1672533060, 1672533120, 1672533180, 1672533240,
    ],
    "open": [
        100, 101, 99, 100, 101,
        99, 100, 101, 99, 100,
        101, 102, 101, 103, 102,
        103, 110, 115, 125, 135,
        140, 138, 135, 132, 128,
        125, 115, 105, 95, 85,
        80, 82, 81, 83, 82,
    ],
    "high": [
        102, 103, 101, 102, 103,
        101, 102, 103, 101, 102,
        103, 104, 103, 105, 104,
        112, 116, 120, 130, 142,
        142, 140, 137, 134, 130,
        127, 118, 108, 97, 87,
        83, 85, 84, 85, 84,
    ],
    "low": [
        98, 99, 97, 98, 99,
        97, 98, 99, 97, 98,
        99, 100, 99, 101, 100,
        101, 108, 113, 123, 133,
        136, 134, 131, 128, 125,
        122, 112, 102, 92, 82,
        78, 80, 79, 81, 80,
    ],
    "close": [
        101, 99, 100, 101, 99,
        100, 101, 99, 100, 101,
        102, 101, 103, 102, 103,
        110, 115, 125, 135, 140,
        138, 135, 132, 128, 125,
        115, 105, 95, 85, 80,
        82, 81, 83, 82, 81,
    ],
    "volume": [
        1000, 1000, 1000, 1000, 1000,
        1000, 1000, 1000, 1000, 1000,
        1200, 1300, 1400, 1500, 1600,
        5000, 8000, 10000, 12000, 9000,
        7000, 6000, 5000, 4500, 4000,
        6000, 8000, 10000, 9000, 8000,
        3000, 2000, 1500, 1200, 1000,
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

    strategy = BreakoutStrategy(base_lookback=5, rsi_lookback=5, volume_lookback=5, rc_threshold=1.35, pivot_window=5)

    signals = strategy.generate_signals(synthetic_data, timeframe="1min")

    expected_buy_timestamps = [pd.Timestamp('2023-01-01 00:06:00'), pd.Timestamp('2023-01-01 00:09:00'), pd.Timestamp('2023-01-01 00:11:00'),pd.Timestamp('2023-01-01 00:13:00'),pd.Timestamp('2023-01-01 00:14:00')]
    expected_sell_timestamps = [pd.Timestamp('2023-01-01 00:32:00'), pd.Timestamp('2023-01-01 00:33:00')]

    buy_timestamps = signals[signals["buy_signal"]].index.tolist()
    sell_timestamps = signals[signals["sell_signal"]].index.tolist()

    # Assert buy/sell signals
    assert buy_timestamps == expected_buy_timestamps, f"Unexpected buy signals: {buy_timestamps}"
    assert sell_timestamps == expected_sell_timestamps, f"Unexpected sell signals: {sell_timestamps}"

    # Check volume confirmation
    # assert all(signals.loc[buy_timestamps, "strong_volume"]), "Buy signals lack volume confirmation."
    # assert all(signals.loc[sell_timestamps, "strong_volume"]), "Sell signals lack volume confirmation."


def test_no_signals_on_stable_data():
    """
    Test that no buy/sell signals are generated on stable data.

    # TODO: can just parametrize this test
    """
    # Stable dataset with no breakouts
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="T"),
        "high": [100] * 20,
        "low": [90] * 20,
        "close": [95] * 20,
        "volume": [1000] * 20,
    }
    stable_data = pd.DataFrame(data).set_index("timestamp")

    breakout = BreakoutStrategy(base_lookback=5, buffer=0.01, stop_loss_factor=1.0, take_profit_factor=2.0)
    signals = breakout.generate_signals(stable_data)

    # Ensure no signals are generated
    assert signals["buy_signal"].sum() == 0, "Buy signals generated on stable data!"
    assert signals["sell_signal"].sum() == 0, "Sell signals generated on stable data!"


def plot_synthetic_data_with_metrics():
    # Same as synthetic Data, cant directly refer to fixture
    data = {
    "timestamp": [
        1672531200, 1672531260, 1672531320, 1672531380, 1672531440,
        1672531500, 1672531560, 1672531620, 1672531680, 1672531740,
        1672531800, 1672531860, 1672531920, 1672531980, 1672532040,
        1672532100, 1672532160, 1672532220, 1672532280, 1672532340,
        1672532400, 1672532460, 1672532520, 1672532580, 1672532640,
        1672532700, 1672532760, 1672532820, 1672532880, 1672532940,
        1672533000, 1672533060, 1672533120, 1672533180, 1672533240,
    ],
    "open": [
        100, 101, 99, 100, 101,
        99, 100, 101, 99, 100,
        101, 102, 101, 103, 102,
        103, 110, 115, 125, 135,
        140, 138, 135, 132, 128,
        125, 115, 105, 95, 85,
        80, 82, 81, 83, 82,
    ],
    "high": [
        102, 103, 101, 102, 103,
        101, 102, 103, 101, 102,
        103, 104, 103, 105, 104,
        112, 116, 120, 130, 142,
        142, 140, 137, 134, 130,
        127, 118, 108, 97, 87,
        83, 85, 84, 85, 84,
    ],
    "low": [
        98, 99, 97, 98, 99,
        97, 98, 99, 97, 98,
        99, 100, 99, 101, 100,
        101, 108, 113, 123, 133,
        136, 134, 131, 128, 125,
        122, 112, 102, 92, 82,
        78, 80, 79, 81, 80,
    ],
    "close": [
        101, 99, 100, 101, 99,
        100, 101, 99, 100, 101,
        102, 101, 103, 102, 103,
        110, 115, 125, 135, 140,
        138, 135, 132, 128, 125,
        115, 105, 95, 85, 80,
        82, 81, 83, 82, 81,
    ],
    "volume": [
        1000, 1000, 1000, 1000, 1000,
        1000, 1000, 1000, 1000, 1000,
        1200, 1300, 1400, 1500, 1600,
        5000, 8000, 10000, 12000, 9000,
        7000, 6000, 5000, 4500, 4000,
        6000, 8000, 10000, 9000, 8000,
        3000, 2000, 1500, 1200, 1000,
    ],
}
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)

    strategy = BreakoutStrategy(base_lookback=5, rsi_lookback=5, volume_lookback=5, rc_threshold=1.35)

    result = strategy.generate_signals(df, timeframe="1min")
    strategy.plot_signals(result)


if __name__ == "__main__":
    # run the file to see / test plots
    plot_synthetic_data_with_metrics()