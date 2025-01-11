import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from utils.historical_data_loader import load_specific_csv_from_zip
import mplfinance as mpf


class Backtest:
    def __init__(self, data, strategy, initial_cash=10000, max_allocation_pct=0.15):
        """
        Initialize the backtest.

        Parameters:
        - data (pd.DataFrame): Historical OHLC data.
        - strategy (BreakoutStrategy): Strategy object to generate signals.
        - initial_cash (float): Starting cash for the portfolio.
        - max_allocation_pct (float): Maximum percentage of portfolio per trade.
        """
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.portfolio_value = initial_cash
        self.max_allocation_pct = max_allocation_pct
        self.trades = []

    def run(self):
        """
        Run the backtest.
        """
        signals = self.strategy.generate_signals(self.data, timeframe="240min")

        total_steps = len(signals)
        with tqdm(total=total_steps, desc="Backtest Progress") as pbar:
            for i, row in signals.iterrows():
                price = row["close"]

                # Calculate maximum allocation for the current portfolio
                max_trade_amount = self.portfolio_value * self.max_allocation_pct

                # Execute buy signal
                if row["buy_signal"] and self.cash > 0:
                    trade_amount = min(self.cash, max_trade_amount)
                    self.position = trade_amount / price
                    self.cash -= trade_amount
                    self.trades.append({
                        "timestamp": i,
                        "action": "buy",
                        "price": price,
                        "amount": trade_amount,
                        "position": self.position,
                    })

                # Execute sell signal
                elif row["sell_signal"] and self.position > 0:
                    sell_amount = self.position * price
                    self.cash += sell_amount
                    self.position = 0
                    self.trades.append({
                        "timestamp": i,
                        "action": "sell",
                        "price": price,
                        "amount": sell_amount,
                        "position": self.position,
                    })

                # Update portfolio value
                self.portfolio_value = self.cash + self.position * price

                # Update the progress bar
                pbar.update(1)

        # Store portfolio value in the signals DataFrame
        signals["portfolio_value"] = self.cash + self.position * signals["close"]
        self.data["portfolio_value"] = signals["portfolio_value"]
        return signals

    def calculate_metrics(self):
        """
        Calculate performance metrics.
        """
        # Portfolio returns
        self.data["returns"] = self.data["portfolio_value"].pct_change()

        # Sharpe ratio
        risk_free_rate = 0.01 / 252  # Daily risk-free rate (e.g., US Treasury yield)
        excess_returns = self.data["returns"].mean() - risk_free_rate
        sharpe_ratio = excess_returns / self.data["returns"].std() if self.data["returns"].std() != 0 else 0

        # Max drawdown
        running_max = self.data["portfolio_value"].cummax()
        drawdown = running_max - self.data["portfolio_value"]
        max_drawdown = drawdown.max()

        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "cumulative_return": self.data["portfolio_value"].iloc[-1] / self.data["portfolio_value"].iloc[0] - 1,
        }


def run_backtest_for_file(file_name, zip_path, initial_cash=10000, max_allocation_pct=0.15, plot=False):
    """
    Run the backtest for a single file.

    Parameters:
    - file_name (str): Name of the CSV file in the ZIP archive.
    - zip_path (str): Path to the ZIP file.
    - initial_cash (float): Starting cash for the backtest.
    - max_allocation_pct (float): Maximum percentage of portfolio per trade.
    - plot (bool): Whether to plot the candlestick chart (deferred to main thread).

    Returns:
    - dict: Backtest metrics and optionally plotting data.
    """
    data_dict = load_specific_csv_from_zip(zip_path, [file_name])
    data = data_dict.get(file_name)

    breakout_strategy = BreakoutStrategy(base_lookback=10, buffer=0.001, rsi_lookback=10, volume_lookback=10)
    mean_reversion_strategy = MeanReversionStrategy(base_lookback=20, rsi_lookback=20, volume_lookback=10)

    # backtest = Backtest(data, breakout_strategy, initial_cash=initial_cash, max_allocation_pct=max_allocation_pct)

    backtest = Backtest(data, mean_reversion_strategy, initial_cash=initial_cash, max_allocation_pct=max_allocation_pct)
    output = backtest.run()
    metrics = backtest.calculate_metrics()

    result = {"file_name": file_name, "metrics": metrics, "trades": backtest.trades}

    if plot:
        # breakout_strategy.plot_signals(output)
        mean_reversion_strategy.plot_signals(output)

    return result


if __name__ == "__main__":
    zip_path = "data/raw/Kraken_OHLCVT.zip"
    files_to_test = ["XBTUSD_240.csv"]

    # Run backtests in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_backtest_for_file, file_name, zip_path, plot=True) for file_name in files_to_test]
        for future in as_completed(futures):
            results.append(future.result())

    # Display results
    for result in results:
        print(f"File: {result['file_name']}")
        print("Metrics:", result["metrics"])
        print("Trades Executed:", len(result["trades"]))
