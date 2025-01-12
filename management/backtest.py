import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from utils.historical_data_loader import load_specific_csv_from_zip
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


@dataclass
class Position:
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    stop_loss: float
    take_profit: float

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: str
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'

class EnhancedBacktest:
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure"""
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")

    def __init__(
        self, 
        data: pd.DataFrame, 
        strategy: Union['BreakoutStrategy', 'MeanReversionStrategy'],
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.15,
        max_positions: int = 5,
        enable_fractional: bool = True,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001
    ):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.enable_fractional = enable_fractional
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self._validate_data(data)
        
        self.capital = initial_capital
        self.equity = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.signals = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_position_exposure(self) -> Dict[str, float]:
        """Calculate current position exposure by side"""
        exposure = {'long': 0.0, 'short': 0.0}
        for position in self.positions:
            exposure[position.side] += position.size
        return exposure

    def _can_open_position(self, side: str, price: float) -> bool:
        """Check if a new position can be opened"""
        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return False

        # Calculate current exposure
        exposure = self._get_position_exposure()
        
        # Don't allow opposite positions
        if side == 'long' and exposure['short'] > 0:
            return False
        if side == 'short' and exposure['long'] > 0:
            return False
        
        # Check if we have enough capital
        position_size = self._calculate_position_size(price)
        position_value = position_size * price
        commission = self._calculate_commission(position_value)
        
        return position_value + commission <= self.capital

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on current capital and position size percentage"""
        position_value = self.capital * self.position_size_pct
        return position_value / price if self.enable_fractional else int(position_value / price)

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price"""
        multiplier = 1 + (self.slippage_pct if side == 'long' else -self.slippage_pct)
        return price * multiplier

    def _calculate_commission(self, position_value: float) -> float:
        """Calculate commission for a trade"""
        return position_value * self.commission_pct

    def _check_stop_loss_take_profit(self, row: pd.Series, timestamp: datetime) -> None:
        """Check if any positions hit their stop-loss or take-profit levels"""
        for position in self.positions[:]:
            if position.side == 'long':
                # Check stop loss
                if row['low'] <= position.stop_loss:
                    self._close_position(position, timestamp, position.stop_loss, 'stop_loss')
                # Check take profit
                elif row['high'] >= position.take_profit:
                    self._close_position(position, timestamp, position.take_profit, 'take_profit')
            else:  # Short position
                # Check stop loss
                if row['high'] >= position.stop_loss:
                    self._close_position(position, timestamp, position.stop_loss, 'stop_loss')
                # Check take profit
                elif row['low'] <= position.take_profit:
                    self._close_position(position, timestamp, position.take_profit, 'take_profit')

    def _open_position(self, row: pd.Series, side: str) -> None:
        """Open a new position"""
        price = self._apply_slippage(row['close'], side)
        
        # Check if we can open a new position
        if not self._can_open_position(side, price):
            return

        size = self._calculate_position_size(price)
        position_value = size * price
        commission = self._calculate_commission(position_value)
        
        # Check if we have enough capital - going for 10k start
        if position_value + commission > self.capital:
            return

        self.capital -= (position_value + commission)
        
        # Create position object - need to check the kraken docs for this when using live
        position = Position(
            entry_price=price,
            size=size,
            side=side,
            entry_time=row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit']
        )
        
        self.positions.append(position)
        self.logger.info("Opened %s position: Size=%s, Price=%s, Time=%s", side, size, price, row.name)

    def _close_position(self, position: Position, timestamp: datetime, price: float, reason: str) -> None:
        """Close an existing position"""
        exit_price = self._apply_slippage(price, 'short' if position.side == 'long' else 'long')
        position_value = position.size * exit_price
        commission = self._calculate_commission(position_value)
        
        # Calculate PnL
        if position.side == 'long':
            pnl = position.size * (exit_price - position.entry_price) - commission
        else:
            pnl = position.size * (position.entry_price - exit_price) - commission
            
        pnl_pct = pnl / (position.size * position.entry_price)
        
        # Update capital. dont forget commission + fees here
        self.capital += position_value - commission
        
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )
        self.trades.append(trade)
        
        self.positions.remove(position)
        self.logger.info("Closed %s position: PnL=%.2f, PnL%%=%.2f, Reason=%s", position.side, pnl, pnl_pct * 100, reason)

    def _update_equity(self, row: pd.Series) -> None:
        """Update equity value including open positions"""
        open_positions_value = sum(
            pos.size * (row['close'] if pos.side == 'long' else pos.entry_price - row['close'])
            for pos in self.positions
        )
        self.equity = self.capital + open_positions_value
        self.equity_curve.append(self.equity)

    def run(self) -> Dict:
        """Run backtest"""
        self.logger.info("Starting backtest...")
        self.signals = self.strategy.generate_signals(self.data)
        
        self.equity_curve = []
        
        for timestamp, row in tqdm(self.signals.iterrows(), total=len(self.signals)):            
            # Check stop-loss and take-profit for existing positions
            self._check_stop_loss_take_profit(row, timestamp)
            
            # Process entry signals
            if row['buy_signal']:
                self._open_position(row, 'long')
            elif row['sell_signal']:
                self._open_position(row, 'short')
            
            self._update_equity(row)
        
        # Close any remaining positions at the end
        final_row = self.signals.iloc[-1]
        for position in self.positions[:]:
            self._close_position(position, final_row.name, final_row['close'], 'end_of_period')
        
        self.logger.info("Backtest completed.")
        return self._generate_results()

    def _generate_results(self) -> Dict:
        """Generate backtest results and metrics"""
        if not self.trades:
            return {
                "signals": self.signals,  # Include signals in results
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "total_profit": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }

        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        profitable_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = profitable_trades / len(self.trades)
        
        profits = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losses = [trade.pnl for trade in self.trades if trade.pnl <= 0]
        
        average_profit = np.mean(profits) if profits else 0
        average_loss = np.mean(losses) if losses else 0
        profit_factor = -sum(profits) / sum(losses) if losses else float('inf')
        
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = ((equity_curve - peak) / peak) * 100
        max_drawdown = drawdown.min()
        
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        return {
            "signals": self.signals,  # Include signals in results
            "total_trades": len(self.trades),
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "total_profit": self.equity - self.initial_capital,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": self.equity_curve
        }

    def plot_results(self) -> None:
        """Plot equity curve and drawdown"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        equity_series = pd.Series(self.equity_curve, index=self.data.index)
        equity_series.plot(ax=ax1, label='Equity')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        drawdown.plot(ax=ax2, label='Drawdown', color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_backtest_for_file(file_name, zip_path, initial_cash=10000, max_allocation_pct=0.25, plot=False):
    """Run the backtest for a single file."""
    data_dict = load_specific_csv_from_zip(zip_path, [file_name])
    data = data_dict.get(file_name)
    
    if data is None:
        raise ValueError(f"Failed to load data for {file_name}")

    # TODO: need a better way to switch between strategies rather than uncommenting
    # breakout_strategy = BreakoutStrategy(base_lookback=10, buffer=0.001, rsi_lookback=10, volume_lookback=10)
    mean_reversion_strategy = MeanReversionStrategy(base_lookback=20, rsi_lookback=20, volume_lookback=20)
    
    backtest = EnhancedBacktest(
        data=data,
        # strategy=breakout_strategy,
        strategy=mean_reversion_strategy,
        initial_capital=initial_cash,
        position_size_pct=max_allocation_pct,
        commission_pct=0.001, 
        slippage_pct=0.002  
    )

    results = backtest.run()

    if plot:
        backtest.plot_results()
        # breakout_strategy.plot_signals(results['signals'])  # Use signals from results
        mean_reversion_strategy.plot_signals(results['signals'])

    return {
        "file_name": file_name,
        "metrics": results,
        "trades": backtest.trades
    }


if __name__ == "__main__":
    ZIP_PATH = "data/raw/Kraken_OHLCVT.zip"
    files_to_test = ["XBTUSD_240.csv"]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_backtest_for_file, 
                file_name, 
                ZIP_PATH, 
                plot=True
            ) for file_name in files_to_test
        ]
        
        for future in as_completed(futures):
            results.append(future.result())

    for result in results:
        print(f"\nResults for {result['file_name']}:")
        print("Metrics:")
        for metric, value in result['metrics'].items():
            if metric not in ['equity_curve', 'signals']:  # Skip equity_curve and signals in printing
                print(f"  {metric}: {value}")
        print(f"Total trades executed: {len(result['trades'])}")