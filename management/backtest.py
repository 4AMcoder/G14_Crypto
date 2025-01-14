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
from utils.logger import setup_logger, get_logger
import itertools
import yaml


@dataclass
class Position:
    entry_price: float
    size: float
    side: str  # note to self 'long' or 'short'
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
    exit_reason: str  # note to self 'signal', 'stop_loss', 'take_profit'

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
        self.logger = get_logger("trading_bot.backtest")
        self.logger.info("Backtest logger initialized")
        
        self._validate_data(data)
        
        self.capital = initial_capital
        self.equity = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.signals = None

    def _get_position_exposure(self) -> Dict[str, float]:
        """Calculate current position exposure by side"""
        exposure = {'long': 0.0, 'short': 0.0}
        for position in self.positions:
            exposure[position.side] += position.size
        return exposure

    def _can_open_position(self, side: str, price: float) -> bool:
        """Check if a new position can be opened"""
        # Check max positions limit. note this needs adjusting alot 
        if len(self.positions) >= self.max_positions:
            return False

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
            self._check_stop_loss_take_profit(row, timestamp)
            
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
                "signals": self.signals,
                "total_trades": 0,
                "return_metrics": {},
                "risk_metrics": {},
                "trade_metrics": {},
                "time_metrics": {}
            }

        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        # Buy & Hold Return
        buy_hold_return = ((self.data['close'].iloc[-1] - self.data['close'].iloc[0]) / 
                        self.data['close'].iloc[0] * 100)
        
        # Return Metrics 
        equity_final = self.equity
        equity_peak = max(self.equity_curve)
        total_return = ((self.equity - self.initial_capital) / self.initial_capital * 100)
        
        # Annualization factor 
        time_diff = (self.data.index[-1] - self.data.index[0]).total_seconds()
        ann_factor = 365 * 24 * 60 * 60 / time_diff
        
        ann_return = ((1 + total_return/100) ** ann_factor - 1) * 100
        ann_volatility = returns.std() * np.sqrt(252) * 100
        
        # Exposure Time
        exposure_time = sum(len(self.positions) > 0 for _ in self.data.index) / len(self.data) * 100
        
        # Trade Analysis
        trade_returns = [(t.exit_price - t.entry_price) / t.entry_price * 100 if t.side == 'long'
                        else (t.entry_price - t.exit_price) / t.entry_price * 100 for t in self.trades]
        
        profitable_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = profitable_trades / len(self.trades) * 100 if self.trades else 0
        
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        
        # Risk Metrics
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = ((equity_curve - peak) / peak * 100)
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Advanced Risk Metrics
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() * np.sqrt(252)) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
        calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Trade Durations
        trade_durations = [(t.exit_time - t.entry_time).total_seconds()/86400 for t in self.trades]
        max_trade_duration = max(trade_durations) if trade_durations else 0
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # System Quality Number (SQN)
        trade_returns_arr = np.array(trade_returns)
        sqn = np.sqrt(len(trade_returns)) * (np.mean(trade_returns) / np.std(trade_returns)) if len(trade_returns) > 0 else 0
        
        # Profit Factor and Expectancy
        gains = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        profit_factor = abs(sum(gains) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        expectancy = (win_rate/100 * np.mean(gains) + (1-win_rate/100) * np.mean(losses)) if gains and losses else 0
        
        return {
            "signals": self.signals,
            "return_metrics": {
                "Equity Final [$]": round(equity_final, 2),
                "Equity Peak [$]": round(equity_peak, 2),
                "Return [%]": round(total_return, 2),
                "Buy & Hold Return [%]": round(buy_hold_return, 2),
                "Return (Ann.) [%]": round(ann_return, 2),
                "Volatility (Ann.) [%]": round(ann_volatility, 2),
                "Exposure Time [%]": round(exposure_time, 2)
            },
            "risk_metrics": {
                "Sharpe Ratio": round(returns.mean() / returns.std() * np.sqrt(252), 2),
                "Sortino Ratio": round(sortino_ratio, 2),
                "Calmar Ratio": round(calmar_ratio, 2),
                "Max. Drawdown [%]": round(abs(max_drawdown), 2),
                "Avg. Drawdown [%]": round(abs(avg_drawdown), 2),
                "SQN": round(sqn, 2)
            },
            "trade_metrics": {
                "Total Trades": len(self.trades),
                "Win Rate [%]": round(win_rate, 2),
                "Best Trade [%]": round(best_trade, 2),
                "Worst Trade [%]": round(worst_trade, 2),
                "Avg. Trade [%]": round(avg_trade, 2),
                "Max. Trade Duration": round(max_trade_duration, 1),
                "Avg. Trade Duration": round(avg_trade_duration, 1),
                "Profit Factor": round(profit_factor, 2),
                "Expectancy [%]": round(expectancy, 2)
            },
            "time_metrics": {
                "Start": self.data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                "End": self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                "Duration": f"{(self.data.index[-1] - self.data.index[0]).days} days"
            }
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

def optimize_strategy_parameters(
    file_name: str,
    zip_path: str,
    strategy_type: str = 'breakout',
    optimize: bool = False,
    initial_cash: float = 10000,
    max_allocation_pct: float = 0.33,
    plot: bool = False,
    n_jobs: int = 5
) -> dict:
    """Run backtest with optional parameter optimization.
    
    Args:
        file_name (str): Name of the data file
        zip_path (str): Path to zip file containing data
        strategy_type (str): Type of strategy to use ('breakout' or 'mean_reversion')
        optimize (bool): Whether to run parameter optimization
        initial_cash (float): Initial capital
        max_allocation_pct (float): Maximum allocation percentage
        plot (bool): Whether to plot results
        n_jobs (int): Number of parallel jobs for optimization
    """
    data_dict = load_specific_csv_from_zip(zip_path, [file_name])
    data = data_dict.get(file_name)
    
    # Strategy parameter ranges for optimization - might remove in place of the yaml config
    param_ranges = {
        'breakout': {
            'lookback': range(10, 31, 5),
            'buffer': [0.001, 0.002, 0.003],
            'stop_loss_factor': [1.0, 1.25, 1.5],
            'take_profit_factor': [2.0, 2.5, 3.0],
            'volume_multiplier': [0.1, 0.15, 0.2],
            'position_size_pct': [0.15, 0.25, 0.33]
        },
        'mean_reversion': {
            'base_lookback': range(10, 31, 5),
            'std_dev_threshold': [1.5, 2.0, 2.5],
            'rsi_lookback': [10, 14, 20],
            'stop_loss_factor': [1.5, 2.0, 2.5],
            'take_profit_factor': [2.5, 3.0, 3.5],
            'volume_threshold': [1.2, 1.5, 1.8],
            'position_size_pct': [0.15, 0.25, 0.33]
        }
    }
    
    # Strategy class mapping - will add more in future but enforce these for now
    strategy_classes = {
        'breakout': BreakoutStrategy,
        'mean_reversion': MeanReversionStrategy
    }

    if strategy_type not in strategy_classes:
        raise ValueError(f"Invalid strategy type. Must be one of {list(strategy_classes.keys())}")
    
    SelectedStrategy = strategy_classes[strategy_type]
    
    if not optimize:
        # Use default parameters
        strategy = SelectedStrategy()
        backtest = EnhancedBacktest(
            data=data[-2016:],
            strategy=strategy,
            initial_capital=initial_cash,
            position_size_pct=max_allocation_pct
        )
        results = backtest.run()
        
        if plot:
            backtest.plot_results()
            strategy.plot_signals(results['signals'])
        
        strategy_params = {
            k: v for k, v in strategy.__dict__.items() 
            if isinstance(v, (int, float, str, bool, dict, list))
        }
            
        return {
            "file_name": file_name,
            "strategy_type": strategy_type,
            "metrics": results,
            "trades": backtest.trades,
            "parameters": strategy_params
        }
    
    else:
        selected_param_ranges = param_ranges[strategy_type]
        
        param_combinations = [dict(zip(selected_param_ranges.keys(), v)) 
                            for v in itertools.product(*selected_param_ranges.values())]

        def evaluate_params(params):
            """Evaluate a single parameter combination"""
            pos_size = params.pop('position_size_pct')
            strategy = SelectedStrategy(**params)
            backtest = EnhancedBacktest(
                data=data,
                strategy=strategy,
                initial_capital=initial_cash,
                position_size_pct=pos_size
            )
            results = backtest.run()
            params['position_size_pct'] = pos_size
            return {
                'params': params,
                'results': results,
                'sharpe': results['risk_metrics']['Sharpe Ratio']
            }

        # Run optimization in parallel - this is still v.slow and multiprocessing causes some scary warnings
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            evaluations = list(tqdm(
                executor.map(evaluate_params, param_combinations),
                total=len(param_combinations),
                desc=f"Optimizing {strategy_type} parameters"
            ))
        
        best_evaluation = max(evaluations, key=lambda x: x['sharpe'])
        best_params = best_evaluation['params']
        best_results = best_evaluation['results']
        
        if plot:
            strategy_params = {k: v for k, v in best_params.items() 
                             if k != 'position_size_pct'}
            best_strategy = SelectedStrategy(**strategy_params)
            best_backtest = EnhancedBacktest(
                data=data,
                strategy=best_strategy,
                initial_capital=initial_cash,
                position_size_pct=best_params['position_size_pct']
            )
            final_results = best_backtest.run()
            best_backtest.plot_results()
            best_strategy.plot_signals(final_results['signals'])
        
        return {
            "file_name": file_name,
            "strategy_type": strategy_type,
            "metrics": best_results,
            "best_parameters": best_params,
            "optimization_results": {
                "parameters_tested": len(param_combinations),
                "best_sharpe": best_evaluation['sharpe']
            }
        }

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # all the config is needed here, ideally not to use defaults
        required_sections = {'data_paths', 'backtest_config', 'test_configs'}
        missing_sections = required_sections - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required sections in config: {missing_sections}")
            
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")


if __name__ == "__main__":
    config = load_config()
    
    logger = setup_logger(
        name="trading_bot",
        config_path=config['logging']['config_path'],
        default_level=logging.INFO,
        log_to_console=config['logging'].get('log_to_console', False)
    )
    
    ZIP_PATH = config['data_paths']['historical_data']
    backtest_settings = config['backtest_config']
    
    results = []
    for test_config in config['test_configs']:
        result = optimize_strategy_parameters(
            file_name=test_config['file'],
            zip_path=ZIP_PATH,
            strategy_type=test_config['strategy'],
            optimize=test_config.get('optimize', False),
            initial_cash=backtest_settings.get('initial_cash', 10000),
            max_allocation_pct=backtest_settings.get('max_allocation_pct', 0.33),
            plot=test_config.get('plot', False),
            n_jobs=backtest_settings.get('n_jobs', 5)
        )
        results.append(result)
    
    # ive based these on the backtesting package metrics, need to select most relevant 
    for result in results:
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS FOR {result['file_name']}")
        print(f"Strategy: {result['strategy_type'].upper()}")
        print(f"{'='*50}")
        
        if 'parameters' in result:
            print("\nStrategy Parameters:")
            for param, value in result['parameters'].items():
                print(f"{param:.<20} {value}")
            print("\n" + "-"*50)

        metrics = result['metrics']

        print("\nRETURN METRICS:")
        for metric, value in metrics['return_metrics'].items():
            print(f"{metric:.<30} {value}")
            
        print("\nRISK METRICS:")
        for metric, value in metrics['risk_metrics'].items():
            print(f"{metric:.<30} {value}")
            
        print("\nTRADE METRICS:")
        for metric, value in metrics['trade_metrics'].items():
            print(f"{metric:.<30} {value}")
            
        print("\nTIME METRICS:")
        for metric, value in metrics['time_metrics'].items():
            print(f"{metric:.<30} {value}")