import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import itertools
import yaml
from utils.logger import setup_logger, get_logger
from utils.historical_data_loader import load_specific_csv_from_zip
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy

@dataclass
class Position:
    entry_price: float
    size: float
    side: str
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
    exit_reason: str

class EnhancedBacktest:
    def __init__(
        self, 
        data: pd.DataFrame, 
        strategy: Union['BreakoutStrategy', 'MeanReversionStrategy'],
        initial_capital: float = 10000.0,
        max_allocation_pct: float = 0.15,
        max_positions: int = 5,
        enable_fractional: bool = True,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001
    ):
        self._validate_data(data)
        
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_allocation_pct = max_allocation_pct
        self.max_positions = max_positions
        self.enable_fractional = enable_fractional
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.logger = get_logger("trading_bot.backtest")
        
        self.capital = initial_capital
        self.equity = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.signals = None

    def _validate_data(self, data: pd.DataFrame) -> None:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")

    def _get_position_exposure(self) -> Dict[str, float]:
        exposure = {'long': 0.0, 'short': 0.0}
        for position in self.positions:
            exposure[position.side] += position.size
        return exposure

    def _can_open_position(self, side: str, price: float) -> bool:
        if len(self.positions) >= self.max_positions:
            return False

        exposure = self._get_position_exposure()
        
        if side == 'long' and exposure['short'] > 0:
            return False
        if side == 'short' and exposure['long'] > 0:
            return False
        
        position_size = self._calculate_position_size(price)
        position_value = position_size * price
        commission = self._calculate_commission(position_value)
        
        return position_value + commission <= self.capital

    def _calculate_position_size(self, price: float) -> float:
        position_value = self.capital * self.max_allocation_pct
        size = position_value / price
        return size if self.enable_fractional else int(size)

    def _apply_slippage(self, price: float, side: str) -> float:
        multiplier = 1 + (self.slippage_pct if side == 'long' else -self.slippage_pct)
        return price * multiplier

    def _calculate_commission(self, position_value: float) -> float:
        return position_value * self.commission_pct
    
    def _check_stop_loss_take_profit(self, row: pd.Series, timestamp: datetime) -> None:
        for position in self.positions[:]:
            if position.side == 'long':
                if row['low'] <= position.stop_loss:
                    self._close_position(position, timestamp, position.stop_loss, 'stop_loss')
                elif row['high'] >= position.take_profit:
                    self._close_position(position, timestamp, position.take_profit, 'take_profit')
            else:  # Short position
                if row['high'] >= position.stop_loss:
                    self._close_position(position, timestamp, position.stop_loss, 'stop_loss')
                elif row['low'] <= position.take_profit:
                    self._close_position(position, timestamp, position.take_profit, 'take_profit')

    def _open_position(self, row: pd.Series, side: str) -> None:
        price = self._apply_slippage(row['close'], side)
        
        if not self._can_open_position(side, price):
            return

        size = self._calculate_position_size(price)
        position_value = size * price
        commission = self._calculate_commission(position_value)
        
        if position_value + commission > self.capital:
            return

        self.capital -= (position_value + commission)
        
        position = Position(
            entry_price=price,
            size=size,
            side=side,
            entry_time=row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit']
        )
        
        self.positions.append(position)
        self.logger.info(f"Opened {side} position: Size={size:.4f}, Price={price:.2f}, Time={row.name}")

    def _close_position(self, position: Position, timestamp: datetime, price: float, reason: str) -> None:
        exit_price = self._apply_slippage(price, 'short' if position.side == 'long' else 'long')
        position_value = position.size * exit_price
        commission = self._calculate_commission(position_value)
        
        pnl = (
            position.size * (exit_price - position.entry_price)
            if position.side == 'long'
            else position.size * (position.entry_price - exit_price)
        ) - commission
            
        pnl_pct = pnl / (position.size * position.entry_price)
        
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
        self.logger.info(
            f"Closed {position.side} position: PnL=${pnl:.2f}, PnL%={pnl_pct*100:.2f}%, "
            f"Reason={reason}"
        )

    def _update_equity(self, row: pd.Series) -> None:
        open_positions_value = sum(
            pos.size * (
                row['close'] if pos.side == 'long'
                else pos.entry_price - row['close']
            )
            for pos in self.positions
        )
        self.equity = self.capital + open_positions_value
        self.equity_curve.append(self.equity)

    def run(self) -> Dict:
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
        
        final_row = self.signals.iloc[-1]
        for position in self.positions[:]:
            self._close_position(
                position, final_row.name, final_row['close'], 'end_of_period'
            )
        
        self.logger.info("Backtest completed.")
        return self._generate_results()

    def _generate_results(self) -> Dict:
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
        
        buy_hold_return = (
            (self.data['close'].iloc[-1] - self.data['close'].iloc[0]) /
            self.data['close'].iloc[0] * 100
        )
        
        equity_final = self.equity
        pnl = self.equity - self.initial_capital
        equity_peak = max(self.equity_curve)
        total_return = (pnl / self.initial_capital * 100)
        
        time_diff = (self.data.index[-1] - self.data.index[0]).total_seconds()
        ann_factor = 365 * 24 * 60 * 60 / time_diff
        
        ann_return = ((1 + total_return/100) ** ann_factor - 1) * 100
        ann_volatility = returns.std() * np.sqrt(252) * 100
        
        exposure_time = (
            sum(len(self.positions) > 0 for _ in self.data.index) /
            len(self.data) * 100
        )

        trade_returns = [
            (t.exit_price - t.entry_price) / t.entry_price * 100
            if t.side == 'long'
            else (t.entry_price - t.exit_price) / t.entry_price * 100
            for t in self.trades
        ]
        
        profitable_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = profitable_trades / len(self.trades) * 100 if self.trades else 0
        
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = ((equity_curve - peak) / peak * 100)
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            (returns.mean() * np.sqrt(252)) /
            (downside_returns.std() * np.sqrt(252))
            if len(downside_returns) > 0
            else 0
        )
        calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown != 0 else 0
        
        trade_durations = [
            (t.exit_time - t.entry_time).total_seconds()/86400
            for t in self.trades
        ]
        max_trade_duration = max(trade_durations) if trade_durations else 0
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        trade_returns_arr = np.array(trade_returns)
        sqn = (
            np.sqrt(len(trade_returns)) *
            (np.mean(trade_returns) / np.std(trade_returns))
            if len(trade_returns) > 0
            else 0
        )
        
        gains = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        profit_factor = (
            abs(sum(gains) / sum(losses))
            if losses and sum(losses) != 0
            else float('inf')
        )
        expectancy = (
            (win_rate/100 * np.mean(gains) + (1-win_rate/100) * np.mean(losses))
            if gains and losses
            else 0
        )

        return {
            "signals": self.signals,
            "return_metrics": {
                "P&L [$]": round(pnl, 2),
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
        drawdown = (equity_series - peak) / peak * 100
        drawdown.plot(ax=ax2, label='Drawdown', color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def optimize_strategy_parameters(
    data: pd.DataFrame,
    strategy_type: str,
    strategy_params: Dict,
    backtest_config: Dict,
    n_jobs: int = -1
) -> Dict:
    """
    Optimize strategy parameters using parallel processing.
    
    Args:
        data (pd.DataFrame): Historical price data
        strategy_type (str): Type of strategy ('breakout' or 'mean_reversion')
        strategy_params (Dict): Strategy parameters from config
        backtest_config (Dict): Backtest configuration settings
        n_jobs (int): Number of parallel jobs (-1 for all cores)
    """
    strategy_classes = {
        'breakout': BreakoutStrategy,
        'mean_reversion': MeanReversionStrategy
    }

    if strategy_type not in strategy_classes:
        raise ValueError(f"Invalid strategy type. Must be one of {list(strategy_classes.keys())}")
    
    SelectedStrategy = strategy_classes[strategy_type]
    
    param_ranges = strategy_params[strategy_type]
    param_combinations = [
        dict(zip(param_ranges.keys(), v)) 
        for v in itertools.product(*[param_ranges[k] for k in param_ranges.keys()])
    ]

    def evaluate_params(params):
        """Evaluate a single parameter combination"""
        pos_size = params.pop('max_allocation_pct', backtest_config['max_allocation_pct'])
        strategy = SelectedStrategy(**params)
        backtest = EnhancedBacktest(
            data=data,
            strategy=strategy,
            initial_capital=backtest_config['initial_cash'],
            max_allocation_pct=pos_size,
            commission_pct=backtest_config['commission_pct'],
            slippage_pct=backtest_config['slippage_pct'],
            max_positions=backtest_config['max_positions'],
            enable_fractional=backtest_config['enable_fractional']
        )
        results = backtest.run()
        params['max_allocation_pct'] = pos_size
        return {
            'params': params,
            'results': results,
            'sharpe': results['risk_metrics']['Sharpe Ratio'],
            'total_return': results['return_metrics']['Return [%]'],
            'max_drawdown': results['risk_metrics']['Max. Drawdown [%]']
        }

    logger = get_logger("trading_bot.optimize")
    logger.info(f"Starting optimization with {len(param_combinations)} parameter combinations")

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        evaluations = list(tqdm(
            executor.map(evaluate_params, param_combinations),
            total=len(param_combinations),
            desc=f"Optimizing {strategy_type} parameters",
            disable=True
        ))
    
    # Sort by multiple metrics
    sorted_evaluations = sorted(
        evaluations,
        key=lambda x: (x['sharpe'], x['total_return'], -x['max_drawdown']),
        reverse=True
    )
    
    best_evaluation = sorted_evaluations[0]
    
    logger.info("Optimization completed")
    logger.info(f"Best Sharpe Ratio: {best_evaluation['sharpe']:.2f}")
    logger.info(f"Best Parameters: {best_evaluation['params']}")

    return {
        "strategy_type": strategy_type,
        "best_parameters": best_evaluation['params'],
        "best_metrics": best_evaluation['results'],
        "optimization_results": {
            "parameters_tested": len(param_combinations),
            "all_evaluations": sorted_evaluations[:3]  # Return top 3 results but this takes
        }
    }

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        required_sections = {'data_paths', 'backtest_config', 'test_configs', 'strategy_params'}
        missing_sections = required_sections - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required sections in config: {missing_sections}")
            
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logger(
        name="trading_bot",
        config_path=config['logging']['config_path'],
        default_level=logging.WARNING,  # Changed to WARNING to reduce output
        log_to_console=config['logging'].get('log_to_console', False)
    )
    
    # Load data from zip file
    data_dict = load_specific_csv_from_zip(
        config['data_paths']['historical_data'],
        [test['file'] for test in config['test_configs']]
    )
    
    # Run tests based on configuration
    results = []
    for test_config in config['test_configs']:
        logger.info(f"Running backtest for {test_config['file']}")
        
        # Get data for this test
        data = data_dict[test_config['file']]
        
        if test_config.get('optimize', False):
            # Run optimization if specified
            result = optimize_strategy_parameters(
                data=data,
                strategy_type=test_config['strategy'],
                strategy_params=config['strategy_params'],
                backtest_config=config['backtest_config'],
                n_jobs=config['backtest_config'].get('n_jobs', -1)
            )
            
            if test_config.get('plot', False):
                # Create strategy with best parameters and plot
                strategy_class = (
                    BreakoutStrategy if test_config['strategy'] == 'breakout'
                    else MeanReversionStrategy
                )
                strategy = strategy_class(**result['best_parameters'])
                backtest = EnhancedBacktest(
                    data=data,
                    strategy=strategy,
                    **{k: v for k, v in config['backtest_config'].items() 
                       if k in ['initial_cash', 'max_allocation_pct', 'commission_pct', 
                              'slippage_pct', 'max_positions', 'enable_fractional']}
                )
                final_results = backtest.run()
                backtest.plot_results()
                strategy.plot_signals(final_results['signals'])
        else:
            # Run single backtest with default parameters
            strategy_class = (
                BreakoutStrategy if test_config['strategy'] == 'breakout'
                else MeanReversionStrategy
            )
            
            # Get default parameters from config
            strategy_params = config['strategy_params'][test_config['strategy']]
            default_params = {
                k: v[0] if isinstance(v, list) else v
                for k, v in strategy_params.items()
                if k != 'max_allocation_pct'
            }
            
            strategy = strategy_class(**default_params)
            
            backtest = EnhancedBacktest(
                data=data,
                strategy=strategy,
                **{k: v for k, v in config['backtest_config'].items() 
                   if k in ['initial_cash', 'max_allocation_pct', 'commission_pct', 
                          'slippage_pct', 'max_positions', 'enable_fractional']}
            )
            
            result = backtest.run()
            
            if test_config.get('plot', False):
                backtest.plot_results()
                strategy.plot_signals(result['signals'])
            
            # Add test configuration to results
            result['file_name'] = test_config['file']
            result['strategy_type'] = test_config['strategy']
        
        results.append(result)
    
    # Print results
    for result in results:
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS FOR {result['file_name']}")
        print(f"Strategy: {result['strategy_type'].upper()}")
        print(f"{'='*50}")
        
        metrics = result.get('best_metrics', result) if 'best_metrics' in result else result
        
        if 'best_parameters' in result:
            print("\nBest Parameters:")
            for param, value in result['best_parameters'].items():
                print(f"{param:.<30} {value}")
        
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