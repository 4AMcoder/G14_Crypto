import pandas as pd
from backtesting import Backtest
from new_strategies import BreakoutStrategy, MeanReversionStrategy, TrendFollowingStrategy
import zipfile
from typing import Dict, List
import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def resample_data(df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
    """Resample OHLCV data to a larger timeframe."""
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return resampled


def load_data_from_zip(zip_path: str, file_names: List[str], resample_to: str = None) -> Dict[str, pd.DataFrame]:
    """Load CSV files from a ZIP archive into DataFrames."""
    data_dict = {}
    column_names = ["timestamp", "Open", "High", "Low", "Close", "Volume", "trades"]

    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in file_names:
            if file_name in z.namelist():
                with z.open(file_name) as f:
                    df = pd.read_csv(
                        f,
                        names=column_names,
                        header=None,
                    )

                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)

                    # if df.index.tz is not None:
                    #     df.index = df.index.tz_localize(None)

                    df = df.sort_index()
                    if resample_to:
                        df = df.resample(resample_to).agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).fillna(method='ffill')
                    
                    data_dict[file_name] = df
                    print(f"Loaded {file_name} with {len(df)} records")
                    print(f"Date range: {df.index[0]} to {df.index[-1]}")
                    print(f"Index type: {type(df.index)}")
                    print(f"Index timezone: {df.index.tz}")
            else:
                print(f"File {file_name} not found in the ZIP archive.")

    return data_dict


def run_backtest(
    file_name: str,
    zip_path: str,
    strategy_class,
    optimize: bool = False,
    param_ranges: dict = None,
    max_tries: int = None,
    plot: bool = False
) -> dict:
    """Run backtest with optional parameter optimization."""
    
    data_dict = load_data_from_zip(zip_path, [file_name], resample_to='1h')
    data = data_dict.get(file_name)
    
    if data is None:
        raise ValueError(f"Failed to load data for {file_name}")
    
    bt = Backtest(
        data,
        strategy_class,
        cash=500000,
        commission=0.001,
        margin=1.0,
        trade_on_close=False,
        exclusive_orders=True
    )
    
    if not optimize:
        stats = bt.run()
        params = {k: v for k, v in strategy_class.__dict__.items() 
                 if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    else:
        # Run optimization
        try:
            stats = bt.optimize(
                maximize='Equity Final [$]',
                method='grid',  # Use grid search for more reliable results
                max_tries=max_tries,
                constraint=lambda p: True,  # Accept all parameter combinations
                **param_ranges,
                return_heatmap=False  # Avoid heatmap-related warnings
            )
            
            # Get optimized parameters from stats._strategy
            params = {}
            for name in param_ranges.keys():
                if hasattr(stats._strategy, name):
                    params[name] = getattr(stats._strategy, name)
            
            # Run final backtest with optimized parameters
            stats = bt.run(**params)
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fall back to default parameters
            stats = bt.run()
            params = {k: v for k, v in strategy_class.__dict__.items() 
                     if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    
    if plot:
        try:
            filename = f"backtest_{strategy_class.__name__}_{file_name.replace('.csv', '')}.html"
            try:
                filename = f"backtest_{strategy_class.__name__}_{file_name.split('.')[0]}.html"
                bt.plot(
                    results=stats,
                    filename=filename,
                    open_browser=True
                )
            except Exception as e:
                logger.error(f"Plot error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
    
    return {
        "file_name": file_name,
        "stats": stats,
        "parameters": params
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    ZIP_PATH = "data/raw/Kraken_OHLCVT.zip"
    files_to_test = ["ETHGBP_30.csv"]
    
    breakout_params = {
        'n_lookback': range(5, 41, 5),        # Shorter to longer timeframes: 5,10,15,...,40
        'buffer': [0.0005, 0.001, 0.002],     # More sensitive to breakouts
        'stop_loss_factor': [0.5, 0.75, 1.0], # Tighter stops
        'take_profit_factor': [2.0, 3.0, 4.0], # Higher profit targets
        'volume_multiplier': [0.05, 0.1, 0.2], # More volume sensitivity
        'rc_threshold': [0.2, 0.5, 0.8],      # RC signal threshold range
        'size_pct': [0.02, 0.05, 0.08]        # Larger position sizes
    }
    
    mean_reversion_params = {
        'n_lookback': range(5, 41, 5),        # Shorter to longer timeframes: 5,10,15,...,40
        'std_dev_threshold': [1.0, 1.5, 2.0], # More sensitive bands
        'rsi_oversold': [20, 25, 30],         # Various oversold levels
        'rsi_overbought': [70, 75, 80],       # Various overbought levels
        'stop_loss_factor': [0.5, 1.0, 1.5],  # Tighter stops
        'take_profit_factor': [2.0, 3.0, 4.0], # Higher profit targets
        'zscore_threshold': [0.5, 1.0, 1.5],   # More sensitive mean reversion
        'volume_threshold': [1.0, 1.5, 2.0],   # Volume sensitivity
        'size_pct': [0.02, 0.05, 0.08]        # Larger position sizes
    }

    results = []
    for file_name in files_to_test:
        logger.info(f"Testing {file_name}")
        
        # Test Breakout Strategy
        logger.info("Running Breakout Strategy optimization...")
        breakout_result = run_backtest(
            file_name=file_name,
            zip_path=ZIP_PATH,
            strategy_class=BreakoutStrategy,
            optimize=True,
            param_ranges=breakout_params,
            max_tries=100,
            plot=True
        )
        results.append(("Breakout", breakout_result))
        
        # Test Mean Reversion Strategy
        logger.info("Running Mean Reversion Strategy optimization...")
        mr_result = run_backtest(
            file_name=file_name,
            zip_path=ZIP_PATH,
            strategy_class=MeanReversionStrategy,
            optimize=True,
            param_ranges=mean_reversion_params,
            max_tries=100,
            plot=True
        )
        results.append(("Mean Reversion", mr_result))

        # Test Trend Following Strategy
        # logger.info("Running Trend Following Strategy optimization...")
        # mr_result = run_backtest(
        #     file_name=file_name,
        #     zip_path=ZIP_PATH,
        #     strategy_class=TrendFollowingStrategy,
        #     optimize=False,
        #     param_ranges=None,
        #     max_tries=100,
        #     plot=True
        # )
        # results.append(("Trend Following", mr_result))

    # Print results
    for strategy_name, result in results:
        print(f"\n{'='*80}")
        print(f"Results for {strategy_name} Strategy on {result['file_name']}")
        print(f"{'='*80}")
        
        print("\nOptimized Parameters:")
        for param, value in result['parameters'].items():
            print(f"  {param}: {value}")
            
        print("\nBacktest Statistics:")
        stats = result['stats']
        
        # Time metrics
        print("\nTime Metrics:")
        time_metrics = ['Start', 'End', 'Duration']
        for metric in time_metrics:
            print(f"  {metric}: {stats[metric]}")
        
        # Return metrics
        print("\nReturn Metrics:")
        return_metrics = [
            'Equity Final [$]',
            'Equity Peak [$]',
            'Return [%]',
            'Buy & Hold Return [%]',
            'Return (Ann.) [%]',
            'Volatility (Ann.) [%]',
            'Exposure Time [%]'
        ]
        for metric in return_metrics:
            try:
                value = stats[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
            except (KeyError, TypeError) as e:
                print(f"  {metric}: N/A")
        
        # Risk metrics
        print("\nRisk Metrics:")
        risk_metrics = [
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Max. Drawdown [%]',
            'Avg. Drawdown [%]',
            'Max. Drawdown Duration',
            'Avg. Drawdown Duration',
            'SQN'
        ]
        for metric in risk_metrics:
            try:
                value = stats[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
            except (KeyError, TypeError) as e:
                print(f"  {metric}: N/A")
        
        # Trade metrics
        print("\nTrade Metrics:")
        trade_metrics = [
            '# Trades',
            'Win Rate [%]',
            'Best Trade [%]',
            'Worst Trade [%]',
            'Avg. Trade [%]',
            'Max. Trade Duration',
            'Avg. Trade Duration',
            'Profit Factor',
            'Expectancy [%]'
        ]
        for metric in trade_metrics:
            try:
                value = stats[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
            except (KeyError, TypeError) as e:
                print(f"  {metric}: N/A")
        
        print("\nTrade Analysis:")
        if stats['_trades'] is not None and not stats['_trades'].empty:
            trades_df = stats['_trades']
            print(f"  Number of trades: {len(trades_df)}")
            print(f"  Average trade duration: {trades_df['Duration'].mean()}")
            print("\n  Last 5 trades:")
            print(trades_df.tail().to_string())
        else:
            print("  No trades executed")