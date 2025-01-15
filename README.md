current state:
- API & websocket connections to kraken established
- historic data available via krakens quarterly data file download (local)
- breakout strategy
- mean reversion strategy
- trend following strategy
- backtesting functionality with evaluation metrics (1 version my own and the other uses backtesting package (0.3.3))
- risk management settings
- some initial parameter optimization with grid search
- some parallelisation to boost processing
- visualisations made with plotly and bokeh covering candlesticks, bollinger bands, volume , rsi etc
- inital unit tests (pytest) covering backtest, websocket, rest api
- logger


To do:
- strategy refinement - research and code improvements to all three
- code better parameter options and permutations (select sensible lookbacks, stop loss/take profit thresholds, grid search)
- double & triple check backtest logic (own and package docs https://kernc.github.io/backtesting.py/doc/backtesting/#gsc.tab=0)
- refine plots (some are mislabelled)

example plot out:
![Screenshot 2025-01-11 233749](https://github.com/user-attachments/assets/907b58f1-bc10-4eff-9fd9-f1908763f96d)
## temporarily putting documentation here

## Trade Execution Flow

## Position Management Overview

The trade execution logic is primarily handled in the `EnhancedBacktest` class, which processes signals from both the Breakout and Mean Reversion strategies in the same way. Here's a detailed breakdown of the execution flow:

### 1. Signal Processing Flow

- The backtest runs through each timestamp in the data using `for timestamp, row in tqdm(self.signals.iterrows())`
- For each timestamp:
  1. First checks stop-loss/take-profit conditions for existing positions
  2. Then processes any new buy/sell signals
  3. Updates equity curve

### 2. Position Opening Logic

When a signal is received, the system follows this decision tree:

```
1. Signal received (buy or sell)
└── Check if position can be opened (_can_open_position)
    ├── Verify against max positions limit (default: 5)
    ├── Check existing exposure
    │   ├── No opposite positions allowed (long vs short)
    │   └── Multiple positions of same direction allowed (within limits)
    └── Verify sufficient capital
        ├── Calculate position size
        ├── Add commission costs
        └── Compare against available capital
```

### 3. Position Sizing Details

Position size is calculated in `_calculate_position_size`:

```python
position_value = self.capital * self.position_size_pct
return position_value / price if self.enable_fractional else int(position_value / price)
```

Key factors:
- Default position_size_pct is 15% (0.15) of available capital
- Fractional positions are enabled by default
- Position value includes slippage and commission calculations
- For a $10,000 account:
  - Max position size would be $1,500 (15%)
  - With BTC at $50,000, position would be 0.03 BTC

### 4. Multiple Signal Handling

#### Consecutive Buy Signals
- If a buy signal occurs when already in a long position:
  - System checks if there's room for another position (max_positions limit)
  - Verifies sufficient remaining capital
  - If conditions met, opens additional position with its own stop-loss/take-profit levels
  - Each position is managed independently

#### Consecutive Sell Signals
- Similar to buy signals but for short positions
- Can have multiple short positions if capital and position limits allow
- Each short position has independent risk management

### 5. Risk Management

For each position:
```
1. Entry
   ├── Stop Loss = entry_price ∓ (ATR * stop_loss_factor)
   └── Take Profit = entry_price ∓ (ATR * take_profit_factor)
   
2. Position Monitoring
   └── For each timestamp:
       ├── Check Stop Loss breach
       │   └── Close position if breached
       └── Check Take Profit breach
           └── Close position if breached
```

### 6. Edge Cases and Important Behaviors

1. **First Sell Signal (No Prior Buy)**
   - System allows opening short positions without requiring prior long positions
   - Same position size and risk management rules apply
   - Verified through `_can_open_position` checks

2. **Capital Management**
   - Each position deducts from available capital:
     ```python
     self.capital -= (position_value + commission)
     ```
   - Capital is returned when position closes:
     ```python
     self.capital += position_value - commission
     ```

3. **Position Size Limitations**
   - Minimum: None explicitly set (but effectively limited by fractional trading setting)
   - Maximum: Limited by:
     - position_size_pct (default 15%)
     - Available capital
     - max_positions limit
     - Opposite position restrictions

### 7. Cost Considerations

Every trade includes:
1. **Slippage**: Applied to entry and exit (default 0.1%)
   ```python
   multiplier = 1 + (self.slippage_pct if side == 'long' else -self.slippage_pct)
   return price * multiplier
   ```

2. **Commission**: Applied to both entry and exit (default 0.1%)
   ```python
   return position_value * self.commission_pct
   ```

### 8. Strategy-Specific Nuances

#### Breakout Strategy
- More aggressive entry conditions
- Typically generates fewer but larger signals
- Uses Bollinger Band breaches as primary trigger
- Includes volume confirmation

#### Mean Reversion Strategy
- More frequent signals
- Smaller position sizes recommended
- Uses RSI and stochastic oscillator for confirmation
- Heavy emphasis on volume and trend strength

## Common Scenarios

1. **Full Capital Utilization**
   ```
   Initial Capital: $10,000
   Position Size: 15% ($1,500 per position)
   Max Positions: 5
   Maximum Deployment: $7,500 (75% of capital)
   ```

2. **Partial Fills**
   ```
   Available Capital: $1,000
   Position Size: 15% ($150)
   Asset Price: $50,000
   Result: Position size adjusted to fit capital
   ```

3. **Signal Rejection**
   - Insufficient capital
   - Max positions reached
   - Opposite position exists
   - Implementation shortfall (slippage/commission exceeds profitability)
