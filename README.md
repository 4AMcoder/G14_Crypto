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
