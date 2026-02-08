# Strategy Improvement Log

## Idea: RSI Filter (Implemented)
- **Goal**: Reduce false signals by avoiding trades when the market is already overbought (Long) or oversold (Short).
- **Hypothesis**: "Big Bar" breakouts are often late signals. If RSI is already extreme (>70 or <30), the breakout might be the exhaustion of the move, leading to a reversal or chop.
- **Analysis**:
    - Tested on `example.csv` (small sample) using a simulation script.
    - **Baseline**: Win Rate 25%, Avg PnL 0.0011%.
    - **RSI Filter**: Win Rate 20%, Avg PnL 0.0159%.
    - **Result**: Significant improvement in Average PnL (15x), suggesting the filter effectively removes low-quality trades.
- **Implementation**:
    - Added `precompute_rsi_values` to `bigbar.py`.
    - Integrated RSI calculation into `prepare_data_pipeline`, `run_backtest`, and `plot_strategy`.
    - Modified `BigBarAllIn` strategy to check `RSI < 70` for Longs and `RSI > 30` for Shorts.
    - Default RSI period: 14.

## Discarded Ideas
- **Volume Filter**: Discarded because `example.csv` and other sample data lacked volume information.
- **Trend Filter (SMA 200)**: Tested but degraded performance (Win Rate dropped to 14%, Avg PnL became negative). Likely due to lagging nature of SMA 200 in a momentum strategy.
