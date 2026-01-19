# daytrading-strategy
My own daytrading-strategy, idea from my father, all right reserved

## Big Bar Trading Strategy

This repository contains an optimized implementation of the Big Bar trading strategy, designed for day trading with specific entry and exit conditions based on ATR (Average True Range) and price action analysis.

### Strategy Overview

The Big Bar strategy is a momentum-based trading system that identifies significant price movements and trades in the direction of the momentum while managing risk through trailing stops and position sizing.

### Trading Logic

#### Entry Conditions

The strategy enters trades when a "big bar" is identified, which is defined as a candlestick with a size greater than a multiple of the ATR (Average True Range).

**Long Entry Conditions:**
1. **Big Bar Detection**: The candle size must be ≥ k_atr × ATR
2. **Bullish Candle**: Close price > Open price (green candle)
3. **Momentum Confirmation**: The weighted sum of previous 3 bars must be positive and exceed the previous_weight threshold
4. **Tail Management**: The upper tail (wick) must be small relative to the candle size (≤ uptail_max_ratio × candle size)

**Short Entry Conditions:**
1. **Big Bar Detection**: The candle size must be ≥ k_atr × ATR
2. **Bearish Candle**: Close price < Open price (red candle)
3. **Momentum Confirmation**: The weighted sum of previous 3 bars must be negative and exceed the previous_weight threshold
4. **Tail Management**: The lower tail (wick) must be small relative to the candle size (≤ uptail_max_ratio × candle size)

#### Position Management

**Position Sizing:**
- The strategy uses an "all-in" approach, allocating 100% of available equity to each trade
- Position size is calculated as: units = equity / entry_price

**Exit Strategy:**

**First Bar After Entry:**
- **Long positions**: Exit if the next bar is bearish OR fails to make a new high
- **Short positions**: Exit if the next bar is bullish OR fails to make a new low
- If conditions are not met, update the initial stop loss

**Trailing Stop Management:**
- **Long positions**: Trailing stop is set at the minimum of the current and previous bar's lows
- **Short positions**: Trailing stop is set at the maximum of the current and previous bar's highs
- Position is closed when price touches the trailing stop

**Stop Loss Calculation:**
- Initial stop is placed at entry bar's low (for long) or high (for short) minus/plus a buffer
- Buffer size = buffer_ratio × candle size

#### Risk Management

**Week Boundary Restrictions:**
- No new positions are opened during the first 6 bars of a trading week
- No new positions are opened during the last 6 bars of a trading week
- Existing positions are closed when entering restricted periods

**Risk Parameters:**
- **ATR Period**: 20 (default, optimized to 38)
- **k_atr**: 2.0 (default, optimized to 0.9)
- **uptail_max_ratio**: 0.7 (default, optimized to 0.8)
- **previous_weight**: 0.01 (default, optimized to 0.02)
- **buffer_ratio**: 0.01 (fixed)

### Technical Implementation

#### Performance Optimizations

1. **Data Preprocessing**: ATR values are pre-computed for optimization ranges to avoid redundant calculations
2. **Vectorized Operations**: Week boundary calculations use pandas vectorization instead of loops
3. **Memory Optimization**: Strategy uses numpy arrays for maximum performance during backtesting
4. **Caching**: Data loading and preprocessing results are cached to avoid repeated computations

#### Key Components

- **BigBarAllIn Class**: Main strategy implementation with optimized trading logic
- **SAMBO Optimization**: Uses the SAMBO algorithm for parameter optimization with built-in heatmap visualization
- **Backtesting Framework**: Built on the Backtesting.py library for comprehensive strategy testing
- **Trade Logging**: Detailed trade records saved to CSV for analysis

### Usage

#### Basic Backtest
```python
python bigbar.py example.csv
```

#### Optimization
```python
python bigbar.py example.csv --no-plot
```

#### Custom Parameters
The strategy can be run with custom parameters for ATR period, k_atr, uptail_max_ratio, previous_weight, and buffer_ratio.

### Files

- `bigbar.py`: Main strategy implementation
- `README.md`: This documentation file
- `requirements.txt`: Python dependencies
- Various CSV files: Trade logs and optimization results
- HTML files: Strategy performance plots and heatmaps

### Dependencies

- pandas
- numpy
- pandas-ta
- backtesting
- matplotlib (for plotting)

### Performance Notes

The strategy has been optimized for performance with several key improvements:
- Eliminated redundant DataFrame copies
- Used vectorized operations for week boundary calculations
- Implemented memory-optimized caching for ATR calculations
- Pre-computed all necessary values before backtesting

### Risk Disclaimer

This is a trading strategy implementation for educational and research purposes. Trading involves substantial risk and is not suitable for everyone. Past performance is not indicative of future results. Users should perform their own analysis and consider their risk tolerance before implementing any trading strategy.

## Big Bar Trading Strategy

This repository contains an optimized implementation of the Big Bar trading strategy, designed for day trading with specific entry and exit conditions based on ATR (Average True Range) and price action analysis.

### Strategy Overview

The Big Bar strategy is a momentum-based trading system that identifies significant price movements and trades in the direction of the momentum while managing risk through trailing stops and position sizing.

### Trading Logic

#### Entry Conditions

The strategy enters trades when a "big bar" is identified, which is defined as a candlestick with a size greater than a multiple of the ATR (Average True Range).

**Long Entry Conditions:**
1. **Big Bar Detection**: The candle size must be ≥ k_atr × ATR
2. **Bullish Candle**: Close price > Open price (green candle)
3. **Momentum Confirmation**: The weighted sum of previous 3 bars must be positive and exceed the previous_weight threshold
4. **Tail Management**: The upper tail (wick) must be small relative to the candle size (≤ uptail_max_ratio × candle size)

**Short Entry Conditions:**
1. **Big Bar Detection**: The candle size must be ≥ k_atr × ATR
2. **Bearish Candle**: Close price < Open price (red candle)
3. **Momentum Confirmation**: The weighted sum of previous 3 bars must be negative and exceed the previous_weight threshold
4. **Tail Management**: The lower tail (wick) must be small relative to the candle size (≤ uptail_max_ratio × candle size)

#### Position Management

**Position Sizing:**
- The strategy uses an "all-in" approach, allocating 100% of available equity to each trade
- Position size is calculated as: units = equity / entry_price

**Exit Strategy:**

**First Bar After Entry:**
- **Long positions**: Exit if the next bar is bearish OR fails to make a new high
- **Short positions**: Exit if the next bar is bullish OR fails to make a new low
- If conditions are not met, update the initial stop loss

**Trailing Stop Management:**
- **Long positions**: Trailing stop is set at the minimum of the current and previous bar's lows
- **Short positions**: Trailing stop is set at the maximum of the current and previous bar's highs
- Position is closed when price touches the trailing stop

**Stop Loss Calculation:**
- Initial stop is placed at entry bar's low (for long) or high (for short) minus/plus a buffer
- Buffer size = buffer_ratio × candle size

#### Risk Management

**Week Boundary Restrictions:**
- No new positions are opened during the first 6 bars of a trading week
- No new positions are opened during the last 6 bars of a trading week
- Existing positions are closed when entering restricted periods

**Risk Parameters:**
- **ATR Period**: 20 (default, optimized to 38)
- **k_atr**: 2.0 (default, optimized to 0.9)
- **uptail_max_ratio**: 0.7 (default, optimized to 0.8)
- **previous_weight**: 0.01 (default, optimized to 0.02)
- **buffer_ratio**: 0.01 (fixed)

### Technical Implementation

#### Performance Optimizations

1. **Data Preprocessing**: ATR values are pre-computed for optimization ranges to avoid redundant calculations
2. **Vectorized Operations**: Week boundary calculations use pandas vectorization instead of loops
3. **Memory Optimization**: Strategy uses numpy arrays for maximum performance during backtesting
4. **Caching**: Data loading and preprocessing results are cached to avoid repeated computations

#### Key Components

- **BigBarAllIn Class**: Main strategy implementation with optimized trading logic
- **SAMBO Optimization**: Uses the SAMBO algorithm for parameter optimization with built-in heatmap visualization
- **Backtesting Framework**: Built on the Backtesting.py library for comprehensive strategy testing
- **Trade Logging**: Detailed trade records saved to CSV for analysis

### Usage

#### Basic Backtest
```python
python bigbar.py example.csv
```

#### Optimization
```python
python bigbar.py example.csv --no-plot
```

#### Custom Parameters
The strategy can be run with custom parameters for ATR period, k_atr, uptail_max_ratio, previous_weight, and buffer_ratio.

### Files

- `bigbar.py`: Main strategy implementation
- `README.md`: This documentation file
- `requirements.txt`: Python dependencies
- Various CSV files: Trade logs and optimization results
- HTML files: Strategy performance plots and heatmaps

### Dependencies

- pandas
- numpy
- pandas-ta
- backtesting
- matplotlib (for plotting)

### Performance Notes

The strategy has been optimized for performance with several key improvements:
- Eliminated redundant DataFrame copies
- Used vectorized operations for week boundary calculations
- Implemented memory-optimized caching for ATR calculations
- Pre-computed all necessary values before backtesting

### Risk Disclaimer
