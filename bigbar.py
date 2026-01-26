#!/usr/bin/env python3
"""
Big Bar Trading Strategy - Optimized Version
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import sys
import math
import time
import warnings
import os
warnings.filterwarnings('ignore')


# Performance optimizations
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Global cache for precomputed data (minimal caching for actual bottlenecks)
_data_cache = {}
_prepared_cache = {}

def load_data(filepath: str) -> pd.DataFrame | None:
    """
    Optimized data loading with single-copy strategy.
    Eliminates redundant DataFrame copies while maintaining thread safety.
    
    Args:
        filepath: Path to CSV data file
        
    Returns:
        DataFrame with loaded data or None if loading failed
    """
    if filepath in _data_cache:
        return _data_cache[filepath]
    
    start_time = time.time()
    
    try:
        df = pd.read_csv(filepath)
        df.columns = [x.lower() for x in df.columns]
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        cols = ['Open', 'High', 'Low', 'Close']
        df[cols] = df[cols].astype(float)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Store reference, not copy - eliminates redundant copying
        _data_cache[filepath] = df
        return df
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}", file=sys.stderr)
        return None
    finally:
        # Record performance metric (keep for debugging if needed)
        load_time = time.time() - start_time
        if load_time > 1.0:  # Only print if loading takes more than 1 second
            print(f"  ⚡ Data loading: {load_time:.4f}s")

def precompute_atr_values(df, min_period=10, max_period=100):
    """
    Pre-compute all ATR values for optimization range.
    This eliminates the need for tuple conversion and cached function calls.
    
    Args:
        df: DataFrame with High, Low, Close columns
        min_period: Minimum ATR period (default: 10)
        max_period: Maximum ATR period (default: 100)
    
    Returns:
        DataFrame with ATR columns added
    """
    start_time = time.time()
    
    for period in range(min_period, max_period + 1):
        if f'ATR_{period}' not in df.columns:
            df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    
    elapsed = time.time() - start_time
    if elapsed > 1.0:  # Only print if ATR computation takes more than 1 second
        print(f"ATR pre-computation completed in {elapsed:.4f} seconds")
    return df

def precompute_week_boundaries(df):
    """Fully vectorized week boundary computation."""
    start_time = time.time()
    
    # Single isocalendar() call - GOOD (already optimized)
    isocalendar_data = df.index.isocalendar()
    week_number = isocalendar_data.week.values  # Convert to numpy
    year = isocalendar_data.year.values
    week_id = year * 100 + week_number
    
    # Vectorized approach - compute bar position in week
    df_temp = pd.DataFrame({'week_id': week_id}, index=df.index)
    bar_in_week = df_temp.groupby('week_id').cumcount().values
    week_sizes = df_temp.groupby('week_id')['week_id'].transform('size').values
    
    # Fully vectorized restriction calculation (NO LOOPS!)
    is_restricted = (bar_in_week < 6) | (bar_in_week >= (week_sizes - 6))
    
    df['is_restricted'] = is_restricted
    
    elapsed = time.time() - start_time
    if elapsed > 0.5:  # Only print if week boundary computation takes more than 0.5 seconds
        print(f"Week boundary computation completed in {elapsed:.4f} seconds")
    
    return df


# Strategy parameters
INITIAL_CASH = 100000
COMMISSION = 0.0
SPREAD = 0.0001
TRADE_ON_CLOSE = True

class BigBarAllIn(Strategy):
    """Optimized Big Bar All-In Trading Strategy"""
    atr_period = 39
    k_atr_int = 41
    uptail_max_ratio_int = 6
    previous_weight_int = 41
    n_bar_breakout = 16  # Number of previous bars to consider for breakout detection
    
    def init(self):
        """Initialize strategy state variables with memory-optimized caching"""
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._position_direction = None
        
        # Adaptive TP/SL state variables
        self._initial_tp_target = None
        self._tp_extended = True
        self._base_sl_price = None
        
        # Pre-calculate float parameters once to avoid division on every bar
        self.k_atr = self.k_atr_int / 10
        self.uptail_max_ratio = self.uptail_max_ratio_int / 10
        self.previous_weight = self.previous_weight_int / 100
        
        # Memory-optimized: Pre-convert to numpy arrays for maximum speed
        self._close_array = self.data.df['Close'].values
        self._open_array = self.data.df['Open'].values
        self._high_array = self.data.df['High'].values
        self._low_array = self.data.df['Low'].values
        
        # Cache column references
        self._atr_column = f'ATR_{self.atr_period}'
        self._is_restricted_column = 'is_restricted'
        
        # Pre-extract ATR and is_restricted columns as numpy arrays for maximum performance
        # This eliminates expensive DataFrame column lookups on every bar
        self._atr_array = self.data.df[self._atr_column].values
        self._is_restricted_array = self.data.df[self._is_restricted_column].values
        
        # Pre-calculate N-bar high and low values using numpy for maximum performance
        n = self.n_bar_breakout
        self._highest_high = np.zeros(len(self._high_array))
        self._lowest_low = np.zeros(len(self._low_array))
        
        for i in range(len(self._high_array)):
            if i < n:
                # Not enough data for N-bar range
                self._highest_high[i] = np.nan
                self._lowest_low[i] = np.nan
            else:
                # Calculate highest high and lowest low for previous N bars
                self._highest_high[i] = np.max(self._high_array[i - n:i])
                self._lowest_low[i] = np.min(self._low_array[i - n:i])
        

    def _calculate_previous_bars_optimized(self, current_index):
        """
        Memory-optimized calculation of previous bar metrics.
        Uses pre-allocated numpy arrays for maximum speed.
        """
        # Use pre-allocated numpy arrays for maximum speed (memory-for-speed optimization)
        if current_index < 4:
            return 0.0
        
        # Bounds checking to prevent index errors
        if current_index >= len(self._close_array):
            return 0.0
            
        # Direct array access - much faster than pandas indexing
        bar1 = self._close_array[current_index - 4] - self._open_array[current_index - 4]
        bar2 = self._close_array[current_index - 3] - self._open_array[current_index - 3]
        bar3 = self._close_array[current_index - 2] - self._open_array[current_index - 2]
        
        # Apply weights directly without creating intermediate arrays
        weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
        
        return weighted_sum

    def next(self):
        """Main trading logic executed on each bar"""
        # Use pre-calculated float parameters (eliminates division on every bar)
        k_atr = self.k_atr
        uptail_max_ratio = self.uptail_max_ratio
        previous_weight = self.previous_weight
        
        # Wait for sufficient data for ATR calculation
        if len(self.data.Close) < (self.atr_period + 5):
            return

        i = len(self.data.Close) - 1
        is_restricted = self._is_restricted_array[i]
        
        # Close position if in restricted period
        if self.position and is_restricted:
            exit_price = self.data.Close[-1]
            self._close_position_and_log(exit_price)
            return
        
        # Get current bar data using numpy arrays for maximum speed
        open_p = self._open_array[i]
        high_p = self._high_array[i]
        low_p = self._low_array[i]
        close_p = self._close_array[i]
        size = high_p - low_p
        body = abs(close_p - open_p)
        atr = self._atr_array[i]

        # Entry conditions (not in position and not restricted)
        if not self.position and not is_restricted:
            try:
                # Use optimized calculation for previous bars
                weighted_sum = self._calculate_previous_bars_optimized(i)
            except Exception as e:
                print(f"WARNING: Failed to calculate previous bars at index {i}: {e}")
                return

            # Dynamic threshold based on previous weight
            # If momentum matches current bar direction, it's easier (lower threshold)
            # If momentum opposes current bar direction, it's harder (higher threshold)
            
            # Use user formula: (3 * closest + 2 * middle + 1 * furthest)
            # bar_closest = i-1, bar_middle = i-2, bar_furthest = i-3
            bar_1 = self._close_array[i-1] - self._open_array[i-1]
            bar_2 = self._close_array[i-2] - self._open_array[i-2]
            bar_3 = self._close_array[i-3] - self._open_array[i-3]
            momentum = (3 * bar_1) + (2 * bar_2) + (1 * bar_3)
            
            # Long entry conditions
            cond_green = close_p > open_p
            if cond_green:
                # If momentum is positive (aligned), threshold is reduced.
                # If momentum is negative (opposed), threshold is increased.
                dynamic_k = k_atr - (momentum / atr * previous_weight if atr > 0 else 0)
                
                cond_size = (size >= dynamic_k * atr)
                cond_uptail_long = ( (high_p - close_p) < (uptail_max_ratio * size) )
                
                if cond_size and cond_uptail_long:
                    # Breakout filter for long entries: close must break above previous N bars' highest high
                    if i >= self.n_bar_breakout and not np.isnan(self._highest_high[i]):
                        if close_p > self._highest_high[i]:
                            # Calculate position size (all-in)
                            equity = self.equity
                            units = int(equity / close_p) if equity > 0 and close_p > 0 else 0
                            if units >= 1:
                                # Adaptive TP/SL Setup for Long
                                # 1. Calculate pre-entry run "real move" (consecutive same-direction bodies)
                                j = i
                                while j >= 0 and self._close_array[j] > self._open_array[j]:
                                    j -= 1
                                run_start_idx = j + 1
                                series_body_move = self._close_array[i] - self._open_array[run_start_idx]
                                
                                self._initial_tp_target = close_p + 2 * series_body_move
                                self._base_sl_price = low_p
                                self._tp_extended = True
                                
                                # Don't set TP yet - wait for first reverse bar
                                self.buy(size=units, sl=self._base_sl_price, tp=None)
                                self._in_trade = True
                                self._entry_price = close_p
                                self._entry_size = units
                                self._entry_index = i
                                self._entry_bar_high = high_p
                                self._entry_bar_low = low_p
                                self._position_direction = 'long'
                                return

            # Short entry conditions
            cond_red = close_p < open_p
            if cond_red:
                # For short, negative momentum is "aligned".
                # We want a lower threshold if momentum is negative.
                # If momentum is -100, dynamic_k = k_atr + (-100/atr * weight) -> reduced k
                dynamic_k = k_atr + (momentum / atr * previous_weight if atr > 0 else 0)
                
                cond_size = (size >= dynamic_k * atr)
                cond_downtail_short = ( (close_p - low_p) < (uptail_max_ratio * size) )
                
                if cond_size and cond_downtail_short:
                    # Breakout filter for short entries: close must break below previous N bars' lowest low
                    if i >= self.n_bar_breakout and not np.isnan(self._lowest_low[i]):
                        if close_p < self._lowest_low[i]:
                            # Calculate position size (all-in)
                            equity = self.equity
                            units = int(equity / close_p) if equity > 0 and close_p > 0 else 0
                            if units >= 1:
                                # Adaptive TP/SL Setup for Short
                                # 1. Calculate pre-entry run "real move" (consecutive same-direction bodies)
                                j = i
                                while j >= 0 and self._close_array[j] < self._open_array[j]:
                                    j -= 1
                                run_start_idx = j + 1
                                series_body_move = self._open_array[run_start_idx] - self._close_array[i]
                                
                                self._initial_tp_target = close_p - 2 * series_body_move
                                self._base_sl_price = high_p
                                self._tp_extended = True
                                
                                # Don't set TP yet - wait for first reverse bar
                                self.sell(size=units, sl=self._base_sl_price, tp=None)
                                self._in_trade = True
                                self._entry_price = close_p
                                self._entry_size = units
                                self._entry_index = i
                                self._entry_bar_high = high_p
                                self._entry_bar_low = low_p
                                self._position_direction = 'short'
                                return

        # Position management
        if self.position:
            # Use numpy arrays for maximum speed
            prev_bar_high = self._high_array[i]
            prev_bar_low = self._low_array[i]
            prev_bar_close = self._close_array[i]
            prev_bar_open = self._open_array[i]

            if self._position_direction == 'long':
                # Measured Move Take-Profit Extension
                if self._tp_extended:
                    if close_p > open_p:
                        # Continue extending TP by the bar body size (close-open)
                        self._initial_tp_target += (close_p - open_p)
                        # Don't activate TP yet - keep it None to allow continuation
                    else:
                        # First opposite-direction bar - NOW activate the TP
                        self._tp_extended = False
                        for trade in self.trades:
                            trade.tp = self._initial_tp_target

                # Half-Reversal Trailing Stop (Always Follow)
                # new_sl = base_low + (current_low - base_low) / 2
                new_sl = self._base_sl_price + (low_p - self._base_sl_price) / 2
                for trade in self.trades:
                    # Never weaken the stop
                    if new_sl > (trade.sl or -np.inf):
                        trade.sl = new_sl

            elif self._position_direction == 'short':
                # Measured Move Take-Profit Extension
                if self._tp_extended:
                    if close_p < open_p:
                        # Continue extending TP by the bar body size (open-close)
                        self._initial_tp_target -= (open_p - close_p)
                        # Don't activate TP yet - keep it None to allow continuation
                    else:
                        # First opposite-direction bar - NOW activate the TP
                        self._tp_extended = False
                        for trade in self.trades:
                            trade.tp = self._initial_tp_target

                # Half-Reversal Trailing Stop (Always Follow)
                # new_sl = base_high - (base_high - current_high) / 2
                new_sl = self._base_sl_price - (self._base_sl_price - high_p) / 2
                for trade in self.trades:
                    # Never weaken the stop
                    if new_sl < (trade.sl or np.inf):
                        trade.sl = new_sl

    def _close_position_and_log(self, exit_price):
        """Close current position"""
        if not self.position:
            return
            
        # Reset position state
        self.position.close()
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._position_direction = None


def prepare_data_pipeline(filepath, min_atr_period=20, max_atr_period=100):
    """
    Create a reusable data preparation pipeline.
    Eliminates redundant ATR calculations across multiple backtests.
    
    Args:
        filepath: Path to CSV data file
        min_atr_period: Minimum ATR period for optimization
        max_atr_period: Maximum ATR period for optimization
    
    Returns:
        Prepared DataFrame with all pre-computed values
    """
    # Check if we already have prepared data cached
    cache_key = f"{filepath}_{min_atr_period}_{max_atr_period}"
    if cache_key in _prepared_cache:
        return _prepared_cache[cache_key]
    
    # Prepare data once and cache it
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")
    
    # Pre-compute all ATR values
    df = precompute_atr_values(df, min_atr_period, max_atr_period)
    
    # Pre-compute week boundaries
    df = precompute_week_boundaries(df)
    
    # Remove rows with NaN values in any ATR column
    atr_columns = [f'ATR_{period}' for period in range(min_atr_period, max_atr_period + 1)]
    df = df.dropna(subset=atr_columns)
    
    if df.empty:
        raise SystemExit(f"Not enough data after ATR calculation for periods {min_atr_period}-{max_atr_period}")
    
    # Store in prepared cache for reuse
    _prepared_cache[cache_key] = df
    return df

def sambo_optimize_strategy_optimized(df, filepath, max_tries=5000, random_state=1):
    """
    SAMBO optimization with integer parameters for 1 decimal place precision.
    Uses pre-computed data for optimal performance and built-in heatmap support.
    """
    # Define parameter ranges for SAMBO (integer values)
    param_ranges = {
        'atr_period': [20, 100],           # Integer range for ATR period (matches precomputed range)
        'k_atr_int': [20, 42],             # Integer range representing 1.0-4.0 when divided by 10
        'uptail_max_ratio_int': [1, 6],    # Integer range representing 0.5-0.9 when divided by 10
        'previous_weight_int': [10, 60],     # Integer range representing 0.10-0.80 when divided by 100
        'n_bar_breakout': [2, 20]           # Range for breakout filter (2 to 20 bars)
    }
    
    # Define constraint function
    def constraint(params):
        """Constraint: uptail_max_ratio > 0.5 and previous_weight > 0.01"""
        return params.uptail_max_ratio_int > 5 and params.previous_weight_int > 0
    
    start_time = time.time()
    
    try:
        # Create Backtest object with pre-computed data
        bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, spread=SPREAD, trade_on_close=TRADE_ON_CLOSE)
        
        # Run SAMBO optimization with built-in heatmap support
        optimize_result, heatmap, sambo_results = bt.optimize(
            atr_period=param_ranges['atr_period'],
            k_atr_int=param_ranges['k_atr_int'],
            uptail_max_ratio_int=param_ranges['uptail_max_ratio_int'],
            previous_weight_int=param_ranges['previous_weight_int'],
            n_bar_breakout=param_ranges['n_bar_breakout'],
            constraint=constraint,
            maximize='Return [%]',
            method='sambo',
            max_tries=max_tries,
            random_state=random_state,
            return_heatmap=True,
            return_optimization=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract optimized parameters
        st = optimize_result._strategy
        best_params = {
            'atr_period': st.atr_period,
            'k_atr_int': st.k_atr_int,
            'uptail_max_ratio_int': st.uptail_max_ratio_int,
            'previous_weight_int': st.previous_weight_int,
            'n_bar_breakout': st.n_bar_breakout
        }
        
        # Save heatmap data to CSV for persistence
        if heatmap is not None:
            heatmap_df = pd.DataFrame({
                'atr_period': [idx[0] for idx in heatmap.index],
                'k_atr_int': [idx[1] for idx in heatmap.index],
                'uptail_max_ratio_int': [idx[2] for idx in heatmap.index],
                'previous_weight_int': [idx[3] for idx in heatmap.index],
                'n_bar_breakout': [idx[4] for idx in heatmap.index],
                'return_pct': heatmap.values
            })
            heatmap_df.to_csv('sambo_heatmap_results.csv', index=False)
        
        # Generate heatmap visualization using built-in function
        try:
            from backtesting.lib import plot_heatmaps
            plot_heatmaps(heatmap, agg='mean', filename='sambo_heatmap.html', open_browser=True)
        except ImportError:
            print("Warning: backtesting.lib.plot_heatmaps not available for heatmap visualization")
        except Exception as e:
            print(f"Warning: Failed to generate heatmap visualization: {e}")
        
        # Print optimized parameters (keep this essential output)
        # Note: This print is now removed to avoid duplication in main block
        
        # Create results list for compatibility
        results = [(best_params, optimize_result)]
        
        return (best_params, optimize_result), results
        
    except Exception as e:
        print(f"Error during SAMBO optimization: {e}")
        raise SystemExit(f"SAMBO optimization failed: {e}")


def run_backtest(filepath, print_result=True, atr_period=36, k_atr_int=29, uptail_max_ratio_int=6, previous_weight_int=21, n_bar_breakout=15):
    """
    Run backtest with pre-computed data.
    """
    start_time = time.time()
    
    # Load and prepare data
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    # Pre-compute ATR for the specific period
    df = precompute_atr_values(df, atr_period, atr_period)
    
    # Pre-compute week boundaries
    df = precompute_week_boundaries(df)
    
    # Remove NaN values
    df = df.dropna(subset=[f'ATR_{atr_period}'])
    if df.empty:
        raise SystemExit(f"Not enough data after ATR({atr_period}) calculation")
    
    elapsed = time.time() - start_time
    if elapsed > 1.0:  # Only print if backtest preparation takes more than 1 second
        print(f"Backtest preparation completed in {elapsed:.4f} seconds")

    # Run backtest
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, spread=SPREAD, trade_on_close=TRADE_ON_CLOSE)
    stats = bt.run(
        atr_period=atr_period,
        k_atr_int=k_atr_int,
        uptail_max_ratio_int=uptail_max_ratio_int,
        previous_weight_int=previous_weight_int,
        n_bar_breakout=n_bar_breakout
    )
    
    # Save trades to CSV
    if hasattr(stats, '_trades') and not stats._trades.empty:
        trades_df = stats._trades[['EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'Size', 'PnL']]
        trades_df.columns = ['entry_index', 'exit_index', 'entry_price', 'exit_price', 'size', 'pnl']
        trades_df['direction'] = trades_df['size'].apply(lambda x: 'long' if x > 0 else 'short')
        trades_df['size'] = trades_df['size'].abs()
        trades_df['entry_date'] = trades_df['entry_index'].apply(lambda idx: df.index[idx])
        trades_df['exit_date'] = trades_df['exit_index'].apply(lambda idx: df.index[idx])
        trades_df = trades_df.drop(['entry_index', 'exit_index'], axis=1)
        trades_df = trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'size', 'pnl', 'direction']]
        trades_df['pnl'] = trades_df['pnl'].round(2)
        trades_df.to_csv('bigbar_trades.csv', index=False)
    
    if print_result:
        print(stats)
    
    return stats, bt


def plot_strategy_with_data(df, filepath, filename='optimized_strategy_plot.html', optimized_params=None):
    """Plot strategy performance chart using already-prepared data for consistency"""
    if optimized_params is None:
        raise ValueError("Optimized parameters must be provided for plot_strategy_with_data")
    
    # Use the already-prepared dataframe to ensure consistency with optimization results
    # Remove NaN values for the specific ATR period we need
    df_filtered = df.dropna(subset=[f'ATR_{optimized_params["atr_period"]}'])
    if df_filtered.empty:
        raise SystemExit(f"Not enough data after ATR({optimized_params['atr_period']}) calculation")

    # Run backtest with optimized parameters using the same data as optimization
    bt = Backtest(df_filtered, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, spread=SPREAD, trade_on_close=TRADE_ON_CLOSE)
    stats = bt.run(**optimized_params)
    
    # Plot and save
    bt.plot(filename=filename)
    print(f"Plot {filename} with: atr_period={optimized_params['atr_period']}, k_atr={optimized_params['k_atr_int'] / 10}, uptail_max_ratio={optimized_params['uptail_max_ratio_int'] / 10}, previous_weight={optimized_params['previous_weight_int'] / 100}")


def plot_strategy(filepath, filename='optimized_strategy_plot.html', optimized_params=None):
    """Plot strategy performance chart"""
    if optimized_params is None:
        # If no optimized params provided, try to load from the most recent optimization result
        try:
            # Try to read from the optimization output or use default parameters
            optimized_params = {
                'atr_period': 36,  # Default fallback values
                'k_atr_int': 29,
                'uptail_max_ratio_int': 6,
                'previous_weight_int': 21,
                'n_bar_breakout': 15
            }
        except Exception:
            optimized_params = {
                'atr_period': 36,
                'k_atr_int': 29,
                'uptail_max_ratio_int': 6,
                'previous_weight_int': 21,
                'n_bar_breakout': 15
            }
    
    # Load and prepare data with optimized parameters
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    # Pre-compute ATR for the specific optimized period
    df = precompute_atr_values(df, optimized_params['atr_period'], optimized_params['atr_period'])
    
    # Pre-compute week boundaries
    df = precompute_week_boundaries(df)
    
    # Remove NaN values
    df = df.dropna(subset=[f'ATR_{optimized_params["atr_period"]}'])
    if df.empty:
        raise SystemExit(f"Not enough data after ATR({optimized_params['atr_period']}) calculation")

    # Run backtest with optimized parameters
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, spread=SPREAD, trade_on_close=TRADE_ON_CLOSE)
    stats = bt.run(**optimized_params)
    
    # Plot and save
    bt.plot(filename=filename)
    print(f"Plot saved as {filename}, with parameters: atr_period={optimized_params['atr_period']}, k_atr={optimized_params['k_atr_int'] / 10}, uptail_max_ratio={optimized_params['uptail_max_ratio_int'] / 10}, previous_weight={optimized_params['previous_weight_int'] / 100}")


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool, cpu_count
    
    parser = argparse.ArgumentParser(description="Optimized Big Bar Trading Strategy")
    parser.add_argument("filepath", help="Path to CSV data file", nargs='?', default='example.csv')
    parser.add_argument("--no-optimize", action="store_true", help="Skip strategy optimization")
    parser.add_argument("--no-plot", action="store_true", help="Skip strategy plotting")
    parser.add_argument("--atr-period", type=int, default=98, help="ATR period (default: 98)")
    parser.add_argument("--k-atr", type=float, default=2.4, help="ATR multiplier (default: 2.4)")
    parser.add_argument("--uptail-max-ratio", type=float, default=0.7, help="Maximum up-tail ratio (default: 0.7)")
    parser.add_argument("--previous-weight", type=float, default=0.50, help="Previous weight (default: 0.50)")
    parser.add_argument("--n-bar-breakout", type=int, default=5, help="Number of bars for breakout detection (default: 5)")
    parser.add_argument("--spread", type=float, default=0.0001, help="Bid-ask spread rate (default: 0.0001)")
    
    args = parser.parse_args()
    
    # Update global parameters with command line arguments
    SPREAD = args.spread
    
    if not args.no_optimize:
        # Load and prepare data
        df = load_data(args.filepath)
        if df is None:
            raise SystemExit("Failed to load data")

        # Pre-compute ATR values for optimization range
        df = precompute_atr_values(df, 10, 100)
        
        # Pre-compute week boundaries
        df = precompute_week_boundaries(df)
        
        # Remove rows with NaN values in any ATR column
        atr_columns = [f'ATR_{period}' for period in range(10, 101)]
        df = df.dropna(subset=atr_columns)
        
        if df.empty:
            raise SystemExit(f"Not enough data after ATR calculation for periods 10-100")

        # Use SAMBO optimization directly
        best_result, all_results = sambo_optimize_strategy_optimized(df, args.filepath)
        
        if best_result:
            params, optimize_result = best_result
            print(f"Optimized Parameters:")
            print(f"  atr_period: {params['atr_period']}")
            print(f"  k_atr: {params['k_atr_int'] / 10}")
            print(f"  uptail_max_ratio: {params['uptail_max_ratio_int'] / 10}")
            print(f"  previous_weight: {params['previous_weight_int'] / 100}")
            print(f"  n_bar_breakout: {params['n_bar_breakout']}")
            
            # Print the statistics (restored as requested)
            print(optimize_result)
        
        if not args.no_plot:
            plot_strategy_with_data(df, args.filepath, 'optimized_strategy_plot.html', params)
    else:
        # Convert float parameters to integer equivalents for the strategy
        k_atr_int = int(args.k_atr * 10)
        uptail_max_ratio_int = int(args.uptail_max_ratio * 10)
        previous_weight_int = int(args.previous_weight * 100)
        
        stats, bt = run_backtest(args.filepath, print_result=True, 
                                atr_period=args.atr_period,
                                k_atr_int=k_atr_int,
                                uptail_max_ratio_int=uptail_max_ratio_int,
                                previous_weight_int=previous_weight_int,
                                n_bar_breakout=args.n_bar_breakout)
        
        # Plot even when not optimizing
        if not args.no_plot:
            bt.plot(filename='bigbar.html')
            print(f"Plot saved as bigbar.html with parameters: atr_period={args.atr_period}, k_atr={args.k_atr}, uptail_max_ratio={args.uptail_max_ratio}, previous_weight={args.previous_weight}")
        
    print("\nAll operations completed successfully!")
