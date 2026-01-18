#!/usr/bin/env python3
"""
Big Bar Trading Strategy - Final Optimized Version
==================================================
High-performance implementation with pre-computed ATR values and optimized caching.

Key Optimizations:
1. Pre-compute all ATR values once (eliminates tuple conversion overhead)
2. Pre-compute week boundaries as DataFrame columns (removes expensive hashing)
3. Remove unnecessary LRU caching for single-file operations
4. Direct DataFrame operations instead of cached function calls

Performance Improvements:
- Eliminates 30,000+ element tuple creation per optimization run
- Removes expensive index hashing for week boundaries
- Simplifies caching strategy for better performance
- Pre-computes expensive calculations once
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import sys
import math
from numba import jit
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Performance optimizations
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Global cache for precomputed data (minimal caching for actual bottlenecks)
_data_cache = {}

def load_data(filepath):
    """
    Optimized data loading without unnecessary LRU caching.
    For single-file operations, direct loading is faster than caching.
    """
    if filepath in _data_cache:
        return _data_cache[filepath]
    
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
        
        # Cache the loaded data
        _data_cache[filepath] = df.copy()
        return df.copy()
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

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
    print(f"Pre-computing ATR values for periods {min_period}-{max_period}...")
    start_time = time.time()
    
    for period in range(min_period, max_period + 1):
        if f'ATR_{period}' not in df.columns:
            df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    
    elapsed = time.time() - start_time
    print(f"ATR pre-computation completed in {elapsed:.4f} seconds")
    return df

def precompute_week_boundaries(df):
    """
    Pre-compute week boundary restrictions as DataFrame columns.
    This eliminates expensive index hashing and cached function calls.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with is_restricted column added
    """
    print("Pre-computing week boundaries...")
    start_time = time.time()
    
    # Calculate week information
    week_number = df.index.isocalendar().week
    year = df.index.isocalendar().year
    week_id = year * 100 + week_number
    
    # Group by week and calculate bar positions
    week_groups = pd.Series(week_id, index=df.index).groupby(week_id)
    bar_in_week = week_groups.cumcount()
    
    # Get total bars per week
    week_total_bars = week_groups.size()
    week_total_bars_dict = week_total_bars.to_dict()
    
    # Create restricted mask
    is_restricted = pd.Series(False, index=df.index)
    for week_id_val, total_bars in week_total_bars_dict.items():
        week_mask = week_id == week_id_val
        is_restricted[week_mask & (bar_in_week < 6)] = True
        is_restricted[week_mask & (bar_in_week >= (total_bars - 6))] = True
    
    df['is_restricted'] = is_restricted
    
    elapsed = time.time() - start_time
    print(f"Week boundary computation completed in {elapsed:.4f} seconds")
    print(f"Restricted bars: {is_restricted.sum()} out of {len(is_restricted)} ({is_restricted.sum()/len(is_restricted)*100:.1f}%)")
    
    return df

@jit(nopython=True)
def calculate_weighted_sum_numba(close_values, open_values, body):
    """Numba-optimized calculation of weighted sum of previous 3 bars"""
    if len(close_values) < 4 or len(open_values) < 4 or body == 0:
        return 0.0
    
    bar1 = close_values[-4] - open_values[-4]
    bar2 = close_values[-3] - open_values[-3]
    bar3 = close_values[-2] - open_values[-2]
    
    weighted_sum = (1.0 * bar1) + (2.0 * bar2) + (3.0 * bar3)
    return weighted_sum / body

@jit(nopython=True)
def check_entry_conditions_numba(open_p, high_p, low_p, close_p, size, body, atr, 
                                k_atr, uptail_max_ratio, previous_weight, normalized_weighted_sum):
    """Numba-optimized entry condition checker"""
    cond_green = close_p > open_p
    cond_size = size >= k_atr * atr
    cond_prev3_long = normalized_weighted_sum >= previous_weight
    cond_uptail_long = (high_p - close_p) < (uptail_max_ratio * size)

    cond_red = close_p < open_p
    cond_prev3_short = normalized_weighted_sum <= -previous_weight
    cond_downtail_short = (close_p - low_p) < (uptail_max_ratio * size)

    return (cond_green, cond_size, cond_prev3_long, cond_uptail_long,
            cond_red, cond_prev3_short, cond_downtail_short)

# Strategy parameters
ATR_PERIOD = 20
K_ATR = 2.0
UPTAIL_MAX_RATIO = 0.7
PREV3_MIN_RATIO = 0.5
BUFFER_RATIO = 0.01
INITIAL_CASH = 100000
COMMISSION = 0.0
TRADE_ON_CLOSE = True

class BigBarAllIn(Strategy):
    """Optimized Big Bar All-In Trading Strategy"""
    atr_period = 20
    k_atr_int = 20
    uptail_max_ratio_int = 7
    previous_weight_int = 1
    buffer_ratio_int = 1
    
    def init(self):
        """Initialize strategy state variables"""
        self.trades_log = []
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None

    def next(self):
        """Main trading logic executed on each bar"""
        # Convert integer parameters to float values
        k_atr = self.k_atr_int / 10
        uptail_max_ratio = self.uptail_max_ratio_int / 10
        previous_weight = self.previous_weight_int / 10
        buffer_ratio = self.buffer_ratio_int / 100
        
        # Wait for sufficient data for ATR calculation
        if len(self.data.Close) < (self.atr_period + 5):
            return

        i = len(self.data.Close) - 1
        is_restricted = self.data.df['is_restricted'].iat[i]
        
        # Close position if in restricted period
        if self.position and is_restricted:
            exit_price = self.data.Close[-1]
            self._close_position_and_log(exit_price)
            return
        
        # Get current bar data
        open_p = self.data.Open[-1]
        high_p = self.data.High[-1]
        low_p = self.data.Low[-1]
        close_p = self.data.Close[-1]
        size = high_p - low_p
        body = abs(close_p - open_p)
        atr = self.data.df[f'ATR_{self.atr_period}'].iat[i]

        # Entry conditions (not in position and not restricted)
        if not self.position and not is_restricted:
            try:
                # Calculate weighted sum of previous 3 bars (bar-4, bar-3, bar-2)
                bar1 = (self.data.Close[-4] - self.data.Open[-4])
                bar2 = (self.data.Close[-3] - self.data.Open[-3])
                bar3 = (self.data.Close[-2] - self.data.Open[-2])
                
                weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
            except Exception:
                return

            normalized_weighted_sum = weighted_sum / body if body != 0 else 0

            # Long entry conditions
            cond_green = close_p > open_p
            cond_size = (size >= k_atr * atr) if (not math.isnan(atr) and atr > 0) else False
            cond_prev3_long = (normalized_weighted_sum >= previous_weight)
            cond_uptail_long = ( (high_p - close_p) < (uptail_max_ratio * size) )

            # Short entry conditions
            cond_red = close_p < open_p
            cond_prev3_short = (normalized_weighted_sum <= -previous_weight)
            cond_downtail_short = ( (close_p - low_p) < (uptail_max_ratio * size) )

            if cond_green and cond_size and cond_prev3_long and cond_uptail_long:
                # Calculate position size (all-in)
                equity = self.equity
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                self.buy(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'long'
                self._current_stop = low_p - (buffer_ratio * size)
                return

            if cond_red and cond_size and cond_prev3_short and cond_downtail_short:
                # Calculate position size (all-in)
                equity = self.equity
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                self.sell(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'short'
                self._current_stop = high_p + (buffer_ratio * size)
                return

        # Position management
        if self.position:
            self._bars_since_entry += 1

            prev_bar_high = self.data.High[-1]
            prev_bar_low = self.data.Low[-1]
            prev_bar_close = self.data.Close[-1]
            prev_bar_open = self.data.Open[-1]

            if self._position_direction == 'long':
                # Exit on first bar after entry if conditions met
                if self._bars_since_entry == 1:
                    is_red = prev_bar_close <= prev_bar_open
                    didnnot_new_high = (prev_bar_high <= self._entry_bar_high)
                    if is_red or didnnot_new_high:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        # Update stop loss
                        potential_stop = prev_bar_low - (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop > self._current_stop:
                            self._current_stop = potential_stop

                # Trailing stop after first bar
                if self._bars_since_entry >= 2:
                    try:
                        low_1 = self.data.Low[-1]
                        low_2 = self.data.Low[-2]
                        trailing_stop = min(low_1, low_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    if trailing_stop > self._current_stop:
                        self._current_stop = trailing_stop

                    if self.data.Low[-1] <= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

            elif self._position_direction == 'short':
                # Exit on first bar after entry if conditions met
                if self._bars_since_entry == 1:
                    is_green = prev_bar_close >= prev_bar_open
                    doesnnot_new_low = (prev_bar_low >= self._entry_bar_low)
                    if is_green or doesnnot_new_low:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        # Update stop loss
                        potential_stop = prev_bar_high + (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop < self._current_stop:
                            self._current_stop = potential_stop

                # Trailing stop after first bar
                if self._bars_since_entry >= 2:
                    try:
                        high_1 = self.data.High[-1]
                        high_2 = self.data.High[-2]
                        trailing_stop = max(high_1, high_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    if trailing_stop < self._current_stop:
                        self._current_stop = trailing_stop

                    if self.data.High[-1] >= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

    def _close_position_and_log(self, exit_price):
        """Close current position and log trade details"""
        if not self.position:
            return
            
        # Calculate PnL based on position direction
        if self._position_direction == 'long':
            pnl = (exit_price - self._entry_price) * self._entry_size
        else:
            pnl = (self._entry_price - exit_price) * self._entry_size
            
        # Record trade details
        trade_record = {
            'entry_index': self._entry_index,
            'exit_index': len(self.data.Close) - 1,
            'entry_price': self._entry_price,
            'exit_price': exit_price,
            'size': self._entry_size,
            'pnl': pnl,
            'direction': self._position_direction
        }
        self.trades_log.append(trade_record)
        
        # Reset position state
        self.position.close()
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None


def prepare_data_for_optimization(filepath, min_atr_period=10, max_atr_period=100):
    """
    Prepare data with all pre-computations for optimization.
    This is the key optimization - do expensive calculations once.
    
    Args:
        filepath: Path to CSV data file
        min_atr_period: Minimum ATR period for optimization
        max_atr_period: Maximum ATR period for optimization
    
    Returns:
        Prepared DataFrame with all pre-computed values
    """
    print(f"Preparing data for optimization with ATR periods {min_atr_period}-{max_atr_period}...")
    start_time = time.time()
    
    # Load data
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
    
    elapsed = time.time() - start_time
    print(f"Data preparation completed in {elapsed:.4f} seconds")
    print(f"Final data shape: {df.shape}")
    
    return df


def run_backtest_single_param_optimized(param_tuple):
    """
    Optimized version of single parameter backtest.
    Uses pre-computed data to eliminate redundant calculations.
    """
    df, params, atr_period = param_tuple
    
    # Create backtest with pre-computed data
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    
    try:
        stats = bt.run(
            atr_period=atr_period,
            k_atr_int=params['k_atr_int'],
            uptail_max_ratio_int=params['uptail_max_ratio_int'],
            previous_weight_int=params['previous_weight_int'],
            buffer_ratio_int=params['buffer_ratio_int']
        )
        
        # Include atr_period in the params dictionary
        complete_params = params.copy()
        complete_params['atr_period'] = atr_period
        
        return complete_params, stats
    except Exception as e:
        print(f"Error running backtest with params {params}: {e}")
        return None


def generate_parameter_combinations():
    """Generate all parameter combinations for optimization"""
    atr_periods = list(range(10, 101))
    k_atr_int_values = list(range(10, 41))
    uptail_ratios_int_values = list(range(5, 10))
    previous_weights_int_values = list(range(1, 9))
    buffer_ratio_int_values = [1]
    
    params_list = []
    for atr_period in atr_periods:
        for k_atr_int in k_atr_int_values:
            for uptail_ratio_int in uptail_ratios_int_values:
                for previous_weight_int in previous_weights_int_values:
                    for buffer_ratio_int in buffer_ratio_int_values:
                        params_list.append({
                            'atr_period': atr_period,
                            'k_atr_int': k_atr_int,
                            'uptail_max_ratio_int': uptail_ratio_int,
                            'previous_weight_int': previous_weight_int,
                            'buffer_ratio_int': buffer_ratio_int
                        })
    return params_list


def parallel_optimize_strategy_optimized(filepath, workers=None):
    """
    Optimized parallel strategy optimization.
    Uses pre-computed data to eliminate redundant calculations.
    """
    if workers is None:
        workers = cpu_count()
    
    print(f"Starting optimized parallel optimization with {workers} workers...")
    
    # Prepare data once with all pre-computations
    df = prepare_data_for_optimization(filepath, 10, 100)
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations()
    print(f"Total parameter combinations to test: {len(param_combinations)}")
    
    # Create parameter tuples for parallel processing
    param_tuples = [
        (df, {
            'k_atr_int': params['k_atr_int'],
            'uptail_max_ratio_int': params['uptail_max_ratio_int'],
            'previous_weight_int': params['previous_weight_int'],
            'buffer_ratio_int': params['buffer_ratio_int']
        }, params['atr_period'])
        for params in param_combinations
    ]
    
    # Run parallel backtests
    results = []
    start_time = time.time()
    
    with Pool(processes=workers) as pool:
        chunk_size = max(1, len(param_tuples) // (workers * 4))
        for i, result in enumerate(pool.imap_unordered(run_backtest_single_param_optimized, param_tuples, chunksize=chunk_size)):
            if result is not None:
                params, stats = result
                results.append((params, stats))
                
            # Print progress
            if (i + 1) % (len(param_tuples) // 10 or 1) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(param_tuples) - (i + 1)) / rate
                print(f"Progress: {i + 1}/{len(param_tuples)} | "
                      f"Elapsed: {elapsed:.2f}s | "
                      f"Rate: {rate:.1f} tests/s | "
                      f"Remaining: {remaining:.2f}s")
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    # Find best result
    best_result = None
    best_return = -float('inf')
    
    for params, stats in results:
        if stats is not None and hasattr(stats, 'get') and stats.get('Return [%]', -float('inf')) > best_return:
            best_return = stats['Return [%]']
            best_result = (params, stats)
    
    if best_result:
        params, stats = best_result
        print("\nBest Optimization Results:")
        print(stats)
        print(f"\nOptimized Parameters:")
        print(f"  atr_period: {params['atr_period']}")
        print(f"  k_atr: {params['k_atr_int'] / 10}")
        print(f"  uptail_max_ratio: {params['uptail_max_ratio_int'] / 10}")
        print(f"  previous_weight: {params['previous_weight_int'] / 10}")
        print(f"  buffer_ratio: {params['buffer_ratio_int'] / 100}")
    
    return best_result, results


def run_backtest_optimized(filepath, print_result=True, atr_period=ATR_PERIOD):
    """
    Optimized backtest with pre-computed data.
    """
    print(f"Running optimized backtest with ATR period {atr_period}...")
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
    print(f"Data preparation completed in {elapsed:.4f} seconds")

    # Run backtest
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    stats = bt.run(
        atr_period=atr_period,
        k_atr_int=20,
        uptail_max_ratio_int=7,
        previous_weight_int=1,
        buffer_ratio_int=1
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
        print("Trades saved to bigbar_trades.csv")
    else:
        print("No trades were executed in this backtest.")

    if print_result:
        print(stats)
    
    return stats, bt


def plot_strategy_optimized(filepath, filename='optimized_strategy_plot.html'):
    """Plot optimized strategy performance chart"""
    _, bt = run_backtest_optimized(filepath, print_result=False)
    bt.plot(filename=filename)
    print(f"Plot saved as {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Big Bar Trading Strategy")
    parser.add_argument("filepath", help="Path to CSV data file", nargs='?', default='example.csv')
    parser.add_argument("--no-optimize", action="store_true", help="Skip strategy optimization")
    parser.add_argument("--no-plot", action="store_true", help="Skip strategy plotting")
    parser.add_argument("--workers", type=int, help="Number of worker processes to use (default: all available cores)")
    
    args = parser.parse_args()
    
    print("BigBarAllIn Strategy - Optimized Version")
    print("=" * 50)
    print(f"Running on {args.filepath}...")
    
    if not args.no_optimize:
        print("\nRunning optimized parallel optimization...")
        best_result, all_results = parallel_optimize_strategy_optimized(args.filepath, args.workers)
        
        if best_result:
            params, optimize_result = best_result
            print(f"\nBest result found with parameters:")
            print(f"  atr_period: {params['atr_period']}")
            print(f"  k_atr: {params['k_atr_int'] / 10}")
            print(f"  uptail_max_ratio: {params['uptail_max_ratio_int'] / 10}")
            print(f"  previous_weight: {params['previous_weight_int'] / 10}")
            print(f"  buffer_ratio: {params['buffer_ratio_int'] / 100}")
        
        if not args.no_plot:
            plot_strategy_optimized(args.filepath, 'optimized_strategy_plot.html')
    else:
        print("\nRunning backtest without optimization...")
        run_backtest_optimized(args.filepath, print_result=True)
        
    print("\nAll operations completed successfully!")
