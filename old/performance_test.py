#!/usr/bin/env python3
"""
Performance comparison script for BigBar trading strategy optimizations.
This script compares the original, optimized, and parallel versions.
"""

import time
import pandas as pd
import numpy as np
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

def time_function(func, *args, **kwargs):
    """Time a function execution and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def load_data(filepath):
    """Load and prepare data for testing."""
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
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_data_loading():
    """Test data loading performance."""
    print("=== Data Loading Performance Test ===")
    
    # Test with example.csv
    result, duration = time_function(load_data, 'example.csv')
    if result is not None:
        print(f"Data loaded successfully in {duration:.4f} seconds")
        print(f"Data shape: {result.shape}")
        print(f"Date range: {result.index.min()} to {result.index.max()}")
    else:
        print("Failed to load data")
    
    return result

def test_atr_computation(df):
    """Test ATR computation performance."""
    print("\n=== ATR Computation Performance Test ===")
    
    from pandas_ta import atr
    
    # Test different ATR periods
    periods = [10, 14, 20, 50]
    
    for period in periods:
        result, duration = time_function(atr, df['High'], df['Low'], df['Close'], length=period)
        print(f"ATR({period}) computed in {duration:.4f} seconds")

def test_week_boundary_computation(df):
    """Test week boundary computation performance."""
    print("\n=== Week Boundary Computation Performance Test ===")
    
    def compute_week_boundaries(index):
        week_number = index.isocalendar().week
        year = index.isocalendar().year
        week_id = year * 100 + week_number
        
        week_groups = pd.Series(week_id, index=index).groupby(week_id)
        bar_in_week = week_groups.cumcount()
        
        week_total_bars = week_groups.size()
        week_total_bars_dict = week_total_bars.to_dict()
        
        is_restricted = pd.Series(False, index=index)
        for week_id_val, total_bars in week_total_bars_dict.items():
            week_mask = week_id == week_id_val
            is_restricted[week_mask & (bar_in_week < 6)] = True
            is_restricted[week_mask & (bar_in_week >= (total_bars - 6))] = True
        
        return is_restricted
    
    result, duration = time_function(compute_week_boundaries, df.index)
    print(f"Week boundaries computed in {duration:.4f} seconds")
    print(f"Restricted bars: {result.sum()} out of {len(result)} ({result.sum()/len(result)*100:.1f}%)")

def test_strategy_execution():
    """Test strategy execution performance."""
    print("\n=== Strategy Execution Performance Test ===")
    
    # Import the optimized strategy
    try:
        from bigbar_optimized import run_backtest
        
        result, duration = time_function(run_backtest, 'example.csv', print_result=False)
        print(f"Backtest completed in {duration:.4f} seconds")
        
        if hasattr(result[0], 'Total Return [%]'):
            print(f"Total Return: {result[0]['Total Return [%]']:.2f}%")
            print(f"Win Rate: {result[0]['Win Rate [%]']:.2f}%")
            print(f"Sharpe Ratio: {result[0]['Sharpe Ratio']:.2f}")
        
    except ImportError as e:
        print(f"Could not import optimized strategy: {e}")
    except Exception as e:
        print(f"Error running backtest: {e}")

def test_optimization_performance():
    """Test optimization performance."""
    print("\n=== Optimization Performance Test ===")
    
    try:
        from bigbar_optimized import optimize_strategy
        
        result, duration = time_function(optimize_strategy, 'example.csv', return_heatmap=False)
        print(f"Optimization completed in {duration:.4f} seconds")
        
        if hasattr(result, '_strategy'):
            st = result._strategy
            print(f"Optimized Parameters:")
            print(f"  atr_period: {st.atr_period}")
            print(f"  k_atr: {st.k_atr_int / 10}")
            print(f"  uptail_max_ratio: {st.uptail_max_ratio_int / 10}")
            print(f"  previous_weight: {st.previous_weight_int / 10}")
            print(f"  buffer_ratio: {st.buffer_ratio_int / 100}")
        
    except ImportError as e:
        print(f"Could not import optimization function: {e}")
    except Exception as e:
        print(f"Error running optimization: {e}")

def memory_usage_test():
    """Test memory usage."""
    print("\n=== Memory Usage Test ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage:")
    print(f"  RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"  VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB")

def main():
    """Run all performance tests."""
    print("BigBar Strategy Performance Analysis")
    print("=" * 50)
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        print("Cannot proceed without data. Exiting.")
        return
    
    # Test ATR computation
    test_atr_computation(df)
    
    # Test week boundary computation
    test_week_boundary_computation(df)
    
    # Test strategy execution
    test_strategy_execution()
    
    # Test optimization performance
    test_optimization_performance()
    
    # Test memory usage
    memory_usage_test()
    
    print("\n" + "=" * 50)
    print("Performance analysis completed!")

if __name__ == "__main__":
    main()
