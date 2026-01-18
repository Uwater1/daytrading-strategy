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
        from bigbar_final_optimized import run_backtest
        
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

def test_parallel_optimization_performance():
    """Test parallel optimization performance."""
    print("\n=== Parallel Optimization Performance Test ===")
    
    try:
        from bigbar_final_optimized import parallel_optimize_strategy
        
        # Run with a small parameter space for quick testing
        # Modify generate_parameter_combinations temporarily for testing
        import bigbar_final_optimized
        
        original_generate = bigbar_final_optimized.generate_parameter_combinations
        
        def test_generate():
            """Generate a small parameter space for testing"""
            atr_periods = [10, 20]
            k_atr_int_values = [10, 20]
            uptail_ratios_int_values = [5, 7]
            previous_weights_int_values = [1, 3]
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
        
        bigbar_final_optimized.generate_parameter_combinations = test_generate
        
        # Call parallel_optimize_strategy and measure time
        start_time = time.time()
        result = parallel_optimize_strategy('example.csv', workers=4)
        duration = time.time() - start_time
        print(f"Parallel optimization completed in {duration:.4f} seconds")
        
        bigbar_final_optimized.generate_parameter_combinations = original_generate
        
        if result:
            best_result, all_results = result
            if best_result:
                params, stats = best_result
                print(f"Best Return: {stats['Return [%]']:.2f}%")
                print(f"Optimized Parameters:")
                print(f"  atr_period: {params['atr_period']}")
                print(f"  k_atr: {params['k_atr_int'] / 10}")
                print(f"  uptail_max_ratio: {params['uptail_max_ratio_int'] / 10}")
                print(f"  previous_weight: {params['previous_weight_int'] / 10}")
                print(f"  buffer_ratio: {params['buffer_ratio_int'] / 100}")
        
    except ImportError as e:
        print(f"Could not import parallel optimization function: {e}")
    except Exception as e:
        print(f"Error running parallel optimization: {e}")
        import traceback
        print(traceback.format_exc())

def test_sequential_optimization_performance():
    """Test sequential optimization performance."""
    print("\n=== Sequential Optimization Performance Test ===")
    
    try:
        from bigbar_final_optimized import optimize_strategy
        
        result, duration = time_function(optimize_strategy, 'example.csv', return_heatmap=False, parallel=False)
        print(f"Sequential optimization completed in {duration:.4f} seconds")
        
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

def test_cpu_utilization():
    """Test CPU utilization during parallel processing."""
    print("\n=== CPU Utilization Test ===")
    
    import psutil
    import os
    from bigbar_final_optimized import parallel_optimize_strategy
    
    # Modify parameter generation for quick test
    import bigbar_final_optimized
    
    original_generate = bigbar_final_optimized.generate_parameter_combinations
    
    def test_generate():
        atr_periods = [10, 20, 30]
        k_atr_int_values = [10, 20, 30]
        uptail_ratios_int_values = [5, 6, 7]
        previous_weights_int_values = [1, 2, 3]
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
    
    bigbar_final_optimized.generate_parameter_combinations = test_generate
    
    # Monitor CPU usage while running optimization
    cpu_usage_samples = []
    
    def monitor_cpu():
        import time
        while True:
            cpu_usage = psutil.cpu_percent(interval=0.5)
            cpu_usage_samples.append(cpu_usage)
            time.sleep(0.5)
    
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
    monitor_thread.start()
    
    print("Running optimization with CPU monitoring...")
    try:
        start_time = time.time()
        parallel_optimize_strategy('example.csv', workers=8)
        duration = time.time() - start_time
        
        print(f"Optimization duration: {duration:.2f} seconds")
        if cpu_usage_samples:
            print(f"CPU Usage:")
            print(f"  Average: {np.mean(cpu_usage_samples):.1f}%")
            print(f"  Peak: {np.max(cpu_usage_samples):.1f}%")
            print(f"  Min: {np.min(cpu_usage_samples):.1f}%")
            print(f"  Samples: {len(cpu_usage_samples)}")
            
    except Exception as e:
        print(f"Error during test: {e}")
    
    bigbar_final_optimized.generate_parameter_combinations = original_generate

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
    
    # Test sequential optimization (quick test)
    test_sequential_optimization_performance()
    
    # Test parallel optimization (quick test)
    test_parallel_optimization_performance()
    
    # Test memory usage
    memory_usage_test()
    
    print("\n" + "=" * 50)
    print("Performance analysis completed!")

if __name__ == "__main__":
    main()
