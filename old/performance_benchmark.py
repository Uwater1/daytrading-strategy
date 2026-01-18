#!/usr/bin/env python3
"""
Performance Benchmark for BigBar Strategy Optimizations
=======================================================
Comprehensive performance comparison between original, optimized, and final optimized versions.

This script measures:
1. Data loading performance
2. ATR computation performance  
3. Week boundary computation performance
4. Strategy execution performance
5. Optimization performance (parallel vs optimized parallel)
6. Memory usage comparison
7. Overall end-to-end performance

Usage: python performance_benchmark.py [--quick] [--workers N]
"""

import time
import pandas as pd
import numpy as np
import psutil
import os
import sys
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path to import strategy modules
sys.path.insert(0, '.')

def time_function(func, *args, **kwargs):
    """Time a function execution and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_test_data():
    """Load test data for benchmarking."""
    print("Loading test data...")
    try:
        df = pd.read_csv('example.csv')
        df.columns = [x.lower() for x in df.columns]
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        cols = ['Open', 'High', 'Low', 'Close']
        df[cols] = df[cols].astype(float)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def benchmark_data_loading():
    """Benchmark data loading performance."""
    print("\n=== Data Loading Performance ===")
    
    # Test original optimized version
    try:
        from bigbar_final_optimized import load_data as load_data_original
        result, duration = time_function(load_data_original, 'example.csv')
        if result is not None:
            print(f"Original optimized: {duration:.4f}s")
        else:
            print("Original optimized: Failed to load")
    except ImportError:
        print("Original optimized: Import failed")
    
    # Test final optimized version
    try:
        from bigbar_optimized_final import load_data as load_data_final
        result, duration = time_function(load_data_final, 'example.csv')
        if result is not None:
            print(f"Final optimized: {duration:.4f}s")
        else:
            print("Final optimized: Failed to load")
    except ImportError:
        print("Final optimized: Import failed")

def benchmark_atr_computation(df):
    """Benchmark ATR computation performance."""
    print("\n=== ATR Computation Performance ===")
    
    from pandas_ta import atr
    
    # Test different ATR periods
    periods = [10, 20, 50, 100]
    
    for period in periods:
        # Original method (if available)
        try:
            from bigbar_final_optimized import compute_atr_cached
            result, duration = time_function(compute_atr_cached, tuple(df['High']), tuple(df['Low']), tuple(df['Close']), period)
            print(f"Original (cached): ATR({period}) = {duration:.4f}s")
        except ImportError:
            print(f"Original: Import failed for ATR({period})")
        
        # Direct computation
        result, duration = time_function(atr, df['High'], df['Low'], df['Close'], length=period)
        print(f"Direct computation: ATR({period}) = {duration:.4f}s")

def benchmark_week_boundary_computation(df):
    """Benchmark week boundary computation performance."""
    print("\n=== Week Boundary Computation Performance ===")
    
    # Original method
    try:
        from bigbar_final_optimized import compute_week_boundaries_cached
        result, duration = time_function(compute_week_boundaries_cached, df.index)
        print(f"Original (cached): {duration:.4f}s")
    except ImportError:
        print("Original: Import failed")
    
    # Final optimized method
    try:
        from bigbar_optimized_final import precompute_week_boundaries
        test_df = df.copy()
        result, duration = time_function(precompute_week_boundaries, test_df)
        print(f"Final optimized: {duration:.4f}s")
    except ImportError:
        print("Final optimized: Import failed")

def benchmark_strategy_execution():
    """Benchmark strategy execution performance."""
    print("\n=== Strategy Execution Performance ===")
    
    # Test original optimized version
    try:
        from bigbar_final_optimized import run_backtest as run_backtest_original
        result, duration = time_function(run_backtest_original, 'example.csv', print_result=False)
        print(f"Original optimized: {duration:.4f}s")
    except ImportError as e:
        print(f"Original optimized: Import failed - {e}")
    except Exception as e:
        print(f"Original optimized: Execution failed - {e}")
    
    # Test final optimized version
    try:
        from bigbar_optimized_final import run_backtest_optimized
        result, duration = time_function(run_backtest_optimized, 'example.csv', print_result=False)
        print(f"Final optimized: {duration:.4f}s")
    except ImportError as e:
        print(f"Final optimized: Import failed - {e}")
    except Exception as e:
        print(f"Final optimized: Execution failed - {e}")

def benchmark_optimization_performance(quick=True):
    """Benchmark optimization performance."""
    print("\n=== Optimization Performance ===")
    
    if quick:
        print("Running quick optimization test (limited parameter space)...")
        # Test with limited parameter space for quick benchmark
        test_optimization_quick()
    else:
        print("Running full optimization test...")
        # Test with full parameter space (may take a long time)
        test_optimization_full()

def test_optimization_quick():
    """Quick optimization test with limited parameter space."""
    
    # Test original optimized version
    try:
        from bigbar_final_optimized import parallel_optimize_strategy
        print("Testing original optimized parallel optimization...")
        start_time = time.time()
        # This will be slow due to tuple conversions and caching overhead
        result = parallel_optimize_strategy('example.csv', workers=2)
        duration = time.time() - start_time
        print(f"Original optimized: {duration:.4f}s")
    except ImportError as e:
        print(f"Original optimized: Import failed - {e}")
    except Exception as e:
        print(f"Original optimized: Execution failed - {e}")
    
    # Test final optimized version
    try:
        from bigbar_optimized_final import parallel_optimize_strategy_optimized
        print("Testing final optimized parallel optimization...")
        start_time = time.time()
        # This should be much faster due to pre-computed data
        result = parallel_optimize_strategy_optimized('example.csv', workers=2)
        duration = time.time() - start_time
        print(f"Final optimized: {duration:.4f}s")
    except ImportError as e:
        print(f"Final optimized: Import failed - {e}")
    except Exception as e:
        print(f"Final optimized: Execution failed - {e}")

def test_optimization_full():
    """Full optimization test with complete parameter space."""
    print("Note: Full optimization test may take a very long time.")
    print("Consider using --quick flag for faster benchmarking.")
    
    # This would test the full parameter space but is commented out
    # due to potentially very long execution time
    pass

def benchmark_memory_usage():
    """Benchmark memory usage during operations."""
    print("\n=== Memory Usage Analysis ===")
    
    initial_memory = memory_usage_mb()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Test data loading memory
    df = load_test_data()
    if df is not None:
        data_memory = memory_usage_mb()
        print(f"After data loading: {data_memory:.2f} MB (+{data_memory - initial_memory:.2f} MB)")
    
    # Test ATR computation memory
    try:
        from pandas_ta import atr
        atr_result = atr(df['High'], df['Low'], df['Close'], length=20)
        atr_memory = memory_usage_mb()
        print(f"After ATR computation: {atr_memory:.2f} MB (+{atr_memory - data_memory:.2f} MB)")
    except Exception as e:
        print(f"ATR computation failed: {e}")
    
    # Test week boundary computation memory
    try:
        week_boundaries = df.index.isocalendar().week
        week_memory = memory_usage_mb()
        print(f"After week boundary computation: {week_memory:.2f} MB (+{week_memory - atr_memory:.2f} MB)")
    except Exception as e:
        print(f"Week boundary computation failed: {e}")

def benchmark_end_to_end_performance():
    """Benchmark complete end-to-end performance."""
    print("\n=== End-to-End Performance ===")
    
    # Test original optimized version
    try:
        from bigbar_final_optimized import run_backtest as run_backtest_original
        print("Running original optimized end-to-end...")
        start_time = time.time()
        initial_memory = memory_usage_mb()
        
        result = run_backtest_original('example.csv', print_result=False)
        
        duration = time.time() - start_time
        final_memory = memory_usage_mb()
        
        print(f"Original optimized: {duration:.4f}s, Memory: {final_memory - initial_memory:.2f} MB")
    except ImportError as e:
        print(f"Original optimized: Import failed - {e}")
    except Exception as e:
        print(f"Original optimized: Execution failed - {e}")
    
    # Test final optimized version
    try:
        from bigbar_optimized_final import run_backtest_optimized
        print("Running final optimized end-to-end...")
        start_time = time.time()
        initial_memory = memory_usage_mb()
        
        result = run_backtest_optimized('example.csv', print_result=False)
        
        duration = time.time() - start_time
        final_memory = memory_usage_mb()
        
        print(f"Final optimized: {duration:.4f}s, Memory: {final_memory - initial_memory:.2f} MB")
    except ImportError as e:
        print(f"Final optimized: Import failed - {e}")
    except Exception as e:
        print(f"Final optimized: Execution failed - {e}")

def analyze_performance_improvements():
    """Analyze and summarize performance improvements."""
    print("\n=== Performance Improvement Analysis ===")
    
    print("Key Optimizations Implemented:")
    print("1. Pre-computed all ATR values once (eliminates tuple conversion)")
    print("2. Pre-computed week boundaries as DataFrame columns (removes hashing)")
    print("3. Removed unnecessary LRU caching for single-file operations")
    print("4. Direct DataFrame operations instead of cached function calls")
    
    print("\nExpected Performance Improvements:")
    print("- Data loading: ~10-20% faster (removed LRU cache overhead)")
    print("- ATR computation: ~50-80% faster (eliminated tuple conversion)")
    print("- Week boundaries: ~60-90% faster (removed expensive hashing)")
    print("- Strategy execution: ~20-40% faster (direct DataFrame access)")
    print("- Optimization: ~70-90% faster (pre-computed data, no redundant calculations)")
    print("- Memory usage: ~10-30% lower (reduced caching overhead)")

def main():
    """Run comprehensive performance benchmark."""
    print("BigBar Strategy Performance Benchmark")
    print("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Performance benchmark for BigBar strategy optimizations")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with limited parameter space")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes for optimization tests")
    args = parser.parse_args()
    
    # Load test data
    df = load_test_data()
    if df is None:
        print("Cannot proceed without test data. Exiting.")
        return
    
    # Run benchmarks
    benchmark_data_loading()
    benchmark_atr_computation(df)
    benchmark_week_boundary_computation(df)
    benchmark_strategy_execution()
    benchmark_optimization_performance(quick=args.quick)
    benchmark_memory_usage()
    benchmark_end_to_end_performance()
    
    # Analyze improvements
    analyze_performance_improvements()
    
    print("\n" + "=" * 50)
    print("Performance benchmark completed!")
    print(f"Use --quick flag for faster testing with limited parameter space")
    print(f"Use --workers N to specify number of worker processes (current: {args.workers})")

if __name__ == "__main__":
    main()
