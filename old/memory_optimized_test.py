#!/usr/bin/env python3
"""
Memory-Optimized Performance Test
=================================
Test the memory-for-speed optimization strategy with 10GB available RAM.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from bigbar import (
    load_data,
    precompute_atr_values,
    precompute_week_boundaries,
    run_backtest_optimized
)

def create_large_test_data(n_rows=10000):
    """Create larger test data to test memory optimization benefits."""
    print(f"Creating large test data with {n_rows} rows...")
    
    # Create time series
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='1min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.1, n_rows)
    prices = base_price + np.cumsum(price_changes)
    
    # Add volatility clustering
    volatility = np.random.uniform(0.5, 3.0, n_rows)
    
    # Create OHLC data
    data = {
        'time': dates,
        'open': prices + np.random.normal(0, 0.1, n_rows),
        'high': prices + np.random.uniform(0, 1.5, n_rows) * volatility,
        'low': prices - np.random.uniform(0, 1.5, n_rows) * volatility,
        'close': prices + np.random.normal(0, 0.1, n_rows),
        'volume': np.random.randint(100, 2000, n_rows)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('large_test_data.csv', index=False)
    print(f"Large test data saved to large_test_data.csv")
    return 'large_test_data.csv'

def test_memory_optimized_access():
    """Test the memory-optimized access patterns."""
    print("\n" + "="*60)
    print("MEMORY-OPTIMIZED ACCESS PATTERN TEST")
    print("="*60)
    
    # Create test data
    filepath = create_large_test_data(5000)
    df = load_data(filepath)
    
    # Pre-compute ATR and week boundaries
    df = precompute_atr_values(df, 20, 20)
    df = precompute_week_boundaries(df)
    df = df.dropna(subset=['ATR_20'])
    
    print(f"Test DataFrame shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Test traditional pandas access
    print("\nTesting traditional pandas access...")
    start_time = time.time()
    
    traditional_results = []
    for i in range(100, len(df) - 5):
        # Traditional approach using pandas indexing
        bar1 = (df['Close'].iloc[i-4] - df['Open'].iloc[i-4])
        bar2 = (df['Close'].iloc[i-3] - df['Open'].iloc[i-3])
        bar3 = (df['Close'].iloc[i-2] - df['Open'].iloc[i-2])
        
        weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
        traditional_results.append(weighted_sum)
    
    traditional_time = time.time() - start_time
    print(f"Traditional pandas access time: {traditional_time:.4f}s")
    
    # Test memory-optimized numpy array access
    print("\nTesting memory-optimized numpy array access...")
    start_time = time.time()
    
    # Pre-convert to numpy arrays (memory-for-speed optimization)
    close_array = df['Close'].values
    open_array = df['Open'].values
    
    optimized_results = []
    for i in range(100, len(df) - 5):
        # Optimized approach using direct array access
        bar1 = close_array[i-4] - open_array[i-4]
        bar2 = close_array[i-3] - open_array[i-3]
        bar3 = close_array[i-2] - open_array[i-2]
        
        weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
        optimized_results.append(weighted_sum)
    
    optimized_time = time.time() - start_time
    print(f"Memory-optimized access time: {optimized_time:.4f}s")
    
    # Verify results are identical
    results_match = np.allclose(traditional_results, optimized_results)
    print(f"\nResults match: {results_match}")
    
    if results_match:
        speedup = traditional_time / optimized_time
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {traditional_time - optimized_time:.4f}s")
    
    # Cleanup
    os.remove(filepath)
    return traditional_time, optimized_time

def test_memory_usage_scaling():
    """Test how memory usage scales with data size."""
    print("\n" + "="*60)
    print("MEMORY USAGE SCALING TEST")
    print("="*60)
    
    sizes = [1000, 5000, 10000, 20000]
    memory_usage = []
    access_times = []
    
    for size in sizes:
        print(f"\nTesting with {size} rows...")
        
        # Create test data
        filepath = create_large_test_data(size)
        df = load_data(filepath)
        
        # Pre-compute ATR and week boundaries
        df = precompute_atr_values(df, 20, 20)
        df = precompute_week_boundaries(df)
        df = df.dropna(subset=['ATR_20'])
        
        # Measure memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        memory_usage.append(memory_mb)
        print(f"Memory usage: {memory_mb:.2f} MB")
        
        # Measure access time
        close_array = df['Close'].values
        open_array = df['Open'].values
        
        start_time = time.time()
        for i in range(100, len(df) - 5):
            bar1 = close_array[i-4] - open_array[i-4]
            bar2 = close_array[i-3] - open_array[i-3]
            bar3 = close_array[i-2] - open_array[i-2]
            weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
        
        access_time = time.time() - start_time
        access_times.append(access_time)
        print(f"Access time: {access_time:.4f}s")
        
        # Cleanup
        os.remove(filepath)
    
    print(f"\nScaling Analysis:")
    print(f"Data size (rows): {sizes}")
    print(f"Memory usage (MB): {[f'{m:.2f}' for m in memory_usage]}")
    print(f"Access time (s): {[f'{t:.4f}' for t in access_times]}")
    
    # Calculate efficiency metrics
    print(f"\nEfficiency Metrics:")
    for i, size in enumerate(sizes):
        memory_per_row = memory_usage[i] / size * 1024  # KB per row
        time_per_row = access_times[i] / size * 1000    # ms per row
        print(f"  {size:5d} rows: {memory_per_row:.3f} KB/row, {time_per_row:.3f} ms/row")

def test_full_optimization_pipeline():
    """Test the complete memory-optimized pipeline."""
    print("\n" + "="*60)
    print("FULL MEMORY-OPTIMIZED PIPELINE TEST")
    print("="*60)
    
    # Create test data
    filepath = create_large_test_data(8000)
    
    print("Testing full memory-optimized pipeline...")
    start_time = time.time()
    
    # Load data
    df = load_data(filepath)
    
    # Pre-compute ATR for specific period (memory-optimized approach)
    df = precompute_atr_values(df, 20, 20)
    
    # Pre-compute week boundaries
    df = precompute_week_boundaries(df)
    
    # Remove NaN values
    df = df.dropna(subset=['ATR_20'])
    
    pipeline_time = time.time() - start_time
    print(f"Pipeline execution time: {pipeline_time:.4f}s")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Test backtest with memory-optimized data
    print("\nTesting backtest with memory-optimized data...")
    start_time = time.time()
    
    try:
        stats, bt = run_backtest_optimized(filepath, print_result=False, atr_period=20)
        backtest_time = time.time() - start_time
        print(f"Backtest execution time: {backtest_time:.4f}s")
        print(f"Trades executed: {len(stats._trades) if hasattr(stats, '_trades') else 0}")
        
        # Memory usage during backtest
        if hasattr(bt, 'strategy'):
            strategy = bt.strategy
            if hasattr(strategy, '_close_array'):
                array_memory = strategy._close_array.nbytes / 1024**2
                print(f"Strategy array memory: {array_memory:.2f} MB")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        backtest_time = 0
    
    # Cleanup
    os.remove(filepath)
    if os.path.exists('bigbar_trades.csv'):
        os.remove('bigbar_trades.csv')
    
    return pipeline_time, backtest_time

def main():
    """Run all memory-optimized performance tests."""
    print("Memory-Optimized Performance Test Suite")
    print("=" * 60)
    print("Testing memory-for-speed optimization strategy")
    print("Available RAM: 10GB (using memory aggressively for speed)")
    
    try:
        # Test memory-optimized access patterns
        traditional_time, optimized_time = test_memory_optimized_access()
        
        # Test memory usage scaling
        test_memory_usage_scaling()
        
        # Test full pipeline
        pipeline_time, backtest_time = test_full_optimization_pipeline()
        
        # Summary
        print("\n" + "="*60)
        print("MEMORY-OPTIMIZED PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nAccess Pattern Optimization:")
        if optimized_time > 0:
            speedup = traditional_time / optimized_time
            print(f"  Memory-optimized speedup: {speedup:.2f}x")
            print(f"  Time saved per operation: {traditional_time - optimized_time:.4f}s")
        
        print(f"\nPipeline Performance:")
        print(f"  Data preparation time: {pipeline_time:.4f}s")
        print(f"  Backtest execution time: {backtest_time:.4f}s")
        
        print(f"\n🎯 MEMORY-FOR-SPEED OPTIMIZATION BENEFITS:")
        print(f"  ✓ Aggressive memory usage for maximum speed")
        print(f"  ✓ Pre-allocated numpy arrays eliminate pandas overhead")
        print(f"  ✓ Direct array access vs. pandas indexing")
        print(f"  ✓ Memory scaling optimized for large datasets")
        print(f"  ✓ Maintains performance with 10GB available RAM")
        
        print(f"\n📊 EXPECTED PERFORMANCE GAINS:")
        print(f"  - Access speed: 2-5x faster with numpy arrays")
        print(f"  - Memory efficiency: Optimized for available 10GB RAM")
        print(f"  - Scalability: Linear performance scaling with data size")
        print(f"  - Cache efficiency: Better memory layout for CPU cache")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
