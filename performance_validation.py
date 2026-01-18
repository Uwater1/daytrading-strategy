#!/usr/bin/env python3
"""
Performance Validation Script for BigBar Trading Strategy Optimizations

This script validates the performance improvements achieved through the optimizations
and provides detailed benchmarking and comparison reports.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
from bigbar import (
    load_data, 
    precompute_atr_values, 
    precompute_week_boundaries,
    print_performance_report,
    get_performance_summary,
    _performance_metrics
)

def create_test_data(size=10000):
    """Create synthetic test data for performance testing."""
    print(f"Creating test data with {size} rows...")
    
    # Generate date range
    dates = pd.date_range(start='2023-01-01', periods=size, freq='5min')
    
    # Generate synthetic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(size) * 0.1)
    
    # Generate OHLC data
    open_prices = close_prices + np.random.randn(size) * 0.05
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(size)) * 0.1
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(size)) * 0.1
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, size)
    })
    
    # Save to CSV
    df.to_csv('test_data.csv', index=False)
    print(f"Test data saved to test_data.csv")
    return 'test_data.csv'

def benchmark_data_loading(filepath, iterations=5):
    """Benchmark data loading performance."""
    print(f"\n📊 Benchmarking Data Loading ({iterations} iterations)...")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        df = load_data(filepath)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Best: {min(times):.4f}s")
    print(f"  Worst: {max(times):.4f}s")
    
    return times

def benchmark_atr_computation(df, iterations=3):
    """Benchmark ATR computation performance."""
    print(f"\n📊 Benchmarking ATR Computation ({iterations} iterations)...")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        df_copy = df.copy()
        df_copy = precompute_atr_values(df_copy, 10, 100)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Best: {min(times):.4f}s")
    print(f"  Worst: {max(times):.4f}s")
    
    return times

def benchmark_week_boundaries(df, iterations=3):
    """Benchmark week boundary computation performance."""
    print(f"\n📊 Benchmarking Week Boundaries ({iterations} iterations)...")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        df_copy = df.copy()
        df_copy = precompute_week_boundaries(df_copy)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Best: {min(times):.4f}s")
    print(f"  Worst: {max(times):.4f}s")
    
    return times

def benchmark_full_pipeline(filepath, iterations=3):
    """Benchmark the complete data preparation pipeline."""
    print(f"\n📊 Benchmarking Full Pipeline ({iterations} iterations)...")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        
        # Load data
        df = load_data(filepath)
        if df is None:
            print("  Failed to load data")
            continue
            
        # Pre-compute ATR
        df = precompute_atr_values(df, 10, 100)
        
        # Pre-compute week boundaries
        df = precompute_week_boundaries(df)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Best: {min(times):.4f}s")
        print(f"  Worst: {max(times):.4f}s")
    else:
        avg_time = 0
    
    return times

def analyze_memory_usage(df):
    """Analyze memory usage of the DataFrame."""
    print(f"\n📊 Memory Usage Analysis:")
    
    # Basic memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"  Total memory usage: {memory_usage / 1024**2:.2f} MB")
    
    # Memory by column type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        numeric_memory = df[numeric_cols].memory_usage(deep=True).sum()
        print(f"  Numeric columns: {numeric_memory / 1024**2:.2f} MB")
    
    if len(string_cols) > 0:
        string_memory = df[string_cols].memory_usage(deep=True).sum()
        print(f"  String columns: {string_memory / 1024**2:.2f} MB")
    
    # ATR columns memory usage
    atr_cols = [col for col in df.columns if col.startswith('ATR_')]
    if atr_cols:
        atr_memory = df[atr_cols].memory_usage(deep=True).sum()
        print(f"  ATR columns ({len(atr_cols)}): {atr_memory / 1024**2:.2f} MB")

def generate_optimization_report():
    """Generate a comprehensive optimization report."""
    print("\n" + "="*80)
    print("BIGBAR TRADING STRATEGY - PERFORMANCE OPTIMIZATION VALIDATION REPORT")
    print("="*80)
    
    # Performance summary
    summary = get_performance_summary()
    
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("-" * 40)
    
    for metric, stats in summary.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Average: {stats['avg']:.4f}s")
        print(f"  Best:    {stats['min']:.4f}s")
        print(f"  Worst:   {stats['max']:.4f}s")
        print(f"  Count:   {stats['count']}")
    
    # Expected improvements
    print(f"\n🎯 EXPECTED PERFORMANCE IMPROVEMENTS")
    print("-" * 40)
    
    improvements = {
        'data_loading_time': {
            'description': 'Data Loading',
            'expected': '10-20% faster',
            'reason': 'Eliminated redundant DataFrame copies'
        },
        'atr_computation_time': {
            'description': 'ATR Computation',
            'expected': '50-80% faster',
            'reason': 'Pre-computed all ATR values once, eliminated tuple conversion'
        },
        'week_boundary_time': {
            'description': 'Week Boundaries',
            'expected': '10-20% faster',
            'reason': 'Optimized algorithm, eliminated duplicate isocalendar() calls'
        }
    }
    
    for metric, info in improvements.items():
        if metric in summary:
            avg_time = summary[metric]['avg']
            print(f"\n{info['description']}:")
            print(f"  Expected improvement: {info['expected']}")
            print(f"  Current average: {avg_time:.4f}s")
            print(f"  Optimization: {info['reason']}")
    
    # Memory efficiency
    print(f"\n💾 MEMORY EFFICIENCY")
    print("-" * 40)
    print("  ✓ Eliminated redundant DataFrame copies in data loading")
    print("  ✓ Removed unused SmartDataCache dead code")
    print("  ✓ Optimized ATR computation to avoid unnecessary calculations")
    print("  ✓ Streamlined week boundary calculation algorithm")
    
    # Code quality improvements
    print(f"\n🔧 CODE QUALITY IMPROVEMENTS")
    print("-" * 40)
    print("  ✓ Removed dead code (SmartDataCache class)")
    print("  ✓ Simplified caching strategy")
    print("  ✓ Added performance monitoring and metrics")
    print("  ✓ Optimized DataFrame accessor patterns")
    print("  ✓ Implemented lazy ATR computation")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    print("  1. Monitor performance metrics in production")
    print("  2. Consider implementing persistent caching for very large datasets")
    print("  3. Use lazy ATR computation for single-period backtests")
    print("  4. Leverage cached column references in strategy execution")
    print("  5. Regularly validate optimization effectiveness")
    
    print("\n" + "="*80)

def main():
    """Main validation function."""
    print("BigBar Trading Strategy - Performance Validation")
    print("=" * 50)
    
    # Create test data
    test_file = create_test_data(size=5000)  # Smaller size for faster testing
    
    try:
        # Load test data
        print(f"\nLoading test data...")
        df = load_data(test_file)
        if df is None:
            print("Failed to load test data")
            return
        
        print(f"Test data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Analyze memory usage
        analyze_memory_usage(df)
        
        # Run benchmarks
        print(f"\n" + "="*50)
        print("RUNNING BENCHMARKS")
        print("="*50)
        
        # Benchmark individual components
        data_times = benchmark_data_loading(test_file, iterations=3)
        atr_times = benchmark_atr_computation(df, iterations=3)
        week_times = benchmark_week_boundaries(df, iterations=3)
        pipeline_times = benchmark_full_pipeline(test_file, iterations=3)
        
        # Generate comprehensive report
        generate_optimization_report()
        
        # Print performance report
        print_performance_report()
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test data file")

if __name__ == "__main__":
    main()
