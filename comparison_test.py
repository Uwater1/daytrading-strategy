#!/usr/bin/env python3
"""
Comparison test between the original bigbar.py and optimized bigbar_final_optimized.py
"""

import time
import warnings
warnings.filterwarnings('ignore')

def time_function(func, *args, **kwargs):
    """Time a function execution and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def test_original_strategy():
    """Test the original bigbar.py strategy."""
    print("=== Testing Original bigbar.py ===")
    try:
        import bigbar
        
        # Run backtest
        backtest_result, duration = time_function(bigbar.run_backtest, 'example.csv', print_result=False)
        print(f"Backtest completed in {duration:.4f} seconds")
        
        if backtest_result and len(backtest_result) > 0:
            stats = backtest_result[0]
            if hasattr(stats, 'get') and 'Return [%]' in stats:
                print(f"Return: {stats['Return [%]']:.2f}%")
                print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
                print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
                print(f"Number of Trades: {stats['# Trades']}")
        
    except Exception as e:
        print(f"Error running original strategy: {e}")
        import traceback
        print(traceback.format_exc())

def test_optimized_strategy():
    """Test the optimized strategy."""
    print("\n=== Testing Optimized bigbar_final_optimized.py ===")
    try:
        import bigbar_final_optimized
        
        # Run backtest
        backtest_result, duration = time_function(bigbar_final_optimized.run_backtest, 'example.csv', print_result=False)
        print(f"Backtest completed in {duration:.4f} seconds")
        
        if backtest_result and len(backtest_result) > 0:
            stats = backtest_result[0]
            if hasattr(stats, 'get') and 'Return [%]' in stats:
                print(f"Return: {stats['Return [%]']:.2f}%")
                print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
                print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
                print(f"Number of Trades: {stats['# Trades']}")
        
    except Exception as e:
        print(f"Error running optimized strategy: {e}")
        import traceback
        print(traceback.format_exc())

def test_parallel_optimization():
    """Test parallel optimization."""
    print("\n=== Testing Parallel Optimization ===")
    try:
        import bigbar_final_optimized
        
        # Modify parameter space for quick testing
        import bigbar_final_optimized as optimized
        
        original_generate = optimized.generate_parameter_combinations
        
        def test_generate():
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
        
        optimized.generate_parameter_combinations = test_generate
        
        # Test with different number of workers
        for workers in [1, 4, 8]:
            print(f"\nTesting with {workers} worker(s)...")
            result, duration = time_function(optimized.parallel_optimize_strategy, 'example.csv', workers=workers)
            
            if result:
                best_result, all_results = result
                if best_result:
                    params, stats = best_result
                    print(f"  Duration: {duration:.4f} seconds")
                    print(f"  Best Return: {stats['Return [%]']:.2f}%")
                    print(f"  Optimized Parameters: atr={params['atr_period']}, k_atr={params['k_atr_int']/10}")
        
        optimized.generate_parameter_combinations = original_generate
        
    except Exception as e:
        print(f"Error running parallel optimization: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """Run all comparison tests."""
    print("BigBar Strategy Comparison Test")
    print("=" * 60)
    
    # Test original strategy
    test_original_strategy()
    
    # Test optimized strategy
    test_optimized_strategy()
    
    # Test parallel optimization
    test_parallel_optimization()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()
