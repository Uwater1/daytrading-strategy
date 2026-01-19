#!/usr/bin/env python3
"""
Test script to verify the week boundary optimization works correctly.
"""

import pandas as pd
import numpy as np
import time

def test_week_boundary_optimization():
    """Test the optimized week boundary computation."""
    print("Testing optimized week boundary computation...")
    
    # Create test data with multiple weeks
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'Open': np.random.randn(1000) + 100,
        'High': np.random.randn(1000) + 101,
        'Low': np.random.randn(1000) + 99,
        'Close': np.random.randn(1000) + 100
    }, index=dates)
    
    print(f"Test data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Test the optimized function
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
    
    print(f"\nOptimized function completed in {elapsed:.4f} seconds")
    print(f"Restricted bars: {df['is_restricted'].sum()} out of {len(df)} ({df['is_restricted'].sum()/len(df)*100:.1f}%)")
    print(f"Added column: {'is_restricted' in df.columns}")
    
    # Verify the logic is correct by checking a few weeks manually
    print("\nVerifying logic correctness:")
    week_groups = df.groupby(df.index.isocalendar().week)
    
    for week_num, week_data in list(week_groups)[:3]:  # Check first 3 weeks
        week_bars = len(week_data)
        restricted_bars = week_data['is_restricted'].sum()
        early_bars = min(6, week_bars)
        late_bars = min(6, week_bars)
        expected_restricted = early_bars + late_bars
        
        print(f"  Week {week_num}: {week_bars} bars, {restricted_bars} restricted (expected: {expected_restricted})")
        
        # Check first 6 bars are restricted
        if week_bars >= 6:
            first_6_restricted = week_data['is_restricted'].iloc[:6].all()
            print(f"    First 6 bars restricted: {first_6_restricted}")
        
        # Check last 6 bars are restricted
        if week_bars >= 6:
            last_6_restricted = week_data['is_restricted'].iloc[-6:].all()
            print(f"    Last 6 bars restricted: {last_6_restricted}")
    
    return df, elapsed

if __name__ == "__main__":
    test_week_boundary_optimization()
