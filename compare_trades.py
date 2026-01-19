import pandas as pd
import numpy as np

def analyze_trades(old_path, new_path):
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    
    print(f"Old Trades Count: {len(old_df)}")
    print(f"New Trades Count: {len(new_df)}")
    
    # Calculate durations (assuming entry_date and exit_date are strings that can be converted)
    old_df['entry_date'] = pd.to_datetime(old_df['entry_date'])
    old_df['exit_date'] = pd.to_datetime(old_df['exit_date'])
    old_df['duration'] = (old_df['exit_date'] - old_df['entry_date']).dt.total_seconds() / 60
    
    new_df['entry_date'] = pd.to_datetime(new_df['entry_date'])
    new_df['exit_date'] = pd.to_datetime(new_df['exit_date'])
    new_df['duration'] = (new_df['exit_date'] - new_df['entry_date']).dt.total_seconds() / 60
    
    print("\n--- Averages ---")
    print(f"Old Avg Duration: {old_df['duration'].mean():.2f} min")
    print(f"New Avg Duration: {new_df['duration'].mean():.2f} min")
    print(f"Old Avg PnL: {old_df['pnl'].mean():.2f}")
    print(f"New Avg PnL: {new_df['pnl'].mean():.2f}")
    
    print("\n--- Win Rate ---")
    print(f"Old Win Rate: {(old_df['pnl'] > 0).mean() * 100:.2f}%")
    print(f"New Win Rate: {(new_df['pnl'] > 0).mean() * 100:.2f}%")
    
    # Identify trades that exist in old but not in new or vice versa (roughly)
    # Since prices might match, we can look at entry dates
    print("\nTotal PnL Sum:")
    print(f"Old Total PnL: {old_df['pnl'].sum():.2f}")
    print(f"New Total PnL: {new_df['pnl'].sum():.2f}")

if __name__ == "__main__":
    analyze_trades('trades_old_logic.csv', 'trades_new_logic.csv')
