import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ========================================
# Data Loading & Preprocessing
# ========================================

def load_data(filepath):
    """
    Load and preprocess OHLCV data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.index = df.index.tz_convert('UTC')
        
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
        df = df.astype(float)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def prepare_strategy_data(df):
    """
    Pre-calculates H4 indicators and merges them onto the 5m DataFrame.
    """
    print("Preparing H4 indicators...")
    
    # 1. Resample to 4-Hour timeframe
    df_h4 = df.resample('4h', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # 2. Calculate Indicators on H4 Data
    # Use standard lengths if possible, else adjust to data size
    l60 = min(60, len(df_h4) // 2)
    l144 = min(144, len(df_h4) // 2)
    
    df_h4['EMA60'] = ta.ema(df_h4['Close'], length=l60)
    df_h4['EMA144'] = ta.ema(df_h4['Close'], length=l144)
    df_h4['EMA60_Slope'] = df_h4['EMA60'].diff()
    df_h4['EMA144_Slope'] = df_h4['EMA144'].diff()

    # 3. Market Structure
    df_h4['H4_SwingHigh'] = df_h4['High'].rolling(window=20).max()
    df_h4['H4_SwingLow'] = df_h4['Low'].rolling(window=20).min()

    df_h4 = df_h4[['EMA60', 'EMA144', 'EMA60_Slope', 'EMA144_Slope', 'H4_SwingHigh', 'H4_SwingLow']]
    df_h4.columns = ['H4_EMA60', 'H4_EMA144', 'H4_EMA60_Slope', 'H4_EMA144_Slope', 'H4_SwingHigh', 'H4_SwingLow']
    
    df_merged = df.join(df_h4.reindex(df.index, method='ffill'))
    return df_merged.dropna()

# ========================================
# Strategy Class
# ========================================

class H4TrendHedgeStrategy(Strategy):
    # Optimization Parameters
    # Convert points to a multiplier of price for better scaling
    pt_mult = 0.0002        # 1 point = price * pt_mult
    slope_thresh = 0.2      # Trend strength
    risk_pct = 0.9          # Use 90% of equity to avoid margin warnings
    
    def init(self):
        self.h4_ema60 = self.I(lambda x: x, self.data.H4_EMA60)
        self.h4_slope = self.I(lambda x: x, self.data.H4_EMA60_Slope)
        self.h4_slope144 = self.I(lambda x: x, self.data.H4_EMA144_Slope)
        self.swing_high = self.I(lambda x: x, self.data.H4_SwingHigh)
        self.swing_low = self.I(lambda x: x, self.data.H4_SwingLow)
        
        self.reset_state()
        
    def reset_state(self):
        self.mode = 'IDLE' # IDLE, LONG, HEDGED_1, HEDGED_2
        self.p1 = 0
        self.entry_idx = 0
        self.order3_time = 0

    def next(self):
        price = self.data.Close[-1]
        
        # Current Point Value based on Entry or Current Price
        # This makes the "30 points" adaptive to the asset price
        p_val = price * self.pt_mult
        
        # ---------------------------------------------------
        # 1. ENTRY LOGIC
        # ---------------------------------------------------
        if self.mode == 'IDLE':
            is_uptrend = self.h4_slope[-1] > self.slope_thresh or \
                         (abs(self.h4_slope[-1]) <= self.slope_thresh and self.h4_slope144[-1] > 0)
            
            if is_uptrend:
                # 61.8% Retracement calculation
                fib618 = self.swing_high[-1] - (0.618 * (self.swing_high[-1] - self.swing_low[-1]))
                
                # Entry if price is deep in the pullback (at or below 61.8%)
                if price <= fib618:
                    self.buy(size=self.risk_pct)
                    self.p1 = price
                    self.mode = 'LONG'
                    self.entry_idx = len(self.data)
            return

        # ---------------------------------------------------
        # 2. MANAGEMENT LOGIC
        # ---------------------------------------------------
        pts30 = 30 * p_val
        pts35 = 35 * p_val
        pts80 = 80 * p_val
        
        # Scenario A: In Initial Long
        if self.mode == 'LONG':
            if price >= self.p1 + pts30:
                # Trend resumed. Logic: Cancel potential hedges (In code, we stay Long)
                pass
            elif price <= (self.p1 - pts30):
                # Trigger Hedge 1: Sell to Flatten
                self.position.close() 
                self.mode = 'HEDGED_1'
        
        # Scenario B: Hedged 1 (Flat)
        elif self.mode == 'HEDGED_1':
            # Stop Loss for Hedge 1 (Back to Long)
            if price > (self.p1 + (10 * p_val)):
                self.buy(size=self.risk_pct)
                self.mode = 'LONG'
            # Trigger Hedge 2 (Net Short)
            elif price <= (self.p1 - pts35):
                self.sell(size=self.risk_pct)
                self.mode = 'HEDGED_2'
                self.order3_time = len(self.data)

        # Scenario C: Hedged 2 (Short)
        elif self.mode == 'HEDGED_2':
            # Time Filter: 2 Hours (24 bars of 5m)
            if (len(self.data) - self.order3_time) > 24:
                if price > (self.p1 - pts35): # Not in profit
                    self.position.close() # Return to Flat
                    self.mode = 'HEDGED_1'
            
            # Hard Stop for Original Concept (P1 - 80)
            if price <= (self.p1 - pts80):
                # The "Primary" would have died. We stay short as per trend reversal logic.
                pass

        # 3. GLOBAL TIME FILTER: 6 Hours (72 bars)
        if (len(self.data) - self.entry_idx) > 72:
            self.position.close()
            self.reset_state()

# ========================================
# Optimization & Run
# ========================================

def run_backtest(filepath, optimize=False):
    raw_data = load_data(filepath)
    data = prepare_strategy_data(raw_data)
    
    bt = Backtest(data, H4TrendHedgeStrategy, cash=100000, commission=.0002, exclusive_orders=True)
    
    if optimize:
        print("Starting Optimization...")
        stats = bt.optimize(
            pt_mult=[0.0001, 0.0002, 0.0003],
            slope_thresh=[0.1, 0.2, 0.3],
            maximize='Sharpe Ratio'
        )
    else:
        stats = bt.run()
    
    print(stats)
    bt.plot()
    return stats

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backtest(sys.argv[1], optimize=False)