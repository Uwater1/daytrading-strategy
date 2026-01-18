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
        # Ensure time column handling is robust
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
            df.index = df.index.tz_convert('UTC')
        
        # Standardize column names
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
    Now includes ATR and RSI.
    """
    print("Preparing H4 indicators (with ATR & RSI)...")
    
    # 1. Resample to 4-Hour timeframe
    df_h4 = df.resample('4h', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # 2. Calculate Indicators on H4 Data
    l60 = min(60, len(df_h4) // 2)
    l144 = min(144, len(df_h4) // 2)
    
    # Trend Existed
    df_h4['EMA60'] = ta.ema(df_h4['Close'], length=l60)
    df_h4['EMA144'] = ta.ema(df_h4['Close'], length=l144)
    df_h4['EMA60_Slope'] = df_h4['EMA60'].diff()
    df_h4['EMA144_Slope'] = df_h4['EMA144'].diff()

    # New: Volatility & Momentum
    df_h4['ATR'] = ta.atr(df_h4['High'], df_h4['Low'], df_h4['Close'], length=14)
    df_h4['RSI'] = ta.rsi(df_h4['Close'], length=14)

    # 3. Market Structure
    df_h4['SwingHigh'] = df_h4['High'].rolling(window=20).max()
    df_h4['SwingLow'] = df_h4['Low'].rolling(window=20).min()

    # Select and Rename
    cols_to_keep = ['EMA60', 'EMA144', 'EMA60_Slope', 'EMA144_Slope', 
                    'SwingHigh', 'SwingLow', 'ATR', 'RSI']
    df_h4 = df_h4[cols_to_keep].copy()
    df_h4.columns = [f'H4_{c}' for c in cols_to_keep]
    
    # Merge back to original timeframe (FFILL)
    # We use reindex to map H4 values to the 5m timestamps
    df_merged = df.join(df_h4.reindex(df.index, method='ffill'))
    return df_merged.dropna()

# ========================================
# Strategy Class
# ========================================

class H4TrendHedgeStrategyV2(Strategy):
    # Optimization Parameters
    slope_thresh = 0.2      # Trend strength
    risk_pct = 0.9          # Position sizing
    
    # Dynamic Sizing Multipliers (ATR based)
    # Replaces fixed "30 points"
    # If ATR is 20 points, then 1.5 * ATR = 30 points.
    stop_atr_mult = 1.5     # Distance for first hedge (P2)
    hedge_dist_mult = 1.75  # Distance for second hedge (P3)
    hard_stop_mult = 4.0    # Distance for hard stop (P1 - 80 roughly)
    
    def init(self):
        # H4 Indicators
        self.h4_ema60 = self.I(lambda x: x, self.data.H4_EMA60)
        self.h4_slope = self.I(lambda x: x, self.data.H4_EMA60_Slope)
        self.h4_slope144 = self.I(lambda x: x, self.data.H4_EMA144_Slope)
        self.swing_high = self.I(lambda x: x, self.data.H4_SwingHigh)
        self.swing_low = self.I(lambda x: x, self.data.H4_SwingLow)
        self.atr = self.I(lambda x: x, self.data.H4_ATR)
        self.rsi = self.I(lambda x: x, self.data.H4_RSI)
        
        self.reset_state()
        
    def reset_state(self):
        self.mode = 'IDLE' 
        # IDLE: Waiting for setup
        # LONG: Primary Order 1 Active
        # HEDGED_1: Order 1 + Order 2 (Net Flat)
        # HEDGED_2: Order 1 + Order 2 + Order 3 (Net Short)
        
        self.p1 = 0.0      # Entry Price
        self.vol = 0.0     # ATR at entry time (frozen)
        
        # State Tracking
        self.entry_bar = 0
        self.hedge_start_bar = 0
        self.hedge_attempts = 0 # Count how many times we entered HEDGED_1
        
        # Scenario D: Chop lockout
        self.is_chopped_out = False 

    def next(self):
        price = self.data.Close[-1]
        current_bar = len(self.data)
        
        # ---------------------------------------------------
        # 1. ENTRY LOGIC (IDLE)
        # ---------------------------------------------------
        if self.mode == 'IDLE':
            # Reset choppy flag if enough time passed? (Optional Refinement)
            # For now, we reset state fully on close, so is_chopped_out resets per trade cycle.
            
            # Trend Check
            is_uptrend = self.h4_slope[-1] > self.slope_thresh or \
                         (abs(self.h4_slope[-1]) <= self.slope_thresh and self.h4_slope144[-1] > 0)
            
            # RSI Filter: Don't buy if Overbought (>70)
            # This prevents buying potential tops
            rsi_ok = self.rsi[-1] < 70
            
            if is_uptrend and rsi_ok:
                # 61.8% Retracement calculation
                high = self.swing_high[-1]
                low = self.swing_low[-1]
                rng = high - low
                if rng == 0: return 
                
                fib618 = high - (0.618 * rng)
                
                # Entry Trigger
                if price <= fib618:
                    self.buy(size=self.risk_pct)
                    self.p1 = price
                    self.vol = self.atr[-1] # Freeze volatility at entry
                    self.mode = 'LONG'
                    self.entry_bar = current_bar
                    self.hedge_attempts = 0
            return
            
        # ---------------------------------------------------
        # CALC DYNAMIC LEVELS
        # ---------------------------------------------------
        # Based on frozen 'self.vol' (ATR) to keep logic consistent during trade
        dist_hedge1 = self.vol * self.stop_atr_mult      # ~30 pts
        dist_hedge2 = self.vol * self.hedge_dist_mult    # ~35 pts
        dist_hard   = self.vol * self.hard_stop_mult     # ~80 pts
        dist_target = dist_hedge1                        # ~30 pts (Symmetric target)
        
        # ---------------------------------------------------
        # 2. MANAGEMENT LOGIC
        # ---------------------------------------------------
        
        # === SCENARIO D: MAX ATTEMPTS ===
        # If we have tried to hedge 3 times and failed, we assume choppy destruction.
        # We stop hedging and just hold the bag with a hard stop or reduced stop.
        if self.hedge_attempts >= 3:
            # Hard stop logic only
            if price <= (self.p1 - dist_hard):
                self.position.close()
                self.reset_state()
            # If price recovers, great.
            # No new hedges.
            return 
        
        # === SCENARIO A: LONG ===
        if self.mode == 'LONG':
            # Target Hit -> Breakout
            if price >= (self.p1 + dist_target):
                # In a real broker we'd move SL to BE. 
                # Here we just conceptually "win" the initial range warfare.
                # We could close for profit or let run. 
                # Original plan says: Move SL to BE, Cancel Hedges.
                # We stay in LONG mode but effectively "Safe".
                pass
            
            # Stop Hit -> Trigger Hedge 1
            elif price <= (self.p1 - dist_hedge1):
                self.position.close() # Flip flat
                self.mode = 'HEDGED_1'
                self.hedge_start_bar = current_bar
                self.hedge_attempts += 1
                
            # Time Exit (Scenario C - Long Stagnation)
            # "Slight Loss Stagnation": T1+6hours, price < P1.
            elif (current_bar - self.entry_bar) > 72: # 6h = 72 * 5m
                if price < self.p1:
                    self.position.close()
                    self.reset_state()

            # Trend Reversal Exit (Profit Taking)
            # If slope turns negative, the trend is over. Secure the bag.
            elif self.h4_slope[-1] < -0.05: # Slight buffer before claiming trend death
                self.position.close()
                self.reset_state()
        
        # === SCENARIO B: HEDGED 1 (FLAT) ===
        elif self.mode == 'HEDGED_1':
            # Recovery -> Price goes back up
            # Stop Loss for Hedge 1 is P1 + buffer.
            # Let's say buffer is 0.3 * ATR (~10pts)
            recovery_price = self.p1 + (0.3 * self.vol)
            
            if price > recovery_price:
                # Re-enter Long
                self.buy(size=self.risk_pct)
                self.mode = 'LONG'
                
            # Collapse -> Trigger Hedge 2 (Short)
            elif price <= (self.p1 - dist_hedge2):
                self.sell(size=self.risk_pct)
                self.mode = 'HEDGED_2'
                self.hedge_start_bar = current_bar # Reset timer for "Short" phase

            # Time Logic (Hedge Reset - Filtering Noise)
            # If we spend 2 hours in this flat state, maybe we should reset?
            # Original plan: "If Order 2 triggers, but after 2h price is still ranging... Close Order 2"
            # Here: We are flat. Closing order 2 means Buying back? 
            # If we are flat, we have NO position. "Close Order 2" implies effectively going back to LONG?
            # This is ambiguous in the text. 
            # Interpretation: If we are flat for too long, just Kill the trade entirely? 
            # Or re-enter long? Let's assume Kill for safety if stuck in limbo.
            elif (current_bar - self.hedge_start_bar) > 24: # 2 hours
                # If we are just hanging around P2 without dropping or recovering...
                # It's chop. Exit everything.
                self.reset_state()

        # === SCENARIO C: HEDGED 2 (SHORT) ===
        elif self.mode == 'HEDGED_2':
            # Time Filter: 2 Hours (24 bars)
            if (current_bar - self.hedge_start_bar) > 24:
                # Check if profitable
                if price > (self.p1 - dist_hedge2): 
                    # We are Short from P3 (P1-35). Price > P1-35. We are losing on the short.
                    # "Close Order 3 immediately" -> Logic: Go back to Flat (Hedged_1) ?
                    self.position.close()
                    self.mode = 'HEDGED_1'
                    self.hedge_start_bar = current_bar
            
            # Hard Stop (Total Disaster)
            if price <= (self.p1 - dist_hard):
                # In original plan: "Stay short". 
                # P1 trade dies. P3 trade continues.
                # Here we are net short. We just stay short.
                pass
                
            # Trailing Stop for Profit (Scenario B.4)
            # If we are deep in profit (e.g. price < P3 - 20pts)
            # Implement a simple trailing stop logic or take profit.
            # Simplified: Exit if we rally back above P3 significantly
            if price > (self.p1 - dist_hedge2 + (0.5 * self.vol)):
                # Stop out the short
                self.position.close()
                self.mode = 'HEDGED_1'

# ========================================
# Optimization & Run
# ========================================

def run_backtest_v2(filepath):
    raw_data = load_data(filepath)
    data = prepare_strategy_data(raw_data)
    
    # We use a slightly looser optimization since we have dynamic sizing now
    bt = Backtest(data, H4TrendHedgeStrategyV2, cash=100000, commission=.0001, exclusive_orders=True)
    
    print("Starting V2 Backtest (Single Run)...")
    # For V2 we trust the robust logic more than parameter fitting
    stats = bt.run()
    
    print(stats)
    print("\nTrades:")
    print(stats._trades.head())
    
    bt.plot(filename='plan_v2_result.html')
    
    # Also save text results
    with open('result_v2.txt', 'w') as f:
        f.write(str(stats))
    
    return stats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plan_v2.py {data}.csv")
        sys.exit(1)
    if len(sys.argv) > 1:
        run_backtest_v2(sys.argv[1])
