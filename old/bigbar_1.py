# test_bigbar_allin.py
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import sys
import math
from functools import lru_cache

# -------------------------
# Caching for Data Loading
# -------------------------
@lru_cache(maxsize=20)
def load_data_cached(filepath):
    """Cached version of load_data to avoid duplicate loading"""
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
        return df.copy()
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_data(filepath):
    """Wrapper to maintain API compatibility"""
    return load_data_cached(filepath)

# -------------
# Configuration
# -------------
ATR_PERIOD = 20
K_ATR = 2.0              # "big" multiplier (you can change)
UPTAIL_MAX_RATIO = 0.7   # upTail < 0.7 * size
PREV3_MIN_RATIO = 0.5    # moving sum prev3 >= 0.5 * body
BUFFER_RATIO = 0.01      # initial stop buffer = BUFFER_RATIO * bar_size (1% of bar size)
INITIAL_CASH = 100000    # starting equity for backtest
COMMISSION = 0.0         # per your request: set to 0 for first test (no commissions)
TRADE_ON_CLOSE = True    # enter at bar close (tries to simulate entry at close)

# -------------------------
# Big Bar Strategy (All-in)
# -------------------------
class BigBarAllIn(Strategy):
    # Strategy parameters (optimizable)
    atr_period = 20
    k_atr = 2.0
    uptail_max_ratio = 0.7
    prev3_min_ratio = 0.5
    buffer_ratio = 0.01
    
    def init(self):
        # prepare internal tracking arrays / state
        self.trades_log = []   # list of dicts {entry_time, exit_time, entry_price, exit_price, size, pnl, direction}
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None  # 'long' or 'short'

    def next(self):
        # require enough lookback for ATR and prev3 checks
        if len(self.data.Close) < (self.atr_period + 5):
            return

        i = len(self.data.Close) - 1
        
        # Get current bar data without repeated float() conversions
        open_p = self.data.Open[-1]
        high_p = self.data.High[-1]
        low_p = self.data.Low[-1]
        close_p = self.data.Close[-1]
        size = high_p - low_p
        body = abs(close_p - open_p)
        atr = self.data.df[f'ATR_{self.atr_period}'].iat[i]

        # --- if currently not in a trade, check entry conditions ---
        if not self.position:
            # need at least 3 previous bars for prev3 sum
            try:
                prev3_sum = (
                    (self.data.Close[-2] - self.data.Open[-2]) +
                    (self.data.Close[-3] - self.data.Open[-3]) +
                    (self.data.Close[-4] - self.data.Open[-4])
                )
            except Exception:
                return

            # Big bar magnitude and green bar (long conditions)
            cond_green = close_p > open_p
            cond_size = (size >= self.k_atr * atr) if (not math.isnan(atr) and atr > 0) else False
            cond_prev3_long = (prev3_sum >= self.prev3_min_ratio * body)
            cond_uptail_long = ( (high_p - close_p) < (self.uptail_max_ratio * size) )

            # Big bar magnitude and red bar (short conditions)
            cond_red = close_p < open_p
            cond_prev3_short = (prev3_sum <= -self.prev3_min_ratio * body)
            cond_downtail_short = ( (close_p - low_p) < (self.uptail_max_ratio * size) )

            if cond_green and cond_size and cond_prev3_long and cond_uptail_long:
                # Enter long: all-in
                equity = self.equity  # dynamic
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                # Place entry (market at close because trade_on_close=True)
                self.buy(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'long'
                # initial stop just below the low of entry bar
                self._current_stop = low_p - (self.buffer_ratio * size)
                return

            if cond_red and cond_size and cond_prev3_short and cond_downtail_short:
                # Enter short: all-in
                equity = self.equity  # dynamic
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                # Place entry (market at close because trade_on_close=True)
                self.sell(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'short'
                # initial stop just above the high of entry bar
                self._current_stop = high_p + (self.buffer_ratio * size)
                return

        # --- if in a position, monitor exit conditions & trailing stop ---
        if self.position:
            # increment bars-since-entry (first bar after entry will see 1)
            self._bars_since_entry += 1

            # current bar's metrics
            prev_bar_high = self.data.High[-1]
            prev_bar_low = self.data.Low[-1]
            prev_bar_close = self.data.Close[-1]
            prev_bar_open = self.data.Open[-1]

            if self._position_direction == 'long':
                # Long position exit logic
                # 1) Two-bar immediate exit rule: If this is the first bar after entry (bars_since_entry == 1)
                #    and it is red OR did not make a new high (i.e., this bar high <= entry_bar_high), exit at this bar close.
                if self._bars_since_entry == 1:
                    is_red = prev_bar_close <= prev_bar_open
                    didnnot_new_high = (prev_bar_high <= self._entry_bar_high)
                    if is_red or didnnot_new_high:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        # price improved; move initial stop up if appropriate but do not lower it below current
                        potential_stop = prev_bar_low - (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop > self._current_stop:
                            self._current_stop = potential_stop

                # 2) After initial 2 bars, trailing stop = lowest low among previous 2 bars (sliding window)
                if self._bars_since_entry >= 2:
                    # compute lowest low among previous 2 bars: [-1] and [-2]
                    try:
                        low_1 = self.data.Low[-1]
                        low_2 = self.data.Low[-2]
                        trailing_stop = min(low_1, low_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    # only move stop up if it's higher than previous (never move stop down)
                    if trailing_stop > self._current_stop:
                        self._current_stop = trailing_stop

                    # if current bar low breached our stop, exit at current close
                    if self.data.Low[-1] <= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

            elif self._position_direction == 'short':
                # Short position exit logic (mirror of long)
                # 1) Two-bar immediate exit rule: If this is the first bar after entry (bars_since_entry == 1)
                #    and it is green OR did not make a new low (i.e., this bar low >= entry_bar_low), exit at this bar close.
                if self._bars_since_entry == 1:
                    is_green = prev_bar_close >= prev_bar_open
                    didnnot_new_low = (prev_bar_low >= self._entry_bar_low)
                    if is_green or didnnot_new_low:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        # price improved; move initial stop down if appropriate but do not raise it above current
                        potential_stop = prev_bar_high + (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop < self._current_stop:
                            self._current_stop = potential_stop

                # 2) After initial 2 bars, trailing stop = highest high among previous 2 bars (sliding window)
                if self._bars_since_entry >= 2:
                    # compute highest high among previous 2 bars: [-1] and [-2]
                    try:
                        high_1 = self.data.High[-1]
                        high_2 = self.data.High[-2]
                        trailing_stop = max(high_1, high_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    # only move stop down if it's lower than previous (never move stop up)
                    if trailing_stop < self._current_stop:
                        self._current_stop = trailing_stop

                    # if current bar high breached our stop, exit at current close
                    if self.data.High[-1] >= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

    def _close_position_and_log(self, exit_price):
        # Close the position and log the trade result (we assume single position only)
        if not self.position:
            return
        # compute pnl in dollars for logging (different for long and short)
        if self._position_direction == 'long':
            pnl = (exit_price - self._entry_price) * self._entry_size
        else:  # short
            pnl = (self._entry_price - exit_price) * self._entry_size
            
        trade_record = {
            'entry_index': self._entry_index,
            'exit_index': len(self.data.Close) - 1,
            'entry_price': self._entry_price,
            'exit_price': exit_price,
            'size': self._entry_size,
            'pnl': pnl,
            'direction': self._position_direction
        }
        self.trades_log.append(trade_record)
        # execute close
        self.position.close()
        # reset trade state
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None


# -------------------
# Helper & main
# -------------------
def run_backtest(filepath, print_result=True, atr_period=ATR_PERIOD):
    # load and prepare data
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    # compute ATR and attach as column (pandas_ta usage)
    df[f'ATR_{atr_period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    # drop NaNs produced by ATR warm-up
    df.dropna(inplace=True)

    # run backtest: trade on close to attempt entry at close of signal bar
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    # Run backtest and get trades from results directly
    stats = bt.run(atr_period=atr_period)
    
    # Extract trades from the backtest results
    if hasattr(stats, '_trades') and not stats._trades.empty:
        # The backtesting library uses negative size to indicate short positions
        trades_df = stats._trades[['EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'Size', 'PnL']]
        trades_df.columns = ['entry_index', 'exit_index', 'entry_price', 'exit_price', 'size', 'pnl']
        # Determine direction from size (positive = long, negative = short)
        trades_df['direction'] = trades_df['size'].apply(lambda x: 'long' if x > 0 else 'short')
        # Ensure size is always positive for display
        trades_df['size'] = trades_df['size'].abs()
        
        trades_df.to_csv('bigbar_trades.csv', index=False)
    else:
        print("No trades were executed in this backtest.")
        if print_result:
            print(stats)
        return stats, bt

    # print results
    if print_result:
        print(stats)
    return stats, bt


def optimize_strategy(filepath, return_heatmap=True):
    # load and prepare data
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    # Reduced optimization grid for better performance
    atr_periods = [15, 20, 25, 30]
    k_atr_values = [1.8, 2.0, 2.2, 2.4]
    uptail_ratios = [0.6, 0.7, 0.8]
    prev3_ratios = [0.4, 0.5, 0.6]
    
    # Precompute only necessary ATR periods
    for period in atr_periods:
        df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    
    # Drop NaNs based on maximum ATR period
    max_atr_period = max(atr_periods)
    df.dropna(inplace=True)

    # run backtest: trade on close to attempt entry at close of signal bar
    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    
    # Define optimization parameters
    if return_heatmap:
        optimize_result, heatmap = bt.optimize(
            atr_period=atr_periods,
            k_atr=k_atr_values,
            uptail_max_ratio=uptail_ratios,
            prev3_min_ratio=prev3_ratios,
            maximize='Return [%]',
            constraint=lambda param: param.uptail_max_ratio > 0.5 and param.prev3_min_ratio > 0.3,
            return_heatmap=True
        )
        print("Optimization Results:")
        print(optimize_result)
        
        # Print optimized parameters
        st = optimize_result._strategy
        print(f"\nOptimized Parameters:")
        print(f"  atr_period: {st.atr_period}")
        print(f"  k_atr: {st.k_atr}")
        print(f"  uptail_max_ratio: {st.uptail_max_ratio}")
        print(f"  prev3_min_ratio: {st.prev3_min_ratio}")
        
        return optimize_result, bt, heatmap
    else:
        optimize_result = bt.optimize(
            atr_period=atr_periods,
            k_atr=k_atr_values,
            uptail_max_ratio=uptail_ratios,
            prev3_min_ratio=prev3_ratios,
            maximize='Return [%]',
            constraint=lambda param: param.uptail_max_ratio > 0.5 and param.prev3_min_ratio > 0.3
        )
        print("Optimization Results:")
        print(optimize_result)
        
        # Print optimized parameters
        st = optimize_result._strategy
        print(f"\nOptimized Parameters:")
        print(f"  atr_period: {st.atr_period}")
        print(f"  k_atr: {st.k_atr}")
        print(f"  uptail_max_ratio: {st.uptail_max_ratio}")
        print(f"  prev3_min_ratio: {st.prev3_min_ratio}")
        
        return optimize_result, bt


def plot_strategy(filepath, filename='strategy_plot.html'):
    _, bt = run_backtest(filepath, print_result=False)
    bt.plot(filename=filename)
    print(f"Plot saved as {filename}")


def plot_heatmaps(filepath):
    # Run optimization with return_heatmap=True to get built-in heatmap support
    optimize_result, bt, heatmap = optimize_strategy(filepath, return_heatmap=True)
    
    # Use built-in heatmap plotting from backtesting.py
    try:
        # Check if matplotlib is available
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Save heatmap as PNG using backtesting's built-in heatmap support
        heatmap.plot().figure.savefig('parameter_heatmaps.png', bbox_inches='tight')
        print("Parameter heatmaps saved as parameter_heatmaps.png")
        
    except ImportError:
        print("Error plotting heatmaps: matplotlib is required for heatmap visualization")
    except Exception as e:
        print(f"Error plotting heatmaps: {e}")

# -------------
# CLI entrypoint
# -------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Big Bar Trading Strategy")
    parser.add_argument("filepath", help="Path to CSV data file", nargs='?', default='data.csv')
    parser.add_argument("--no-optimize", action="store_true", help="Skip strategy optimization")
    parser.add_argument("--no-plot", action="store_true", help="Skip strategy plotting")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip parameter heatmap generation")
    
    args = parser.parse_args()
    
    # Run optimization and plotting by default
    print(f"Running BigBarAllIn strategy on {args.filepath}...")
    
    if not args.no_optimize:
        # Run optimization with heatmap
        optimize_result, bt, heatmap = optimize_strategy(args.filepath, return_heatmap=True)
        
        # Plot optimized results
        if not args.no_plot:
            bt.plot(filename='optimized_strategy_plot.html')
        
        # Generate parameter heatmaps using built-in functionality
        if not args.no_heatmaps:
            try:
                from backtesting.lib import plot_heatmaps
                
                # Create interactive heatmap using backtesting.lib.plot_heatmaps
                plot_heatmaps(heatmap, agg='mean')
                
            except ImportError:
                print("Error plotting heatmaps: backtesting.lib.plot_heatmaps is required")
            except Exception as e:
                print(f"Error plotting heatmaps: {e}")
    else:
        # Run backtest without optimization
        print("\n1. Running backtest without optimization...")
        run_backtest(args.filepath, print_result=True)
        
    print("\nAll operations completed successfully!")
