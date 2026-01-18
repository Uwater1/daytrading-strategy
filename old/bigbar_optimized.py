import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import sys
import math
from functools import lru_cache
from numba import jit, float64, int64, boolean
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Global cache for precomputed data
_data_cache = {}
_atr_cache = {}
_week_cache = {}

@lru_cache(maxsize=20)
def load_data_cached(filepath):
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

def load_data(filepath):
    return load_data_cached(filepath)

@lru_cache(maxsize=100)
def compute_atr_cached(high, low, close, period):
    return ta.atr(high, low, close, length=period)

@jit(nopython=True)
def compute_week_boundaries_numba(index_values):
    n = len(index_values)
    is_restricted = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        # Extract year and week from index_values
        # This is a simplified version - actual implementation would need proper datetime handling
        pass
    
    return is_restricted

@jit(nopython=True)
def calculate_weighted_sum_numba(close_values, open_values, body):
    if len(close_values) < 4 or len(open_values) < 4 or body == 0:
        return 0.0
    
    bar1 = close_values[-4] - open_values[-4]
    bar2 = close_values[-3] - open_values[-3]
    bar3 = close_values[-2] - open_values[-2]
    
    weighted_sum = (1.0 * bar1) + (2.0 * bar2) + (3.0 * bar3)
    return weighted_sum / body

@jit(nopython=True)
def check_entry_conditions_numba(open_p, high_p, low_p, close_p, size, body, atr, 
                                k_atr, uptail_max_ratio, previous_weight, normalized_weighted_sum):
    cond_green = close_p > open_p
    cond_size = size >= k_atr * atr
    cond_prev3_long = normalized_weighted_sum >= previous_weight
    cond_uptail_long = (high_p - close_p) < (uptail_max_ratio * size)

    cond_red = close_p < open_p
    cond_prev3_short = normalized_weighted_sum <= -previous_weight
    cond_downtail_short = (close_p - low_p) < (uptail_max_ratio * size)

    return (cond_green, cond_size, cond_prev3_long, cond_uptail_long,
            cond_red, cond_prev3_short, cond_downtail_short)

def compute_week_boundaries_cached(index):
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

ATR_PERIOD = 20
K_ATR = 2.0
UPTAIL_MAX_RATIO = 0.7
PREV3_MIN_RATIO = 0.5
BUFFER_RATIO = 0.01
INITIAL_CASH = 100000
COMMISSION = 0.0
TRADE_ON_CLOSE = True

class BigBarAllIn(Strategy):
    atr_period = 20
    k_atr_int = 20
    uptail_max_ratio_int = 7
    previous_weight_int = 1
    buffer_ratio_int = 1
    
    def init(self):
        self.trades_log = []
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None

    def next(self):
        k_atr = self.k_atr_int / 10
        uptail_max_ratio = self.uptail_max_ratio_int / 10
        previous_weight = self.previous_weight_int / 10
        buffer_ratio = self.buffer_ratio_int / 100
        
        if len(self.data.Close) < (self.atr_period + 5):
            return

        i = len(self.data.Close) - 1
        is_restricted = self.data.df['is_restricted'].iat[i]
        
        if self.position and is_restricted:
            exit_price = self.data.Close[-1]
            self._close_position_and_log(exit_price)
            return
        
        open_p = self.data.Open[-1]
        high_p = self.data.High[-1]
        low_p = self.data.Low[-1]
        close_p = self.data.Close[-1]
        size = high_p - low_p
        body = abs(close_p - open_p)
        atr = self.data.df[f'ATR_{self.atr_period}'].iat[i]

        if not self.position and not is_restricted:
            try:
                bar1 = (self.data.Close[-4] - self.data.Open[-4])
                bar2 = (self.data.Close[-3] - self.data.Open[-3])
                bar3 = (self.data.Close[-2] - self.data.Open[-2])
                
                weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
            except Exception:
                return

            normalized_weighted_sum = weighted_sum / body if body != 0 else 0

            cond_green = close_p > open_p
            cond_size = (size >= k_atr * atr) if (not math.isnan(atr) and atr > 0) else False
            cond_prev3_long = (normalized_weighted_sum >= previous_weight)
            cond_uptail_long = ( (high_p - close_p) < (uptail_max_ratio * size) )

            cond_red = close_p < open_p
            cond_prev3_short = (normalized_weighted_sum <= -previous_weight)
            cond_downtail_short = ( (close_p - low_p) < (uptail_max_ratio * size) )

            if cond_green and cond_size and cond_prev3_long and cond_uptail_long:
                equity = self.equity
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                self.buy(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'long'
                self._current_stop = low_p - (buffer_ratio * size)
                return

            if cond_red and cond_size and cond_prev3_short and cond_downtail_short:
                equity = self.equity
                if equity <= 0 or close_p <= 0:
                    return
                units = int(equity / close_p)
                if units < 1:
                    return

                self.sell(size=units)
                self._in_trade = True
                self._entry_price = close_p
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = high_p
                self._entry_bar_low = low_p
                self._bars_since_entry = 0
                self._position_direction = 'short'
                self._current_stop = high_p + (buffer_ratio * size)
                return

        if self.position:
            self._bars_since_entry += 1

            prev_bar_high = self.data.High[-1]
            prev_bar_low = self.data.Low[-1]
            prev_bar_close = self.data.Close[-1]
            prev_bar_open = self.data.Open[-1]

            if self._position_direction == 'long':
                if self._bars_since_entry == 1:
                    is_red = prev_bar_close <= prev_bar_open
                    didnnot_new_high = (prev_bar_high <= self._entry_bar_high)
                    if is_red or didnnot_new_high:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        potential_stop = prev_bar_low - (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop > self._current_stop:
                            self._current_stop = potential_stop

                if self._bars_since_entry >= 2:
                    try:
                        low_1 = self.data.Low[-1]
                        low_2 = self.data.Low[-2]
                        trailing_stop = min(low_1, low_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    if trailing_stop > self._current_stop:
                        self._current_stop = trailing_stop

                    if self.data.Low[-1] <= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

            elif self._position_direction == 'short':
                if self._bars_since_entry == 1:
                    is_green = prev_bar_close >= prev_bar_open
                    didnnot_new_low = (prev_bar_low >= self._entry_bar_low)
                    if is_green or didnnot_new_low:
                        exit_price = prev_bar_close
                        self._close_position_and_log(exit_price)
                        return
                    else:
                        potential_stop = prev_bar_high + (BUFFER_RATIO * (self._entry_bar_high - self._entry_bar_low))
                        if potential_stop < self._current_stop:
                            self._current_stop = potential_stop

                if self._bars_since_entry >= 2:
                    try:
                        high_1 = self.data.High[-1]
                        high_2 = self.data.High[-2]
                        trailing_stop = max(high_1, high_2)
                    except Exception:
                        trailing_stop = self._current_stop

                    if trailing_stop < self._current_stop:
                        self._current_stop = trailing_stop

                    if self.data.High[-1] >= self._current_stop:
                        exit_price = self.data.Close[-1]
                        self._close_position_and_log(exit_price)
                        return

    def _close_position_and_log(self, exit_price):
        if not self.position:
            return
        if self._position_direction == 'long':
            pnl = (exit_price - self._entry_price) * self._entry_size
        else:
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
        self.position.close()
        self._in_trade = False
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None
        self._position_direction = None


def run_backtest(filepath, print_result=True, atr_period=ATR_PERIOD):
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    # Compute ATR with proper handling
    df[f'ATR_{atr_period}'] = compute_atr_cached(tuple(df['High']), tuple(df['Low']), tuple(df['Close']), atr_period)
    
    # Only drop NaN values from ATR column, keep other data
    df = df.dropna(subset=[f'ATR_{atr_period}'])
    
    df['is_restricted'] = compute_week_boundaries_cached(df.index)

    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    stats = bt.run(
        atr_period=atr_period,
        k_atr_int=20,
        uptail_max_ratio_int=7,
        previous_weight_int=1,
        buffer_ratio_int=1
    )
    
    if hasattr(stats, '_trades') and not stats._trades.empty:
        trades_df = stats._trades[['EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'Size', 'PnL']]
        trades_df.columns = ['entry_index', 'exit_index', 'entry_price', 'exit_price', 'size', 'pnl']
        trades_df['direction'] = trades_df['size'].apply(lambda x: 'long' if x > 0 else 'short')
        trades_df['size'] = trades_df['size'].abs()
        trades_df['entry_date'] = trades_df['entry_index'].apply(lambda idx: df.index[idx])
        trades_df['exit_date'] = trades_df['exit_index'].apply(lambda idx: df.index[idx])
        trades_df = trades_df.drop(['entry_index', 'exit_index'], axis=1)
        trades_df = trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'size', 'pnl', 'direction']]
        trades_df['pnl'] = trades_df['pnl'].round(2)
        trades_df.to_csv('bigbar_trades.csv', index=False)
    else:
        print("No trades were executed in this backtest.")
        if print_result:
            print(stats)
        return stats, bt

    if print_result:
        print(stats)
    return stats, bt


def optimize_strategy(filepath, return_heatmap=True):
    df = load_data(filepath)
    if df is None:
        raise SystemExit("Failed to load data")

    atr_periods = [10, 100]
    k_atr_int_values = [10, 40]  
    uptail_ratios_int_values = [5, 9]  
    previous_weights_int_values = [1, 8]  
    buffer_ratio_int_values = [1]  
    
    min_atr_period = min(atr_periods)
    max_atr_period = max(atr_periods)
    for period in range(min_atr_period, max_atr_period + 1):
        df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    
    df.dropna(inplace=True)
    
    df['week_number'] = df.index.isocalendar().week
    df['year'] = df.index.isocalendar().year
    df['week_id'] = df['year'] * 100 + df['week_number']
    df['bar_in_week'] = df.groupby('week_id').cumcount()
    week_total_bars = df.groupby('week_id').size()
    week_total_bars_dict = week_total_bars.to_dict()
    df['is_restricted'] = False
    for week_id, total_bars in week_total_bars_dict.items():
        first_6 = df['week_id'] == week_id
        df.loc[first_6 & (df['bar_in_week'] < 6), 'is_restricted'] = True
        df.loc[first_6 & (df['bar_in_week'] >= (total_bars - 6)), 'is_restricted'] = True

    bt = Backtest(df, BigBarAllIn, cash=INITIAL_CASH, commission=COMMISSION, trade_on_close=TRADE_ON_CLOSE)
    
    if return_heatmap:
        optimize_result, heatmap = bt.optimize(
            atr_period=atr_periods,
            k_atr_int=k_atr_int_values,
            uptail_max_ratio_int=uptail_ratios_int_values,
            previous_weight_int=previous_weights_int_values,
            buffer_ratio_int=buffer_ratio_int_values,
            maximize='Return [%]',
            constraint=lambda param: param.uptail_max_ratio_int > 5 and param.previous_weight_int > 0,
            return_heatmap=True,
            method='sambo'
        )
        print("Optimization Results:")
        print(optimize_result)
        
        st = optimize_result._strategy
        print(f"\nOptimized Parameters:")
        print(f"  atr_period: {st.atr_period}")
        print(f"  k_atr: {st.k_atr_int / 10}")
        print(f"  uptail_max_ratio: {st.uptail_max_ratio_int / 10}")
        print(f"  previous_weight: {st.previous_weight_int / 10}")
        print(f"  buffer_ratio: {st.buffer_ratio_int / 100}")
        
        return optimize_result, bt, heatmap
    else:
        optimize_result = bt.optimize(
            atr_period=atr_periods,
            k_atr_int=k_atr_int_values,
            uptail_max_ratio_int=uptail_ratios_int_values,
            previous_weight_int=previous_weights_int_values,
            buffer_ratio_int=buffer_ratio_int_values,
            maximize='Return [%]',
            constraint=lambda param: param.uptail_max_ratio_int > 5 and param.previous_weight_int > 0,
            method='sambo'
        )
        print("Optimization Results:")
        print(optimize_result)
        
        st = optimize_result._strategy
        print(f"\nOptimized Parameters:")
        print(f"  atr_period: {st.atr_period}")
        print(f"  k_atr: {st.k_atr_int / 10}")
        print(f"  uptail_max_ratio: {st.uptail_max_ratio_int / 10}")
        print(f"  previous_weight: {st.previous_weight_int / 10}")
        print(f"  buffer_ratio: {st.buffer_ratio_int / 100}")
        
        return optimize_result, bt


def plot_strategy(filepath, filename='strategy_plot.html'):
    _, bt = run_backtest(filepath, print_result=False)
    bt.plot(filename=filename)
    print(f"Plot saved as {filename}")


def plot_heatmaps(filepath):
    optimize_result, bt, heatmap = optimize_strategy(filepath, return_heatmap=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        heatmap.plot().figure.savefig('parameter_heatmaps.png', bbox_inches='tight')
        print("Parameter heatmaps saved as parameter_heatmaps.png")
        
    except ImportError:
        print("Error plotting heatmaps: matplotlib is required for heatmap visualization")
    except Exception as e:
        print(f"Error plotting heatmaps: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Big Bar Trading Strategy")
    parser.add_argument("filepath", help="Path to CSV data file", nargs='?', default='data.csv')
    parser.add_argument("--no-optimize", action="store_true", help="Skip strategy optimization")
    parser.add_argument("--no-plot", action="store_true", help="Skip strategy plotting")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip parameter heatmap generation")
    
    args = parser.parse_args()
    
    print(f"Running BigBarAllIn strategy on {args.filepath}...")
    
    if not args.no_optimize:
        optimize_result, bt, heatmap = optimize_strategy(args.filepath, return_heatmap=True)
        
        if not args.no_plot:
            bt.plot(filename='optimized_strategy_plot.html')
        
        if not args.no_heatmaps:
            try:
                from backtesting.lib import plot_heatmaps
                plot_heatmaps(heatmap, agg='mean')
                
            except ImportError:
                print("Error plotting heatmaps: backtesting.lib.plot_heatmaps is required")
            except Exception as e:
                print(f"Error plotting heatmaps: {e}")
    else:
        print("\n1. Running backtest without optimization...")
        run_backtest(args.filepath, print_result=True)
        
    print("\nAll operations completed successfully!")
