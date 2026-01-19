import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import numpy as np

# Mocking the BigBarAllIn to test different stop logics quickly
class BigBarTest(Strategy):
    atr_period = 20
    k_atr_int = 20
    uptail_max_ratio_int = 7
    previous_weight_int = 1
    buffer_ratio_int = 1
    stop_type = 'old' # 'old', 'new', 'clean_tight', 'buffer_tight'

    def init(self):
        self.k_atr = self.k_atr_int / 10
        self.uptail_max_ratio = self.uptail_max_ratio_int / 10
        self.previous_weight = self.previous_weight_int / 100
        self.buffer_ratio = self.buffer_ratio_int / 100
        
        self._close_array = self.data.df['Close'].values
        self._open_array = self.data.df['Open'].values
        self._high_array = self.data.df['High'].values
        self._low_array = self.data.df['Low'].values
        self._atr_array = self.data.df[f'ATR_{self.atr_period}'].values
        self._is_restricted_array = self.data.df['is_restricted'].values
        self._current_stop = None
        self._position_direction = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0

    def next(self):
        i = len(self.data) - 1
        if i < self.atr_period + 5: return
        
        is_restricted = self._is_restricted_array[i]
        if self.position and is_restricted:
            self.position.close()
            return

        open_p = self._open_array[i]
        high_p = self._high_array[i]
        low_p = self._low_array[i]
        close_p = self._close_array[i]
        size = high_p - low_p
        atr = self._atr_array[i]

        if not self.position and not is_restricted:
            # Simplified momentum for testing consistency
            bar_1 = self._close_array[i-1] - self._open_array[i-1]
            bar_2 = self._close_array[i-2] - self._open_array[i-2]
            bar_3 = self._close_array[i-3] - self._open_array[i-3]
            momentum = (3 * bar_1) + (2 * bar_2) + (1 * bar_3)
            
            if close_p > open_p:
                dynamic_k = self.k_atr - (momentum / atr * self.previous_weight if atr > 0 else 0)
                if size >= dynamic_k * atr and (high_p - close_p) < (self.uptail_max_ratio * size):
                    self.buy()
                    self._position_direction = 'long'
                    self._current_stop = low_p + (close_p - low_p) / 2 - (self.buffer_ratio * size)
                    self._entry_bar_high = high_p
                    self._entry_bar_low = low_p
                    self._bars_since_entry = 0
            elif close_p < open_p:
                dynamic_k = self.k_atr + (momentum / atr * self.previous_weight if atr > 0 else 0)
                if size >= dynamic_k * atr and (close_p - low_p) < (self.uptail_max_ratio * size):
                    self.sell()
                    self._position_direction = 'short'
                    self._current_stop = high_p - (high_p - close_p) / 2 + (self.buffer_ratio * size)
                    self._entry_bar_high = high_p
                    self._entry_bar_low = low_p
                    self._bars_since_entry = 0

        elif self.position:
            self._bars_since_entry += 1
            if self._bars_since_entry == 1:
                # First bar exit logic
                if self._position_direction == 'long':
                    if close_p <= open_p or high_p <= self._entry_bar_high:
                        self.position.close()
                    else:
                        self._current_stop = max(self._current_stop, low_p - (self.buffer_ratio * (self._entry_bar_high - self._entry_bar_low)))
                else:
                    if close_p >= open_p or low_p >= self._entry_bar_low:
                        self.position.close()
                    else:
                        self._current_stop = min(self._current_stop, high_p + (self.buffer_ratio * (self._entry_bar_high - self._entry_bar_low)))
            else:
                # Trailing stop
                if self.stop_type == 'old':
                    if self._position_direction == 'long':
                        trial = min(low_p, self._low_array[i-1])
                        if trial > self._current_stop: self._current_stop = trial
                        if low_p <= self._current_stop: self.position.close()
                    else:
                        trial = max(high_p, self._high_array[i-1])
                        if trial < self._current_stop: self._current_stop = trial
                        if high_p >= self._current_stop: self.position.close()
                elif self.stop_type == 'new':
                    if self._position_direction == 'long':
                        trial = min(self._low_array[i-1], self._low_array[i-2])
                        if trial > self._current_stop: self._current_stop = trial
                        if low_p <= self._current_stop: self.position.close()
                    else:
                        trial = max(self._high_array[i-1], self._high_array[i-2])
                        if trial < self._current_stop: self._current_stop = trial
                        if high_p >= self._current_stop: self.position.close()
                elif self.stop_type == 'clean_tight':
                    # Exit if current low breaks previous bar's low
                    if self._position_direction == 'long':
                        if low_p < self._low_array[i-1]:
                            self.position.close()
                        else:
                            self._current_stop = max(self._current_stop, self._low_array[i-1])
                    else:
                        if high_p > self._high_array[i-1]:
                            self.position.close()
                        else:
                            self._current_stop = min(self._current_stop, self._high_array[i-1])
                elif self.stop_type == 'mid_tight':
                    # Exit if current low breaks midpoint of previous bar
                    if self._position_direction == 'long':
                        mid = self._low_array[i-1] + (self._close_array[i-1] - self._low_array[i-1])/2
                        if low_p < mid:
                            self.position.close()
                elif self.stop_type == 'proper_tight':
                    # Check current bar against PREVIOUS stop, then update
                    if self._position_direction == 'long':
                        if low_p <= self._current_stop:
                            self.position.close()
                        else:
                            self._current_stop = max(self._current_stop, low_p)
                    else:
                        if high_p >= self._current_stop:
                            self.position.close()
                        else:
                            self._current_stop = min(self._current_stop, high_p)
                elif self.stop_type == 'proper_buffer_tight':
                    # Check current bar against PREVIOUS stop, then update with buffer
                    if self._position_direction == 'long':
                        if low_p <= self._current_stop:
                            self.position.close()
                        else:
                            self._current_stop = max(self._current_stop, low_p - (self.buffer_ratio * size))
                    else:
                        if high_p >= self._current_stop:
                            self.position.close()
                        else:
                            self._current_stop = min(self._current_stop, high_p + (self.buffer_ratio * size))
                elif self.stop_type == 'proper_close_tight':
                    # Only move stop if current bar closes higher (for long)
                    if self._position_direction == 'long':
                        if low_p <= self._current_stop:
                            self.position.close()
                        elif close_p > self._close_array[i-1]:
                            self._current_stop = max(self._current_stop, low_p)
                    else:
                        if high_p >= self._current_stop:
                            self.position.close()
                        elif close_p < self._close_array[i-1]:
                            self._current_stop = min(self._current_stop, high_p)

if __name__ == "__main__":
    df = pd.read_csv('gd5m.csv')
    df.columns = [x.lower() for x in df.columns]
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
    
    # Simple restriction
    from datetime import time as dtime
    df['is_restricted'] = False # Simplified for testing
    
    df = df.dropna()
    
    for st in ['old', 'new', 'proper_tight', 'proper_buffer_tight', 'proper_close_tight']:
        bt = Backtest(df, BigBarTest, cash=100000, commission=0, trade_on_close=True)
        stats = bt.run(stop_type=st)
        print(f"Stop Type: {st:20} | Return: {stats['Return [%]']:8.3f}% | MDD: {stats['Max. Drawdown [%]']:8.3f}% | Trades: {stats['# Trades']}")
