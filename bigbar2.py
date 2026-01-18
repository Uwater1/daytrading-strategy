# bigbar_allin_single_file.py
# Combined version of base.py + test_bigbar_allin.py
# Includes plotting and heatmap output for browser viewing

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
import sys
import math
from pathlib import Path

# ==========================
# Data loading (from base.py)
# ==========================

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Required columns
    required = {'time', 'open', 'high', 'low', 'close'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    # Parse time
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Ensure float dtype
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)

    return df


# ==========================
# Strategy parameters
# ==========================

ATR_PERIOD = 80
UPTAIL_MAX_RATIO = 0.5
PREV3_MIN_RATIO = 0.8
BUFFER_RATIO = 0.02

INITIAL_CASH = 100_000
COMMISSION = 0.0
TRADE_ON_CLOSE = True


# ==========================
# Strategy definition
# ==========================

class BigBarAllIn(Strategy):
    k_atr = 3.0  # this will be optimized / heatmapped

    def init(self):
        self.trades_log = []
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None

    def next(self):
        if len(self.data.Close) < (ATR_PERIOD + 5):
            return

        i = len(self.data.Close) - 1

        o = float(self.data.Open[-1])
        h = float(self.data.High[-1])
        l = float(self.data.Low[-1])
        c = float(self.data.Close[-1])

        size = h - l
        body = abs(c - o)
        atr = float(self.data.df['atr'].iat[i])

        # =====================
        # ENTRY LOGIC (LONG)
        # =====================
        if not self.position:
            prev3 = (
                (self.data.Close[-2] - self.data.Open[-2]) +
                (self.data.Close[-3] - self.data.Open[-3]) +
                (self.data.Close[-4] - self.data.Open[-4])
            )

            cond_green = c > o
            cond_big = size >= self.k_atr * atr
            cond_prev3 = prev3 >= PREV3_MIN_RATIO * body
            cond_uptail = (h - c) < UPTAIL_MAX_RATIO * size

            if cond_green and cond_big and cond_prev3 and cond_uptail:
                equity = self.equity
                units = int(equity // c)
                if units <= 0:
                    return

                self.buy(size=units)
                self._entry_price = c
                self._entry_size = units
                self._entry_index = i
                self._entry_bar_high = h
                self._entry_bar_low = l
                self._bars_since_entry = 0
                self._current_stop = l - BUFFER_RATIO * size
                return

        # =====================
        # EXIT & TRAILING STOP
        # =====================
        if self.position:
            self._bars_since_entry += 1

            # Immediate next-bar failure
            if self._bars_since_entry == 1:
                is_red = c <= o
                no_new_high = h <= self._entry_bar_high
                if is_red or no_new_high:
                    self._exit(c)
                    return

            # Trailing stop after 2 bars
            if self._bars_since_entry >= 2:
                trailing = min(self.data.Low[-1], self.data.Low[-2])
                if trailing > self._current_stop:
                    self._current_stop = trailing

                if l <= self._current_stop:
                    self._exit(c)
                    return

    def _exit(self, price):
        pnl = (price - self._entry_price) * self._entry_size
        self.trades_log.append({
            'entry_index': self._entry_index,
            'exit_index': len(self.data.Close) - 1,
            'entry_price': self._entry_price,
            'exit_price': price,
            'size': self._entry_size,
            'pnl': pnl
        })
        self.position.close()
        self._reset()

    def _reset(self):
        self._entry_price = None
        self._entry_size = None
        self._entry_index = None
        self._entry_bar_high = None
        self._entry_bar_low = None
        self._bars_since_entry = 0
        self._current_stop = None


# ==========================
# Backtest runner + plotting
# ==========================

def run(csv_file):
    df = load_data(csv_file)

    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    df.dropna(inplace=True)

    bt = Backtest(
        df,
        BigBarAllIn,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        trade_on_close=TRADE_ON_CLOSE
    )

    # Heatmap over k_atr
    heatmap = bt.optimize(
        k_atr=[2.0, 2.5, 3.0, 3.5],
        maximize='Equity Final [$]',
        return_heatmap=True
    )

    stats = bt.run()

    # ==========================
    # Output files
    # ==========================
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)

    plot_filename = out_dir / 'equity_curve.html'
    heatmap_filename = out_dir / 'heatmap.html'

    bt.plot(filename=plot_filename)
    plot_heatmaps(heatmap, filename=heatmap_filename)

    # ==========================
    # Kelly + win rate
    # ==========================
    strat = bt._strategy
    trades = pd.DataFrame(strat.trades_log)
    trades.to_csv(out_dir / 'trades.csv', index=False)

    pnl = trades['pnl'].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = len(wins) / len(pnl) if len(pnl) else np.nan
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0

    b = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    kelly = win_rate - (1 - win_rate) / b if b and b > 0 else np.nan

    print('\n===== RESULTS =====')
    print(f'Trades: {len(pnl)}')
    print(f'Win rate: {win_rate:.4f}')
    print(f'Avg win: {avg_win:.2f}')
    print(f'Avg loss: {avg_loss:.2f}')
    print(f'Kelly fraction: {kelly:.4f}')
    print(f'Equity final: {stats["Equity Final [$]"]:.2f}')
    print('\nSaved:')
    print(f' - {plot_filename}')
    print(f' - {heatmap_filename}')
    print(f' - {out_dir / "trades.csv"}')

    return stats


# ==========================
# CLI
# ==========================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python bigbar_allin_single_file.py your_5min.csv')
        sys.exit(1)

    run(sys.argv[1])
