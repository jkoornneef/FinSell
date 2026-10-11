"""Backtrader strategy integration."""

import backtrader as bt


class SignalStrategy(bt.Strategy):
    """A Backtrader strategy driven by a pre-computed signal column.

    Params:
        signal_col: Name of the signal data line (1=buy, -1=sell, 0=hold)
        stake: Number of shares per trade
    """

    params = (
        ("stake", 100),
    )

    def __init__(self):
        self.signal = self.datas[0].signal

    def next(self):
        if self.signal[0] == 1 and not self.position:
            self.buy(size=self.params.stake)
        elif self.signal[0] == -1 and self.position:
            self.sell(size=self.params.stake)


class MACDCrossStrategy(bt.Strategy):
    """MACD crossover strategy using Backtrader's built-in indicators."""

    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
        ("stake", 100),
    )

    def __init__(self):
        macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal,
        )
        self.crossover = bt.indicators.CrossOver(macd.macd, macd.signal)

    def next(self):
        if self.crossover > 0 and not self.position:
            self.buy(size=self.params.stake)
        elif self.crossover < 0 and self.position:
            self.sell(size=self.params.stake)
