"""Backtesting utilities wrapping Backtrader."""

import backtrader as bt
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    strategy: type[bt.Strategy],
    cash: float = 100_000.0,
    commission: float = 0.001,
    plot: bool = False,
    **strategy_kwargs,
) -> dict:
    """Run a backtest on OHLCV data with the given strategy.

    Args:
        df: DataFrame with OHLCV columns (and optionally a 'signal' column)
        strategy: A Backtrader Strategy class
        cash: Starting portfolio cash
        commission: Commission per trade (0.001 = 0.1%)
        plot: Whether to display a plot after the run
        **strategy_kwargs: Additional params passed to the strategy

    Returns:
        Dict with starting_value, ending_value, return_pct, and trade stats.
    """
    cerebro = bt.Cerebro()

    # Prepare data feed
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(strategy, **strategy_kwargs)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    starting_value = cerebro.broker.getvalue()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()

    strat = results[0]
    trade_analysis = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    if plot:
        cerebro.plot()

    return {
        "starting_value": starting_value,
        "ending_value": ending_value,
        "return_pct": (ending_value - starting_value) / starting_value * 100,
        "sharpe_ratio": sharpe.get("sharperatio"),
        "max_drawdown_pct": drawdown.get("max", {}).get("drawdown"),
        "total_trades": trade_analysis.get("total", {}).get("total", 0),
        "trades": trade_analysis,
    }
