"""Quick-start example: fetch data, detect patterns, generate signals, and backtest."""

from finsell.data import fetch_ohlcv
from finsell.indicators import add_indicators
from finsell.patterns import find_swing_highs, find_swing_lows, detect_double_bottom
from finsell.signals import macd_crossover_signals
from finsell.backtest import run_backtest
from finsell.strategy import MACDCrossStrategy


def main():
    # 1. Fetch data
    print("Fetching AAPL daily data (1 year)...")
    df = fetch_ohlcv("AAPL", period="1y", interval="1d")
    print(f"  Got {len(df)} bars\n")

    # 2. Add technical indicators
    df = add_indicators(df)
    print("Indicators added:")
    print(f"  Columns: {list(df.columns)}\n")

    # 3. Detect patterns
    swing_highs = find_swing_highs(df, order=5)
    swing_lows = find_swing_lows(df, order=5)
    print(f"Swing highs detected: {swing_highs.sum()}")
    print(f"Swing lows detected:  {swing_lows.sum()}\n")

    double_bottoms = detect_double_bottom(df, tolerance=0.03)
    print(f"Double bottom patterns found: {len(double_bottoms)}")
    for p in double_bottoms[:3]:
        print(f"  {p['first_idx'].date()} -> {p['second_idx'].date()}: "
              f"lows={p['first_low']:.2f}/{p['second_low']:.2f}, neckline={p['neckline']:.2f}")
    print()

    # 4. Generate signals
    signals = macd_crossover_signals(df)
    buy_count = (signals["signal"] == 1).sum()
    sell_count = (signals["signal"] == -1).sum()
    print(f"MACD crossover signals: {buy_count} buys, {sell_count} sells\n")

    # 5. Backtest
    print("Running MACD crossover backtest...")
    results = run_backtest(df, MACDCrossStrategy, cash=100_000)
    print(f"  Starting value: ${results['starting_value']:,.2f}")
    print(f"  Ending value:   ${results['ending_value']:,.2f}")
    print(f"  Return:          {results['return_pct']:.2f}%")
    print(f"  Sharpe ratio:    {results['sharpe_ratio']}")
    print(f"  Max drawdown:    {results['max_drawdown_pct']}%")
    print(f"  Total trades:    {results['total_trades']}")


if __name__ == "__main__":
    main()
