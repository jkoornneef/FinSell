from datetime import datetime, timedelta

import pandas as pd

from finsell import data_cache


def test_fetch_window_uses_today_not_tomorrow():
    now = datetime(2026, 8, 11, 13, 23, 39)

    start, end = data_cache._get_fetch_window(now=now, lookback_days=365)

    assert start == now - timedelta(days=365)
    assert end.date() == now.date()
    assert end.date() != now.date() + timedelta(days=1)


def test_fetch_history_recovers_from_corrupt_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    ticker = "TEST.TO"
    p = data_cache._cache_path(ticker)
    p.mkdir(parents=True, exist_ok=True)
    (p / "monthly.parquet").write_bytes(b"not-a-valid-parquet")

    def fake_fetch(_ticker, interval, start, end):
        idx = pd.date_range(start=start, periods=3, freq="MS")
        return pd.DataFrame(
            {
                "Open": [10, 11, 12],
                "High": [11, 12, 13],
                "Low": [9, 10, 11],
                "Close": [10.5, 11.5, 12.5],
                "Volume": [100, 200, 300],
            },
            index=idx,
        )

    monkeypatch.setattr(data_cache, "_fetch", fake_fetch)

    monthly, daily = data_cache.fetch_history(ticker)

    assert len(monthly) == 3
    assert len(daily) == 3
    assert (p / "monthly.parquet").exists()


def test_nan_pattern_scores_are_ignored_for_momentum():
    from finsell import dashboard_v2

    patterns = {
        "cup_handle": 0.5,
        "distribution": 0.25,
        "golden_cross": float("nan"),
        "purchase_timing": None,
    }

    cleaned = dashboard_v2._sanitize_pattern_scores(patterns)

    assert cleaned == {"cup_handle": 0.5, "distribution": 0.25}
    assert dashboard_v2._pattern_momentum(cleaned) == 0.375
