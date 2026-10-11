import types

from finsell import composite


class DummyTicker:
    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def info(self):
        return None


def test_industry_group_rs_returns_neutral_on_missing_info(monkeypatch):
    monkeypatch.setattr(composite.yf, "Ticker", DummyTicker)

    assert composite.industry_group_rs("XFN.TO") == 50.0
