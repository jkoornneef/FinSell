import finsell.dashboard_v2 as dashboard_v2


def test_build_research_gauge_export_payload_includes_values_and_descriptions():
    entries = dashboard_v2._build_research_gauge_export_payload(
        ticker="AAPL",
        patterns={
            "cup_handle": 0.82,
            "distribution": -0.25,
            "purchase_timing": 0.6,
        },
        decisions={
            "buy": 0.7,
            "sell": 0.1,
            "hold": 0.4,
            "buy_more": 0.9,
        },
    )

    assert isinstance(entries, list)
    assert any(item["name"] == "Composite Score" and item["value"] == 0.82 for item in entries)
    assert any(item["name"] == "BUY" and item["value"] == 0.7 for item in entries)
    assert all("description" in item and isinstance(item["description"], str) for item in entries)
    assert any("Higher is better" in item["description"] for item in entries)


def test_account_ticker_options_include_holdings_and_watchers():
    holdings = [
        ("AAPL", False),
        ("MSFT", True),
    ]

    options = dashboard_v2._account_ticker_options(holdings)

    assert options == [
        {"label": "AAPL (Holding)", "value": "AAPL"},
        {"label": "MSFT (Watcher)", "value": "MSFT"},
    ]
