"""Portfolio persistence — JSON file for holdings and watchers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"


@dataclass
class Holding:
    """A single stock holding or watcher."""

    ticker: str
    shares: float = 0.0  # 0 = watcher
    wac: float = 0.0  # weighted average cost; 0 for watchers
    sector: str = ""
    market_override: str = "auto"  # auto | us | tsx | dual
    notes: str = ""
    added: str = ""  # ISO date when added

    @property
    def is_watcher(self) -> bool:
        return self.shares == 0


@dataclass
class Portfolio:
    """Full portfolio state."""

    holdings: list[Holding] = field(default_factory=list)
    lru: list[str] = field(default_factory=list)  # recently researched tickers
    lru_max: int = 20

    # ---- CRUD ----

    def add(self, ticker: str, shares: float = 0, wac: float = 0,
            sector: str = "", market_override: str = "auto", notes: str = "") -> Holding:
        """Add or update a holding. If ticker exists, update it."""
        ticker = ticker.upper().strip()
        existing = self.get(ticker)
        if existing:
            existing.shares = shares
            existing.wac = wac
            if sector:
                existing.sector = sector
            if market_override != "auto":
                existing.market_override = market_override
            if notes:
                existing.notes = notes
            return existing

        from datetime import date
        h = Holding(
            ticker=ticker, shares=shares, wac=wac,
            sector=sector, market_override=market_override,
            notes=notes, added=date.today().isoformat(),
        )
        self.holdings.append(h)
        return h

    def remove(self, ticker: str) -> bool:
        """Remove a holding by ticker. Returns True if found."""
        ticker = ticker.upper().strip()
        before = len(self.holdings)
        self.holdings = [h for h in self.holdings if h.ticker != ticker]
        return len(self.holdings) < before

    def get(self, ticker: str) -> Holding | None:
        """Find a holding by ticker."""
        ticker = ticker.upper().strip()
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    @property
    def active_holdings(self) -> list[Holding]:
        """Holdings with shares > 0."""
        return [h for h in self.holdings if not h.is_watcher]

    @property
    def watchers(self) -> list[Holding]:
        """Holdings with 0 shares (watchers)."""
        return [h for h in self.holdings if h.is_watcher]

    # ---- LRU ----

    def touch_lru(self, ticker: str) -> None:
        """Move ticker to front of LRU."""
        ticker = ticker.upper().strip()
        if ticker in self.lru:
            self.lru.remove(ticker)
        self.lru.insert(0, ticker)
        self.lru = self.lru[: self.lru_max]

    # ---- Persistence ----

    def save(self, path: Path | str | None = None) -> None:
        """Save portfolio to JSON."""
        path = Path(path) if path else _DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "holdings": [asdict(h) for h in self.holdings],
            "lru": self.lru,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Portfolio":
        """Load portfolio from JSON. Returns empty portfolio if file missing."""
        path = Path(path) if path else _DEFAULT_PATH
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            holdings = [Holding(**h) for h in raw.get("holdings", [])]
            lru = raw.get("lru", [])
            return cls(holdings=holdings, lru=lru)
        except (json.JSONDecodeError, TypeError, KeyError):
            return cls()
