@echo off
call "%~dp0.venv\Scripts\activate.bat"
setlocal
set "EXTRA_ARGS="
for %%a in (%*) do (
    if "%%a"=="--portfolio" set "HAS_PORTFOLIO=1" else (
        if defined HAS_PORTFOLIO (
            set "EXTRA_ARGS=%EXTRA_ARGS% %%a"
            set "HAS_PORTFOLIO="
        ) else set "EXTRA_ARGS=%EXTRA_ARGS% %%a"
    )
)
python "%~dp0finsell\dashboard_v2.py" --portfolio=""L:\FinSell\data\sample\portfolio.json"" %EXTRA_ARGS%


rem examples:
rem dashboard.bat
rem dashboard.bat --ticker NVDA
rem dashboard.bat -t NVDA -s "Technology" --port 9000
rem dashboard.bat --help

rem Modes: Holdings (default tab), Research, Compare
rem Holdings are stored in data/portfolio.json
rem --port	-p	Dashboard port (default 8050)

rem # Sectors
rem "Technology": "XLK",
rem "Financial Services": "XLF",
rem "Healthcare": "XLV",
rem "Consumer Cyclical": "XLY",
rem "Consumer Defensive": "XLP",
rem "Industrials": "XLI",
rem "Energy": "XLE",
rem "Utilities": "XLU",
rem "Real Estate": "XLRE",
rem "Communication Services": "XLC",
rem "Basic Materials": "XLB",
rem "SP500": "SPY",     # SPDR S&P 500 ETF
rem "DJI": "DIA",       # SPDR Dow Jones Industrial Average ETF
rem "TSX": "^GSPTSE",    # S&P/TSX Composite Index
rem "NASDAQ": "^IXIC",    # Nasdaq Composite Index
