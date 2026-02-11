"""OHLCV loading: Yahoo Finance, CSV, and synthetic data for backtesting.

Data format: pandas DataFrame with columns Open, High, Low, Close, Volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_ohlcv_yahoo(
    symbol: str = "SPY",
    period: str = "1y",
) -> pd.DataFrame:
    """
    Load OHLCV history from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. SPY, AAPL).
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.

    Returns:
        DataFrame with DatetimeIndex and columns Open, High, Low, Close, Volume.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    dataframe = ticker.history(period=period)
    if dataframe.empty:
        raise ValueError(f"No data returned for {symbol} (period={period})")
    # Ensure standard column names (yfinance uses Capital case)
    return dataframe[["Open", "High", "Low", "Close", "Volume"]].copy()


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV from a CSV file.

    Expected columns: Date (or datetime), Open, High, Low, Close, Volume.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {path}")

    dataframe = pd.read_csv(path)
    # Normalize column names to match yfinance
    col_map = {c.lower(): c for c in dataframe.columns}
    for target in ["open", "high", "low", "close", "volume"]:
        if target in col_map and col_map[target] != target.capitalize():
            dataframe = dataframe.rename(columns={col_map[target]: target.capitalize()})

    date_col = next(
        (c for c in dataframe.columns if c.lower() in ("date", "datetime")),
        None,
    )
    if date_col:
        dataframe[date_col] = pd.to_datetime(dataframe[date_col])
        dataframe = dataframe.set_index(date_col)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [r for r in required if r not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return dataframe[required].copy()


def generate_synthetic_ohlcv(
    days: int = 252,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV for testing when no market data is available.

    Uses a simple random walk for closes; volumes and OHLC are derived.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")

    # Random walk for close
    returns = rng.normal(0.0005, 0.015, size=days)
    close = 100 * np.exp(np.cumsum(returns))

    # Simple OHLC from close
    high = close * (1 + np.abs(rng.normal(0, 0.01, size=days)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, size=days)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    volume = (rng.uniform(500_000, 2_000_000, size=days)).astype(int)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
