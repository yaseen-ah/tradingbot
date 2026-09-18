"""
signals.py — Order flow signal detection logic.

This is the single source of truth for the strategy rules (BOS, Order Blocks,
FVGs, liquidity sweeps, confidence scoring, ATR). Both bot.py (live Discord
alerts) and backtest.py (historical backtesting) import from here, so the
backtest is guaranteed to be testing the exact same logic that runs live.
"""

import pandas as pd
import numpy as np
import yfinance as yf


# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────

def fetch_data(pair: str, interval: str = "5m", period: str = "2d",
                start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Use `period` (e.g. "5d", "2y") for a rolling window ending today, or
    `start`/`end` (e.g. "2021-01-01") for a fixed historical range — used by
    the backtester. yfinance intraday granularity is limited by Yahoo itself:
    1m ~ last 7 days, up to 90m ~ last 60 days, 1h ~ last 730 days.
    Daily ("1d") and above have multi-year history, which is why the
    backtester defaults to daily bars.
    """
    ticker = yf.Ticker(pair)
    if start or end:
        df = ticker.history(start=start, end=end, interval=interval)
    else:
        df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {pair}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
#  ORDER FLOW STRATEGY LOGIC
# ─────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Identify swing highs and swing lows."""
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(lookback, len(df) - lookback):
        window_high = df["High"].iloc[i - lookback:i + lookback + 1]
        window_low = df["Low"].iloc[i - lookback:i + lookback + 1]
        if df["High"].iloc[i] == window_high.max():
            df.at[df.index[i], "swing_high"] = True
        if df["Low"].iloc[i] == window_low.min():
            df.at[df.index[i], "swing_low"] = True
    return df


def detect_break_of_structure(df: pd.DataFrame) -> str | None:
    """
    Break of Structure (BOS): price breaks above last swing high (bullish)
    or below last swing low (bearish).
    """
    highs = df[df["swing_high"]]["High"]
    lows = df[df["swing_low"]]["Low"]
    if highs.empty or lows.empty:
        return None

    last_swing_high = highs.iloc[-1]
    last_swing_low = lows.iloc[-1]
    last_close = df["Close"].iloc[-1]

    if last_close > last_swing_high:
        return "bullish"
    if last_close < last_swing_low:
        return "bearish"
    return None


def detect_order_blocks(df: pd.DataFrame, bos_direction: str) -> dict | None:
    """Last opposing candle before the BOS — institutional entry zone."""
    if bos_direction == "bullish":
        bearish = df[df["Close"] < df["Open"]]
        if bearish.empty:
            return None
        ob = bearish.iloc[-1]
        return {"type": "demand", "top": float(ob["High"]), "bottom": float(ob["Low"]), "time": str(ob.name)}
    elif bos_direction == "bearish":
        bullish = df[df["Close"] > df["Open"]]
        if bullish.empty:
            return None
        ob = bullish.iloc[-1]
        return {"type": "supply", "top": float(ob["High"]), "bottom": float(ob["Low"]), "time": str(ob.name)}
    return None


def detect_fair_value_gap(df: pd.DataFrame) -> dict | None:
    """3-candle imbalance: candle 1 high < candle 3 low (bullish) or reverse (bearish)."""
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if c1["High"] < c3["Low"]:
        return {"type": "bullish", "top": float(c3["Low"]), "bottom": float(c1["High"])}
    if c1["Low"] > c3["High"]:
        return {"type": "bearish", "top": float(c1["Low"]), "bottom": float(c3["High"])}
    return None


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> str | None:
    """Price briefly wicks through a recent high/low then reverses (stop-hunt)."""
    if len(df) < lookback:
        return None
    recent = df.iloc[-lookback:]
    prev_high = recent["High"].iloc[:-3].max()
    prev_low = recent["Low"].iloc[:-3].min()
    last_3 = df.iloc[-3:]

    swept_high = last_3["High"].max() > prev_high and last_3["Close"].iloc[-1] < prev_high
    swept_low = last_3["Low"].min() < prev_low and last_3["Close"].iloc[-1] > prev_low

    if swept_high:
        return "bearish_sweep"
    if swept_low:
        return "bullish_sweep"
    return None


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range for SL/TP sizing."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def calculate_confidence(bos: str, ob: dict, fvg: dict, sweep: str) -> int:
    """Score confluence of signals (0-100)."""
    score = 0
    if bos:
        score += 30
    if ob:
        score += 25
    if fvg:
        score += 25
    if sweep:
        if (bos == "bullish" and sweep == "bullish_sweep") or \
           (bos == "bearish" and sweep == "bearish_sweep"):
            score += 20
        else:
            score += 5
    return min(score, 100)


def evaluate_window(window: pd.DataFrame, swing_lookback: int = 5, sweep_lookback: int = 20) -> dict:
    """
    Run the full signal pipeline on a window of bars (the most recent bar is
    treated as "now"). Returns a dict with bos/ob/fvg/sweep/confidence/atr,
    or {"bos": None} if no BOS. This is what both live scanning and
    walk-forward backtesting call at each step.
    """
    if len(window) < max(swing_lookback * 2 + 1, sweep_lookback, 15):
        return {"bos": None}

    w = detect_swing_points(window, lookback=swing_lookback)
    bos = detect_break_of_structure(w)
    if not bos:
        return {"bos": None}

    ob = detect_order_blocks(w, bos)
    fvg = detect_fair_value_gap(w)
    sweep = detect_liquidity_sweep(w, lookback=sweep_lookback)
    confidence = calculate_confidence(bos, ob, fvg, sweep)
    atr = compute_atr(w)

    return {
        "bos": bos, "order_block": ob, "fvg": fvg,
        "liquidity_sweep": sweep, "confidence": confidence, "atr": atr,
    }
