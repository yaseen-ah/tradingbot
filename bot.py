"""
Forex Order Flow Trading Bot
Analyses market structure, detects order flow signals, and sends trade alerts to Discord.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timezone
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────

def fetch_data(pair: str, interval: str = "5m", period: str = "2d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance for a Forex pair."""
    ticker = yf.Ticker(pair)
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
    Returns 'bullish', 'bearish', or None.
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
    """
    Order Block: the last down-candle before a bullish BOS (demand OB)
    or last up-candle before a bearish BOS (supply OB).
    Returns OB zone dict or None.
    """
    if bos_direction == "bullish":
        # Find last bearish candle before current swing
        bearish = df[df["Close"] < df["Open"]]
        if bearish.empty:
            return None
        ob = bearish.iloc[-1]
        return {
            "type": "demand",
            "top": float(ob["High"]),
            "bottom": float(ob["Low"]),
            "time": str(ob.name),
        }
    elif bos_direction == "bearish":
        bullish = df[df["Close"] > df["Open"]]
        if bullish.empty:
            return None
        ob = bullish.iloc[-1]
        return {
            "type": "supply",
            "top": float(ob["High"]),
            "bottom": float(ob["Low"]),
            "time": str(ob.name),
        }
    return None


def detect_fair_value_gap(df: pd.DataFrame) -> dict | None:
    """
    FVG: a 3-candle pattern where candle 1's high < candle 3's low (bullish FVG)
    or candle 1's low > candle 3's high (bearish FVG).
    """
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    if c1["High"] < c3["Low"]:
        return {
            "type": "bullish",
            "top": float(c3["Low"]),
            "bottom": float(c1["High"]),
        }
    if c1["Low"] > c3["High"]:
        return {
            "type": "bearish",
            "top": float(c1["Low"]),
            "bottom": float(c3["High"]),
        }
    return None


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> str | None:
    """
    Liquidity sweep: price briefly breaks a recent high/low then reverses.
    Indicates stop-hunt before institutional move.
    """
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
    """
    Score confluence of signals. Each confirmed factor adds weight.
    Returns 0–100.
    """
    score = 0
    if bos:
        score += 30
    if ob:
        score += 25
    if fvg:
        score += 25
    if sweep:
        # Sweep in same direction as BOS adds more weight
        if (bos == "bullish" and sweep == "bullish_sweep") or \
           (bos == "bearish" and sweep == "bearish_sweep"):
            score += 20
        else:
            score += 5
    return min(score, 100)


def analyse_pair(pair: str) -> dict | None:
    """
    Full order flow analysis pipeline for a single pair.
    Returns a signal dict if a trade setup is found, else None.
    """
    try:
        df = fetch_data(pair, interval=CONFIG["interval"], period=CONFIG["period"])
    except Exception as e:
        log.warning(f"Failed to fetch {pair}: {e}")
        return None

    df = detect_swing_points(df)
    bos = detect_break_of_structure(df)
    if not bos:
        log.info(f"{pair}: No BOS detected.")
        return None

    ob = detect_order_blocks(df, bos)
    fvg = detect_fair_value_gap(df)
    sweep = detect_liquidity_sweep(df)
    confidence = calculate_confidence(bos, ob, fvg, sweep)

    if confidence < CONFIG["min_confidence"]:
        log.info(f"{pair}: Confidence {confidence}% below threshold.")
        return None

    atr = compute_atr(df)
    entry = float(df["Close"].iloc[-1])
    atr_multiplier_sl = CONFIG["atr_sl_multiplier"]
    atr_multiplier_tp = CONFIG["atr_tp_multiplier"]

    if bos == "bullish":
        direction = "BUY"
        stop_loss = round(entry - atr * atr_multiplier_sl, 5)
        take_profit = round(entry + atr * atr_multiplier_tp, 5)
    else:
        direction = "SELL"
        stop_loss = round(entry + atr * atr_multiplier_sl, 5)
        take_profit = round(entry - atr * atr_multiplier_tp, 5)

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rrr = round(reward / risk, 2) if risk > 0 else 0

    chart = build_ascii_chart(df, entry)

    return {
        "pair": pair,
        "direction": direction,
        "entry": round(entry, 5),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rrr": rrr,
        "confidence": confidence,
        "bos": bos,
        "order_block": ob,
        "fvg": fvg,
        "liquidity_sweep": sweep,
        "atr": round(atr, 5),
        "chart": chart,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─────────────────────────────────────────────
#  ASCII CHART
# ─────────────────────────────────────────────

def build_ascii_chart(df: pd.DataFrame, entry: float, rows: int = 12, cols: int = 40) -> str:
    """Build a simple ASCII candlestick-style price chart from recent closes."""
    recent = df["Close"].tail(cols).values
    high = recent.max()
    low = recent.min()
    spread = high - low if high != low else 1e-9

    lines = []
    for r in range(rows):
        level = high - (r / (rows - 1)) * spread
        row_str = ""
        for price in recent:
            if abs(price - level) <= spread / (rows * 1.5):
                row_str += "●"
            elif abs(entry - level) <= spread / (rows * 2):
                row_str += "─"
            else:
                row_str += " "
        label = f" {level:.5f}" if r == 0 or r == rows - 1 or r == rows // 2 else ""
        lines.append(row_str + label)

    lines.append("─" * cols)
    lines.append(f"Last {cols} candles  |  Entry: {entry:.5f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  DISCORD NOTIFICATION
# ─────────────────────────────────────────────

def build_discord_payload(signal: dict) -> dict:
    """Build a rich Discord embed from a signal dict."""
    direction = signal["direction"]
    colour = 0x00C853 if direction == "BUY" else 0xFF1744  # green / red
    conf = signal["confidence"]
    conf_bar = "█" * (conf // 10) + "░" * (10 - conf // 10)

    # Confluence summary
    confluence_lines = []
    if signal["bos"]:
        confluence_lines.append(f"✅ Break of Structure ({signal['bos'].upper()})")
    if signal["order_block"]:
        ob = signal["order_block"]
        confluence_lines.append(f"✅ {ob['type'].capitalize()} Order Block ({ob['bottom']:.5f} – {ob['top']:.5f})")
    if signal["fvg"]:
        fvg = signal["fvg"]
        confluence_lines.append(f"✅ {fvg['type'].capitalize()} Fair Value Gap ({fvg['bottom']:.5f} – {fvg['top']:.5f})")
    if signal["liquidity_sweep"]:
        confluence_lines.append(f"✅ Liquidity Sweep ({signal['liquidity_sweep'].replace('_', ' ').title()})")

    confluence_text = "\n".join(confluence_lines) if confluence_lines else "None detected"

    embed = {
        "title": f"{'🟢' if direction == 'BUY' else '🔴'} {signal['pair']}  ·  {direction}",
        "color": colour,
        "description": (
            f"**Confidence:** `{conf_bar}` {conf}%\n"
            f"**Time:** {signal['timestamp']}"
        ),
        "fields": [
            {"name": "Entry", "value": f"`{signal['entry']}`", "inline": True},
            {"name": "Stop Loss", "value": f"`{signal['stop_loss']}`", "inline": True},
            {"name": "Take Profit", "value": f"`{signal['take_profit']}`", "inline": True},
            {"name": "Risk/Reward", "value": f"`1 : {signal['rrr']}`", "inline": True},
            {"name": "ATR", "value": f"`{signal['atr']}`", "inline": True},
            {"name": "Interval", "value": f"`{CONFIG['interval']}`", "inline": True},
            {"name": "Confluence", "value": confluence_text, "inline": False},
            {"name": "📊 Price Chart (last 40 candles)", "value": f"```\n{signal['chart']}\n```", "inline": False},
        ],
        "footer": {"text": "Forex Order Flow Bot  ·  For manual review only — not financial advice."},
    }
    return {"embeds": [embed]}


def send_discord_alert(signal: dict) -> bool:
    """POST embed to Discord webhook."""
    webhook_url = CONFIG["discord_webhook_url"]
    if not webhook_url or webhook_url == "YOUR_DISCORD_WEBHOOK_URL":
        log.error("Discord webhook URL not set in config.py")
        return False

    payload = build_discord_payload(signal)
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info(f"Alert sent to Discord for {signal['pair']}.")
        return True
    except requests.RequestException as e:
        log.error(f"Discord send failed: {e}")
        return False


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run():
    log.info("🤖 Forex Order Flow Bot started.")
    log.info(f"Pairs: {CONFIG['pairs']}")
    log.info(f"Interval: {CONFIG['interval']}  |  Scan every: {CONFIG['scan_interval_seconds']}s")
    log.info(f"Min confidence: {CONFIG['min_confidence']}%")

    # Track sent signals to avoid duplicate alerts
    sent_signals: set = set()

    while True:
        for pair in CONFIG["pairs"]:
            log.info(f"Scanning {pair}...")
            signal = analyse_pair(pair)

            if signal:
                signal_key = f"{pair}_{signal['direction']}_{signal['entry']}"
                if signal_key not in sent_signals:
                    log.info(
                        f"✅ Setup found: {pair} {signal['direction']} | "
                        f"Entry {signal['entry']} | SL {signal['stop_loss']} | "
                        f"TP {signal['take_profit']} | RRR 1:{signal['rrr']} | "
                        f"Conf {signal['confidence']}%"
                    )
                    sent = send_discord_alert(signal)
                    if sent:
                        sent_signals.add(signal_key)
                        # Only keep last 100 sent signals in memory
                        if len(sent_signals) > 100:
                            sent_signals.pop()
                else:
                    log.info(f"{pair}: Signal already sent, skipping duplicate.")

        log.info(f"Scan complete. Sleeping {CONFIG['scan_interval_seconds']}s...\n")
        time.sleep(CONFIG["scan_interval_seconds"])


if __name__ == "__main__":
    run()
