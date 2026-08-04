"""
config.py — Edit this file to configure your bot.
No other files need to be touched for basic setup.
"""

CONFIG = {

    # ── Discord ───────────────────────────────────────────────────────────
    # Create a webhook: Discord channel → Edit → Integrations → Webhooks → New Webhook → Copy URL
    "discord_webhook_url": "https://discord.com/api/webhooks/1514606939630206996/5hIKWIrcqm8ecPlI_MSRO8cUfqVTYZyRHMZAPx0jL4WxLbq1OKOBgv4nuvVO_uswkICh",

    # ── Pairs to scan ─────────────────────────────────────────────────────
    # yfinance Forex format: "EURUSD=X", "GBPUSD=X", "USDJPY=X", etc.
    "pairs": [
        "GBPJPY=X",
        "XAUUSD=X",
        "GBPUSD=X",
    ],

    # ── Timeframe ─────────────────────────────────────────────────────────
    # interval options: "1m", "5m", "15m", "30m", "1h", "1d"
    # period options:   "1d", "2d", "5d", "1mo"
    # Note: yfinance only provides 1m data for the last 7 days.
    "interval": "5m",
    "period": "5d",

    # ── Scan frequency ────────────────────────────────────────────────────
    # How often the bot re-scans all pairs (in seconds).
    # Recommended: 300 (5 min) for 15m charts, 60 for 5m charts.
    "scan_interval_seconds": 60,

    # ── Signal filtering ──────────────────────────────────────────────────
    # Minimum confluence confidence score to send an alert (0–100).
    # 60 = at least 2 confluent signals needed.
    # 80 = high-quality setups only (BOS + OB + FVG or sweep).
    "min_confidence": 60,

    # ── Risk management (ATR-based) ───────────────────────────────────────
    # Stop Loss  = entry ± (ATR × atr_sl_multiplier)
    # Take Profit = entry ± (ATR × atr_tp_multiplier)
    # Default gives roughly 1:2 RRR. Adjust to your preference.
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,

}
