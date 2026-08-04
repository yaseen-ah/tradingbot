# Forex Order Flow Bot

A Python bot that scans Forex pairs for high-probability order flow setups and sends rich trade alerts to Discord for manual execution.

---

## How It Works

The bot uses four confluent order flow signals:

| Signal | What it detects |
|---|---|
| **Break of Structure (BOS)** | Price breaks above a recent swing high (bullish) or below a swing low (bearish) — confirms directional bias |
| **Order Block (OB)** | The last opposing candle before the BOS — institutional entry zone to watch for a reaction |
| **Fair Value Gap (FVG)** | A 3-candle imbalance where price moved so fast it left an unfilled gap — price often returns to fill it |
| **Liquidity Sweep** | Price briefly wicks through a recent high/low then reverses — stop-hunt by institutions before the real move |

Each signal adds to a **confidence score (0–100%)**. Only setups above your configured threshold trigger a Discord alert.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create a Discord Webhook
1. Open the Discord channel you want alerts in
2. Click ⚙️ Edit Channel → Integrations → Webhooks → New Webhook
3. Copy the URL

### 3. Configure the bot
Open `config.py` and fill in:
```python
"discord_webhook_url": "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN",
```

Adjust other settings as needed:
- **`pairs`** — Forex pairs to scan (yfinance format: `"EURUSD=X"`)
- **`interval`** — Candle timeframe (`"5m"`, `"15m"`, `"1h"`, etc.)
- **`min_confidence`** — Alert threshold (60 = 2 signals, 80 = 3+ signals)
- **`atr_sl_multiplier`** / **`atr_tp_multiplier`** — Adjust SL/TP distance

### 4. Run the bot
```bash
python bot.py
```

---

## Discord Alert Format

Each alert includes:
- 🟢/🔴 Pair name and direction (BUY/SELL)
- Confidence score with visual bar
- Entry price, Stop Loss, Take Profit
- Risk/Reward ratio
- ATR value
- Confluence breakdown (which signals fired)
- ASCII price chart of last 40 candles

---

## Running 24/7

**On a VPS / cloud server:**
```bash
# Using screen (keeps running after SSH disconnect)
screen -S forex_bot
python bot.py
# Press Ctrl+A then D to detach

# Or use systemd / PM2 / Docker for production
```

**On Windows (scheduled):**
Use Task Scheduler to run `python bot.py` at startup.

---

## Customising the Strategy

All signal logic is in `bot.py` in clearly separated functions:

| Function | What to modify |
|---|---|
| `detect_swing_points()` | Change `lookback` for more/fewer swing points |
| `detect_break_of_structure()` | Tighten or loosen BOS conditions |
| `detect_order_blocks()` | Refine OB selection logic |
| `detect_fair_value_gap()` | Adjust gap size minimum |
| `detect_liquidity_sweep()` | Change sweep lookback window |
| `calculate_confidence()` | Re-weight the scoring system |

---

## Data Source

Uses **yfinance** (free, ~15 min delayed during market hours for Forex).  
For live real-time data, consider upgrading to:
- **OANDA API** (free Forex, real-time)
- **Polygon.io** (paid, institutional grade)
- **Interactive Brokers API** (if you trade there)

---

