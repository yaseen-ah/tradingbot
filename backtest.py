"""
backtest.py — Walk-forward backtester for the order flow strategy.

Simulates the exact same BOS / Order Block / FVG / Liquidity Sweep signals
used by bot.py (via signals.py) against multi-year historical OHLCV data,
bar by bar, with no lookahead: a signal formed on bars up to bar i-1 is
executed at bar i's open.

Usage:
    python backtest.py --pairs GBPUSD=X EURUSD=X --start 2021-01-01 --end 2026-01-01
    python backtest.py --pairs GBPUSD=X --years 5 --min-confidence 60
    python backtest.py --demo                      # synthetic data, no network needed

Outputs (in --outdir, default ./backtest_results):
    report.txt              summary metrics, strategy vs buy-and-hold
    trades_<PAIR>.csv        every simulated trade
    equity_curve.png         equity curves + drawdown chart
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from signals import (
    fetch_data,
    evaluate_window,
)

# Annualization factor (periods/year) by yfinance interval, assuming FX trades ~24h.
PERIODS_PER_YEAR = {
    "1m": 252 * 24 * 60, "2m": 252 * 24 * 30, "5m": 252 * 24 * 12,
    "15m": 252 * 24 * 4, "30m": 252 * 24 * 2, "60m": 252 * 24, "1h": 252 * 24,
    "1d": 252, "5d": 252 / 5, "1wk": 52, "1mo": 12,
}


def pip_size_for(pair: str) -> float:
    """Rough pip size heuristic for spread-cost modelling."""
    p = pair.upper()
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "XAG" in p or p.startswith("GC=") or p.startswith("SI="):
        return 0.01
    return 0.0001


# ─────────────────────────────────────────────
#  SYNTHETIC DATA (for --demo / offline smoke-testing)
# ─────────────────────────────────────────────

def synthetic_ohlcv(n: int = 1500, start_price: float = 1.2500, seed: int = 7,
                     start_date: str = "2021-01-01", freq: str = "1D") -> pd.DataFrame:
    """Generate a plausible daily FX-like OHLCV series via a random walk with drift."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.00005, scale=0.006, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0025, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0025, n)))
    vol = rng.integers(1000, 5000, n)
    idx = pd.date_range(start=start_date, periods=n, freq=freq)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


# ─────────────────────────────────────────────
#  WALK-FORWARD SIMULATION
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, pair: str, interval: str, initial_capital: float,
                  risk_per_trade: float, atr_sl_mult: float, atr_tp_mult: float,
                  min_confidence: int, window: int, swing_lookback: int, sweep_lookback: int,
                  max_holding_bars: int, spread_pips: float) -> dict:
    """
    Bar-by-bar walk-forward simulation for a single pair.
    Returns dict with 'trades' (list), 'equity_curve' (Series), 'buyhold_curve' (Series).
    """
    pip = pip_size_for(pair)
    spread_cost_price = spread_pips * pip

    capital = initial_capital
    position = None  # dict while a trade is open
    trades = []
    equity_dates, equity_vals = [], []

    start_i = max(window, 30)
    for i in range(start_i, len(df)):
        bar = df.iloc[i]
        date = df.index[i]

        if position is not None:
            direction = position["direction"]
            sl, tp = position["sl"], position["tp"]
            hit_sl = (direction == "BUY" and bar["Low"] <= sl) or (direction == "SELL" and bar["High"] >= sl)
            hit_tp = (direction == "BUY" and bar["High"] >= tp) or (direction == "SELL" and bar["Low"] <= tp)
            position["bars_held"] += 1
            timed_out = position["bars_held"] >= max_holding_bars

            exit_price, reason = None, None
            if hit_sl:
                exit_price, reason = sl, "SL"
            elif hit_tp:
                exit_price, reason = tp, "TP"
            elif timed_out:
                exit_price, reason = float(bar["Close"]), "Timeout"

            if exit_price is not None:
                units = position["units"]
                raw_pnl = units * (exit_price - position["entry_price"]) if direction == "BUY" \
                    else units * (position["entry_price"] - exit_price)
                commission = units * spread_cost_price
                pnl = raw_pnl - commission
                capital += pnl
                r_multiple = pnl / position["risk_dollars"] if position["risk_dollars"] > 0 else 0.0
                trades.append({
                    "pair": pair, "direction": direction,
                    "entry_time": position["entry_time"], "entry_price": position["entry_price"],
                    "exit_time": date, "exit_price": exit_price, "reason": reason,
                    "confidence": position["confidence"], "pnl": pnl, "r_multiple": r_multiple,
                })
                position = None
            else:
                unrealized = position["units"] * (bar["Close"] - position["entry_price"]) if direction == "BUY" \
                    else position["units"] * (position["entry_price"] - bar["Close"])
                equity_dates.append(date)
                equity_vals.append(capital + unrealized)
                continue

        if position is None:
            if i - window < 0:
                equity_dates.append(date)
                equity_vals.append(capital)
                continue
            window_slice = df.iloc[i - window:i]  # up to (not including) today -> no lookahead
            result = evaluate_window(window_slice, swing_lookback=swing_lookback, sweep_lookback=sweep_lookback)
            if result["bos"] and result["confidence"] >= min_confidence and result["atr"] > 0 and not np.isnan(result["atr"]):
                direction = "BUY" if result["bos"] == "bullish" else "SELL"
                entry_price = float(bar["Open"])
                atr = result["atr"]
                if direction == "BUY":
                    sl = entry_price - atr * atr_sl_mult
                    tp = entry_price + atr * atr_tp_mult
                else:
                    sl = entry_price + atr * atr_sl_mult
                    tp = entry_price - atr * atr_tp_mult
                risk_per_unit = abs(entry_price - sl)
                if risk_per_unit > 0:
                    risk_dollars = capital * risk_per_trade
                    units = risk_dollars / risk_per_unit
                    position = {
                        "direction": direction, "entry_price": entry_price, "entry_time": date,
                        "sl": sl, "tp": tp, "units": units, "risk_dollars": risk_dollars,
                        "bars_held": 0, "confidence": result["confidence"],
                    }

        equity_dates.append(date)
        equity_vals.append(capital)

    # Close any still-open position at the final bar
    if position is not None:
        last_close = float(df["Close"].iloc[-1])
        direction = position["direction"]
        units = position["units"]
        raw_pnl = units * (last_close - position["entry_price"]) if direction == "BUY" \
            else units * (position["entry_price"] - last_close)
        commission = units * spread_cost_price
        pnl = raw_pnl - commission
        capital += pnl
        r_multiple = pnl / position["risk_dollars"] if position["risk_dollars"] > 0 else 0.0
        trades.append({
            "pair": pair, "direction": direction, "entry_time": position["entry_time"],
            "entry_price": position["entry_price"], "exit_time": df.index[-1], "exit_price": last_close,
            "reason": "EndOfData", "confidence": position["confidence"], "pnl": pnl, "r_multiple": r_multiple,
        })
        equity_vals[-1] = capital

    equity_curve = pd.Series(equity_vals, index=pd.Index(equity_dates), name="equity")
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

    bh_start_price = float(df["Close"].iloc[start_i])
    buyhold_curve = initial_capital * (df["Close"].iloc[start_i:] / bh_start_price)
    buyhold_curve.name = "buy_and_hold"

    return {"trades": trades, "equity_curve": equity_curve, "buyhold_curve": buyhold_curve}


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_metrics(equity: pd.Series, trades: list, interval: str, initial_capital: float) -> dict:
    if len(equity) < 2:
        return {"error": "Not enough data to compute metrics."}

    rets = equity.pct_change().dropna()
    periods_per_year = PERIODS_PER_YEAR.get(interval, 252)

    total_return_pct = (equity.iloc[-1] / initial_capital - 1) * 100
    n_years = max(len(equity) / periods_per_year, 1e-9)
    cagr_pct = ((equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100 if equity.iloc[-1] > 0 else float("nan")

    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = float("nan")

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else float("nan")
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else float("nan")
    avg_r = np.mean([t["r_multiple"] for t in trades]) if trades else float("nan")
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0.0
    rrr_realized = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    return {
        "total_return_pct": total_return_pct, "cagr_pct": cagr_pct, "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct, "num_trades": len(trades), "win_rate_pct": win_rate,
        "profit_factor": profit_factor, "avg_r_multiple": avg_r, "avg_realized_rrr": rrr_realized,
        "final_equity": equity.iloc[-1],
    }


def fmt(v, suffix="", decimals=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) if not np.isinf(v) else False)):
        return "n/a"
    if isinstance(v, float) and np.isinf(v):
        return "∞"
    return f"{v:,.{decimals}f}{suffix}"


# ─────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────

def build_report(pair_results: dict, args) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  ORDER FLOW STRATEGY — BACKTEST REPORT")
    lines.append("=" * 72)
    lines.append(f"Pairs: {', '.join(pair_results.keys())}")
    lines.append(f"Interval: {args.interval}   Window: {args.window} bars   "
                 f"Min confidence: {args.min_confidence}%")
    lines.append(f"ATR SL/TP multipliers: {args.atr_sl_mult} / {args.atr_tp_mult}   "
                 f"Risk per trade: {args.risk_per_trade * 100:.1f}%   Spread: {args.spread_pips} pips")
    lines.append(f"Starting capital per pair: ${args.capital:,.0f}")
    lines.append("")

    for pair, res in pair_results.items():
        m = res["strategy_metrics"]
        bh = res["buyhold_metrics"]
        lines.append("-" * 72)
        lines.append(f"  {pair}")
        lines.append("-" * 72)
        lines.append(f"{'Metric':<28}{'Strategy':>20}{'Buy & Hold':>20}")
        lines.append(f"{'Total return':<28}{fmt(m['total_return_pct'], '%'):>20}{fmt(bh['total_return_pct'], '%'):>20}")
        lines.append(f"{'CAGR':<28}{fmt(m['cagr_pct'], '%'):>20}{fmt(bh['cagr_pct'], '%'):>20}")
        lines.append(f"{'Sharpe ratio':<28}{fmt(m['sharpe']):>20}{fmt(bh['sharpe']):>20}")
        lines.append(f"{'Max drawdown':<28}{fmt(m['max_drawdown_pct'], '%'):>20}{fmt(bh['max_drawdown_pct'], '%'):>20}")
        lines.append(f"{'Final equity':<28}{fmt(m['final_equity'], '', 0):>20}{fmt(bh['final_equity'], '', 0):>20}")
        lines.append("")
        lines.append(f"{'Trades':<28}{m['num_trades']:>20}")
        lines.append(f"{'Win rate':<28}{fmt(m['win_rate_pct'], '%'):>20}")
        lines.append(f"{'Profit factor':<28}{fmt(m['profit_factor']):>20}")
        lines.append(f"{'Avg R-multiple':<28}{fmt(m['avg_r_multiple']):>20}")
        lines.append(f"{'Realized avg win:loss (RRR)':<28}{fmt(m['avg_realized_rrr']):>20}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("Note: backtest results are hypothetical, ignore real-world slippage")
    lines.append("beyond the modeled spread, and past performance does not indicate")
    lines.append("future results. Not financial advice.")
    lines.append("=" * 72)
    return "\n".join(lines)


def plot_equity(pair_results: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pair_results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False)

    for row, (pair, res) in enumerate(pair_results.items()):
        eq = res["equity_curve"]
        bh = res["buyhold_curve"]
        ax1, ax2 = axes[row][0], axes[row][1]

        ax1.plot(eq.index, eq.values, label="Strategy", color="#1f77b4", linewidth=1.4)
        ax1.plot(bh.index, bh.values, label="Buy & Hold", color="#888888", linewidth=1.2, linestyle="--")
        ax1.set_title(f"{pair} — Equity Curve")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25)

        running_max = eq.cummax()
        dd = (eq - running_max) / running_max * 100
        ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
        ax2.set_title(f"{pair} — Strategy Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(outdir, "equity_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Backtest the order flow strategy on historical data.")
    p.add_argument("--pairs", nargs="+", default=["GBPUSD=X", "EURUSD=X"], help="yfinance tickers")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--years", type=float, default=5, help="Used if --start not given: lookback in years")
    p.add_argument("--interval", type=str, default="1d",
                    help="yfinance interval. Use 1d for multi-year backtests (Yahoo limits intraday history).")
    p.add_argument("--capital", type=float, default=10000, help="Starting capital per pair")
    p.add_argument("--risk-per-trade", type=float, default=0.01, help="Fraction of capital risked per trade")
    p.add_argument("--atr-sl-mult", type=float, default=1.5)
    p.add_argument("--atr-tp-mult", type=float, default=3.0)
    p.add_argument("--min-confidence", type=int, default=60)
    p.add_argument("--window", type=int, default=100, help="Bars of history used for each signal check")
    p.add_argument("--swing-lookback", type=int, default=5)
    p.add_argument("--sweep-lookback", type=int, default=20)
    p.add_argument("--max-holding-bars", type=int, default=50)
    p.add_argument("--spread-pips", type=float, default=1.5)
    p.add_argument("--outdir", type=str, default="./backtest_results")
    p.add_argument("--demo", action="store_true", help="Use synthetic data instead of yfinance (offline smoke test)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.start is None and not args.demo:
        end_dt = pd.Timestamp.today().normalize()
        start_dt = end_dt - pd.DateOffset(years=args.years)
        args.start = start_dt.strftime("%Y-%m-%d")
        args.end = args.end or end_dt.strftime("%Y-%m-%d")

    pair_results = {}
    for pair in args.pairs:
        print(f"Fetching {pair} ({'synthetic demo data' if args.demo else f'{args.start} to {args.end}'})...")
        try:
            if args.demo:
                df = synthetic_ohlcv(n=1500, seed=abs(hash(pair)) % 1000)
            else:
                df = fetch_data(pair, interval=args.interval, start=args.start, end=args.end)
        except Exception as e:
            print(f"  Skipping {pair}: {e}")
            continue

        if len(df) < args.window + 30:
            print(f"  Skipping {pair}: only {len(df)} bars returned, need at least {args.window + 30}.")
            continue

        result = run_backtest(
            df, pair, args.interval, args.capital, args.risk_per_trade,
            args.atr_sl_mult, args.atr_tp_mult, args.min_confidence, args.window,
            args.swing_lookback, args.sweep_lookback, args.max_holding_bars, args.spread_pips,
        )
        strat_metrics = compute_metrics(result["equity_curve"], result["trades"], args.interval, args.capital)
        bh_metrics = compute_metrics(result["buyhold_curve"], [], args.interval, args.capital)

        pair_results[pair] = {
            "equity_curve": result["equity_curve"], "buyhold_curve": result["buyhold_curve"],
            "trades": result["trades"], "strategy_metrics": strat_metrics, "buyhold_metrics": bh_metrics,
        }

        trades_df = pd.DataFrame(result["trades"])
        trades_path = os.path.join(args.outdir, f"trades_{pair.replace('=', '_')}.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"  {len(result['trades'])} trades simulated -> {trades_path}")

    if not pair_results:
        print("No pairs produced results. Check tickers / date range / network access.")
        sys.exit(1)

    report = build_report(pair_results, args)
    print("\n" + report)
    report_path = os.path.join(args.outdir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    chart_path = plot_equity(pair_results, args.outdir)
    print(f"\nSaved: {report_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
