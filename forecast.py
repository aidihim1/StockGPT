# forecast.py -- StockGPT dynamic forecast with market-grounded realism
#
# Three layers of realism:
#   1. Fresh data  -- always run update_data.py first so context = today's market
#   2. Vol scaling -- predictions scaled to current volatility regime (not historical avg)
#   3. Market conditioning -- each stock's forecast adjusted for Nifty direction
#
# Usage:
#   python update_data.py              # fetch today's prices first (ALWAYS do this)
#   python forecast.py                 # full market forecast: 1, 5, 20, 60 days
#   python forecast.py --days 1 5 83   # custom horizons (83 trading days = ~Aug 2026)
#   python forecast.py --stock RELIANCE # deep daily path for one stock

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn.functional as F

from model import StockGPT, StockGPTConfig

# ── Settings ──────────────────────────────────────────────────────────────────
BLOCK_SIZE       = 256
N_SAMPLES        = 200      # Monte Carlo paths per stock
TEMPERATURE      = 0.8
MIN_CONTEXT      = 50       # min non-NaN days to include a stock
VOL_LOOKBACK     = 20       # days to measure current volatility regime
BETA_LOOKBACK    = 60       # days to estimate beta to market
VOL_RATIO_CAP    = (0.3, 4.0)  # clamp vol scaling ratio to prevent extremes
MKT_PROXY_N      = 100     # top N stocks by data completeness used as market proxy
# ──────────────────────────────────────────────────────────────────────────────

BIN_EDGES   = np.linspace(-1.0, 1.0, 401)
BIN_CENTERS = np.concatenate([
    [-1.0],
    (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2,
    [1.0]
])  # shape (402,)


# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenize_series(returns: np.ndarray) -> np.ndarray:
    valid = returns[~np.isnan(returns)]
    valid = np.clip(valid, -1.0 + 1e-9, 1.0 - 1e-9)
    return np.digitize(valid, BIN_EDGES).astype(np.int16)


def compute_vol_ratio(series: np.ndarray) -> float:
    """
    Ratio of current volatility (last VOL_LOOKBACK days) to full-history volatility.
    > 1 means market is more volatile than usual → scale predictions up.
    < 1 means quieter than usual → scale predictions down.
    """
    clean = series[~np.isnan(series)]
    if len(clean) < VOL_LOOKBACK + 10:
        return 1.0
    hist_vol   = float(np.std(clean))
    recent_vol = float(np.std(clean[-VOL_LOOKBACK:]))
    if hist_vol < 1e-8:
        return 1.0
    ratio = recent_vol / hist_vol
    return float(np.clip(ratio, VOL_RATIO_CAP[0], VOL_RATIO_CAP[1]))


def compute_beta(stock_series: np.ndarray, market_series: np.ndarray) -> float:
    """
    OLS beta of stock returns vs market returns over last BETA_LOOKBACK days.
    beta > 1 = amplifies market moves; beta < 1 = dampens them.
    """
    n = BETA_LOOKBACK
    s = stock_series[-n:]
    m = market_series[-n:]

    # Align and drop NaN
    mask = ~(np.isnan(s) | np.isnan(m))
    s, m = s[mask], m[mask]

    if len(s) < 20:
        return 1.0

    var_m = np.var(m)
    if var_m < 1e-10:
        return 1.0

    return float(np.cov(s, m)[0, 1] / var_m)


def trading_days_to_date(n: int) -> str:
    date = datetime.now()
    count = 0
    while count < n:
        date += timedelta(days=1)
        if date.weekday() < 5:
            count += 1
    return date.strftime("%Y-%m-%d")


def load_model(checkpoint_path: str, device: torch.device) -> StockGPT:
    print(f"Loading model from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = StockGPTConfig(**ckpt["config"])
    model = StockGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded (trained to step {ckpt['step']})")
    return model


# ── Core forecast engine ───────────────────────────────────────────────────────

@torch.no_grad()
def run_monte_carlo_paths(
    model: StockGPT,
    context_tokens: np.ndarray,
    total_days: int,
    n_samples: int,
    device: torch.device,
    vol_ratio: float = 1.0,
) -> np.ndarray:
    """
    Run N Monte Carlo paths in parallel (fully batched on GPU).
    Pre-allocated fixed buffer — no repeated memory allocation per step.
    """
    base_ctx = context_tokens[-BLOCK_SIZE:]
    ctx_len = len(base_ctx)

    # Pre-allocate full buffer: (n_samples, ctx_len + total_days)
    buf = torch.zeros((n_samples, ctx_len + total_days), dtype=torch.long, device=device)
    buf[:, :ctx_len] = torch.tensor(base_ctx, dtype=torch.long, device=device).unsqueeze(0)

    bin_centers = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=device)
    all_returns = np.zeros((n_samples, total_days), dtype=np.float32)

    for day in range(total_days):
        end = ctx_len + day
        inp = buf[:, max(0, end - BLOCK_SIZE):end]       # (n_samples, BLOCK_SIZE)
        logits, _ = model(inp)                            # (n_samples, seq, vocab)
        last_logits = logits[:, -1, :]                    # (n_samples, vocab)
        probs = F.softmax(last_logits / TEMPERATURE, dim=-1)

        next_toks = torch.multinomial(probs, 1).squeeze(1)  # (n_samples,)
        buf[:, end] = next_toks                              # write into pre-allocated slot

        raw_returns = bin_centers[next_toks]                 # (n_samples,)
        all_returns[:, day] = (raw_returns * vol_ratio).cpu().numpy()

    return all_returns  # (n_samples, total_days)


def compute_horizon_metrics(paths: np.ndarray, horizons: list) -> dict:
    results = {}
    for h in horizons:
        if h > paths.shape[1]:
            continue
        cum = np.expm1(np.sum(np.log1p(np.clip(paths[:, :h], -0.9999, None)), axis=1))
        results[h] = {
            "mean_pct":      round(float(np.mean(cum)) * 100, 3),
            "median_pct":    round(float(np.median(cum)) * 100, 3),
            "std_pct":       round(float(np.std(cum)) * 100, 3),
            "p10_pct":       round(float(np.percentile(cum, 10)) * 100, 3),
            "p90_pct":       round(float(np.percentile(cum, 90)) * 100, 3),
            "prob_up_pct":   round(float(np.mean(cum > 0)) * 100, 1),
        }
    return results


# ── Market proxy (Nifty-like) ─────────────────────────────────────────────────

def build_market_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    Build a market proxy return series = equal-weighted median of top N stocks
    by data completeness. This is our Nifty-like benchmark.
    """
    completeness = df.notna().sum()
    top_stocks   = completeness.nlargest(MKT_PROXY_N).index
    proxy        = df[top_stocks].median(axis=1).values   # median = robust to outliers
    return proxy


# ── Main ──────────────────────────────────────────────────────────────────────

def run_forecast(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    print(f"\nLoading {args.data} ...")
    df_long = pd.read_csv(args.data, parse_dates=["date"])
    last_date = df_long["date"].max()
    print(f"  Rows       : {len(df_long):,}")
    print(f"  Stocks     : {df_long['stock'].nunique()}")
    print(f"  Last date  : {last_date.date()}  <-- this is what the model sees as 'today'")

    today_actual = datetime.now().date()
    data_lag = (today_actual - last_date.date()).days
    if data_lag > 3:
        print(f"\n  WARNING: Data is {data_lag} days old! Run update_data.py first for accurate forecasts.\n")

    # Convert long -> wide for internal use (market proxy + per-stock series)
    df = df_long.pivot(index="date", columns="stock", values="return_1d")

    horizons    = sorted(args.days)
    max_horizon = max(horizons)

    print(f"\nHorizons: {horizons} trading days")
    for h in horizons:
        print(f"  {h:3d}d -> approx {trading_days_to_date(h)}")

    # ── Market proxy ────────────────────────────────────────────────────────────
    print("\nBuilding market proxy (Nifty-like benchmark) ...")
    market_series = build_market_proxy(df)

    # Forecast the market itself (used to adjust stock-level forecasts)
    mkt_tokens     = tokenize_series(market_series)
    mkt_vol_ratio  = compute_vol_ratio(market_series)
    print(f"  Market vol ratio (recent vs historical): {mkt_vol_ratio:.2f}x")

    mkt_paths      = run_monte_carlo_paths(
        model, mkt_tokens, max_horizon, args.n_samples, device, vol_ratio=mkt_vol_ratio
    )
    mkt_metrics    = compute_horizon_metrics(mkt_paths, horizons)

    print("  Market proxy expected returns:")
    for h, m in mkt_metrics.items():
        print(f"    {h:3d}d: {m['mean_pct']:+.2f}%  (prob up: {m['prob_up_pct']}%)")

    # ── Single stock deep forecast ───────────────────────────────────────────────
    if args.stock:
        stock = args.stock.upper()
        if stock not in df.columns:
            print(f"\nERROR: {stock} not found.")
            return

        print(f"\n=== Deep forecast for {stock} ===")
        series    = df[stock].values
        tokens    = tokenize_series(series)
        vol_ratio = compute_vol_ratio(series)
        beta      = compute_beta(series, market_series)
        print(f"  History       : {len(tokens)} trading days")
        print(f"  Vol ratio     : {vol_ratio:.2f}x  (current vol vs historical)")
        print(f"  Market beta   : {beta:.2f}")

        paths   = run_monte_carlo_paths(model, tokens, max_horizon, args.n_samples, device, vol_ratio)
        metrics = compute_horizon_metrics(paths, horizons)

        rows = []
        for h, m in metrics.items():
            mkt_exp = mkt_metrics.get(h, {}).get("mean_pct", 0)
            # Market-conditioned adjustment:
            # model already captures some market signal via token patterns,
            # but we reinforce: shift idiosyncratic component by beta * mkt_delta
            # where mkt_delta = actual mkt forecast - neutral (0%)
            mkt_adjustment = beta * mkt_exp  # in % terms
            adj_mean = round(m["mean_pct"] + mkt_adjustment * 0.5, 3)  # 50% weight to adjustment
            rows.append({
                "horizon_days":    h,
                "approx_date":     trading_days_to_date(h),
                "raw_mean_pct":    m["mean_pct"],
                "market_adj_pct":  adj_mean,
                "median_pct":      m["median_pct"],
                "std_pct":         m["std_pct"],
                "p10_pct":         m["p10_pct"],
                "p90_pct":         m["p90_pct"],
                "prob_up_pct":     m["prob_up_pct"],
                "vol_ratio":       round(vol_ratio, 2),
                "beta":            round(beta, 2),
                "mkt_exp_pct":     mkt_exp,
            })

        result_df = pd.DataFrame(rows)
        print(result_df.to_string(index=False))

        # Save daily path
        daily_med  = np.median(paths, axis=0)
        daily_p25  = np.percentile(paths, 25, axis=0)
        daily_p75  = np.percentile(paths, 75, axis=0)
        cum_med    = np.expm1(np.cumsum(np.log1p(np.clip(daily_med, -0.9999, None)))) * 100
        path_df = pd.DataFrame({
            "day":           range(1, max_horizon + 1),
            "approx_date":   [trading_days_to_date(d) for d in range(1, max_horizon + 1)],
            "daily_median":  (daily_med * 100).round(3),
            "daily_p25":     (daily_p25 * 100).round(3),
            "daily_p75":     (daily_p75 * 100).round(3),
            "cum_return_pct": cum_med.round(3),
        })
        out = f"deep_{stock}_{datetime.now().strftime('%Y%m%d')}.csv"
        path_df.to_csv(out, index=False)
        print(f"\nDaily path saved: {out}")
        return

    # ── Full market ranking ──────────────────────────────────────────────────────
    stocks = list(df.columns)
    total  = len(stocks)
    print(f"\nForecasting {total} stocks ...\n")

    results = []
    for i, stock in enumerate(stocks):
        series = df[stock].values
        tokens = tokenize_series(series)
        if len(tokens) < MIN_CONTEXT:
            continue

        vol_ratio = compute_vol_ratio(series)
        beta      = compute_beta(series, market_series)

        paths   = run_monte_carlo_paths(model, tokens, max_horizon, args.n_samples, device, vol_ratio)
        metrics = compute_horizon_metrics(paths, horizons)

        row = {"stock": stock, "vol_ratio": round(vol_ratio, 2), "beta": round(beta, 2)}

        for h, m in metrics.items():
            mkt_exp = mkt_metrics.get(h, {}).get("mean_pct", 0)
            # Market-conditioned expected return:
            # Replace modelled market component with actual market forecast
            # adj = raw_forecast + beta * (mkt_forecast - 0)  (market neutral baseline = 0)
            adj_mean = round(m["mean_pct"] + beta * mkt_exp * 0.5, 3)
            row[f"raw_ret_{h}d"]      = m["mean_pct"]
            row[f"adj_ret_{h}d"]      = adj_mean       # USE THIS for decisions
            row[f"prob_up_{h}d"]      = m["prob_up_pct"]
            row[f"p10_{h}d"]          = m["p10_pct"]
            row[f"p90_{h}d"]          = m["p90_pct"]

        results.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} done ...", flush=True)

    rankings = pd.DataFrame(results)

    # Monday filter: next trading day is Monday -> exclude high vol_ratio stocks
    # (weekend gap risk makes high-volatility small caps unpredictable)
    next_trading_day = datetime.now()
    for _ in range(7):
        next_trading_day += timedelta(days=1)
        if next_trading_day.weekday() < 5:
            break
    if next_trading_day.weekday() == 0:  # Monday
        before = len(rankings)
        rankings = rankings[rankings["vol_ratio"] <= 1.5]
        print(f"\n  [Monday filter] Removed {before - len(rankings)} high-vol stocks (vol_ratio > 1.5) to reduce weekend gap risk.")

    # Composite score = average adj_ret across all horizons
    adj_cols = [c for c in rankings.columns if c.startswith("adj_ret_")]
    rankings["composite_score"] = rankings[adj_cols].mean(axis=1)
    rankings["composite_rank"]  = rankings["composite_score"].rank(ascending=False).astype(int)
    rankings = rankings.sort_values("composite_rank")

    # Per-horizon ranks
    for h in horizons:
        col = f"adj_ret_{h}d"
        if col in rankings.columns:
            rankings[f"rank_{h}d"] = rankings[col].rank(ascending=False).astype(int)

    today_str = datetime.now().strftime("%Y%m%d")
    out_file  = f"forecasts_{today_str}.csv"
    rankings.to_csv(out_file, index=False)
    print(f"\nForecasts saved: {out_file}")

    display_cols = ["stock", "composite_rank"] + adj_cols + [f"prob_up_{horizons[-1]}d", "vol_ratio", "beta"]
    display_cols = [c for c in display_cols if c in rankings.columns]

    print(f"\n=== Top {args.top_n} stocks (Long candidates) ===")
    print(rankings[display_cols].head(args.top_n).to_string(index=False))

    print(f"\n=== Bottom {args.top_n} stocks (Short candidates) ===")
    print(rankings[display_cols].tail(args.top_n).to_string(index=False))

    # Summary of market conditions used
    print(f"\n=== Market conditions at forecast time ===")
    print(f"  Data as of    : {df.index[-1].date()}")
    print(f"  Market vol    : {mkt_vol_ratio:.2f}x normal")
    for h, m in mkt_metrics.items():
        print(f"  Mkt {h:3d}d exp  : {m['mean_pct']:+.2f}%  (prob up: {m['prob_up_pct']}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockGPT Market-Grounded Forecast")
    parser.add_argument("--checkpoint", type=str,  default="stockgpt_final.pt")
    parser.add_argument("--data",       type=str,  default="dataset.csv")
    parser.add_argument("--days",       type=int,  nargs="+", default=[1, 5, 20],
                        help="Forecast horizons in trading days")
    parser.add_argument("--top_n",      type=int,  default=20)
    parser.add_argument("--n_samples",  type=int,  default=N_SAMPLES)
    parser.add_argument("--stock",      type=str,  default=None,
                        help="Single stock deep forecast (e.g. --stock RELIANCE)")
    args = parser.parse_args()
    run_forecast(args)
