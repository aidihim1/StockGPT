# backtest.py -- Evaluates StockGPT predictions using the paper's methodology:
#   1. Walk-forward prediction   -- model predicts each month using only past data
#   2. Portfolio backtesting     -- buy top decile, short bottom decile, track P&L
#   3. Fama-MacBeth regression   -- statistical proof predictions explain future returns
#   4. Factor alpha              -- alpha vs momentum, size, reversal factors
#
# Usage:
#   python backtest.py                        # full backtest on combined_returns.csv
#   python backtest.py --start 2022-01-01     # backtest from a specific date
#   python backtest.py --top_pct 10           # long top 10%, short bottom 10%

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from model import StockGPT, StockGPTConfig

# ── Settings ──────────────────────────────────────────────────────────────────
BLOCK_SIZE      = 256
N_SAMPLES       = 5          # MC paths (fewer for speed during backtest)
TEMPERATURE     = 0.8
MIN_HISTORY     = 100        # min days of history before predicting a stock
REBALANCE       = "monthly"  # "daily" or "monthly"
TOP_PCT         = 10         # long top X%, short bottom X%
TRANSACTION_COST = 0.001     # 0.1% per trade (brokerage + impact)
# ──────────────────────────────────────────────────────────────────────────────

BIN_EDGES   = np.linspace(-1.0, 1.0, 401)
BIN_CENTERS = np.concatenate([
    [-1.0],
    (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2,
    [1.0]
])


def tokenize_series(returns: np.ndarray) -> np.ndarray:
    valid = returns[~np.isnan(returns)]
    valid = np.clip(valid, -1.0 + 1e-9, 1.0 - 1e-9)
    return np.digitize(valid, BIN_EDGES).astype(np.int16)


def load_model(checkpoint_path: str, device: torch.device) -> StockGPT:
    print(f"Loading model from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = StockGPTConfig(**ckpt["config"])
    model = StockGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded (step {ckpt['step']})")
    return model


BATCH_STOCKS = 32   # process this many stocks at once (32x10=320 seqs fits in 4GB VRAM)

@torch.no_grad()
def predict_all_stocks(
    model: StockGPT,
    token_list: list,
    horizon: int,
    n_samples: int,
    device: torch.device
) -> np.ndarray:
    """
    Score all stocks in chunks to fit within 4GB VRAM.
    Returns: np.ndarray of shape (n_stocks,) with expected compounded returns.
    """
    n_stocks = len(token_list)
    bin_centers = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=device)
    all_results = np.zeros(n_stocks, dtype=np.float32)

    for start in range(0, n_stocks, BATCH_STOCKS):
        chunk = token_list[start:start + BATCH_STOCKS]
        chunk_size = len(chunk)

        # Build context on CPU with numpy (vectorized), then single GPU transfer
        ctx_np = np.zeros((chunk_size * n_samples, BLOCK_SIZE + horizon), dtype=np.int64)
        for i, tokens in enumerate(chunk):
            seq = tokens[-BLOCK_SIZE:]
            seq_len = len(seq)
            base_row = np.zeros(BLOCK_SIZE, dtype=np.int64)
            base_row[BLOCK_SIZE - seq_len:] = seq
            # Fill all n_samples rows at once (numpy broadcast, no Python loop)
            ctx_np[i * n_samples:(i + 1) * n_samples, :BLOCK_SIZE] = base_row
        ctx = torch.from_numpy(ctx_np).to(device)

        log_cum = torch.zeros(chunk_size * n_samples, device=device)

        for day in range(horizon):
            end = BLOCK_SIZE + day
            inp = ctx[:, max(0, end - BLOCK_SIZE):end]
            logits, _ = model(inp)
            probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
            toks = torch.multinomial(probs, 1).squeeze(1)
            ctx[:, end] = toks
            log_cum += torch.log1p(torch.clamp(bin_centers[toks], min=-0.9999))

        cumulative = torch.expm1(log_cum).cpu().numpy()
        cumulative = cumulative.reshape(chunk_size, n_samples).mean(axis=1)
        all_results[start:start + chunk_size] = cumulative

    return all_results


@torch.no_grad()
def predict_expected_return(
    model: StockGPT,
    context_tokens: np.ndarray,
    horizon: int,
    n_samples: int,
    device: torch.device
) -> float:
    """Single stock expected return — kept for compatibility."""
    result = predict_all_stocks(model, [context_tokens], horizon, n_samples, device)
    return float(result[0])


# ── Factor construction ────────────────────────────────────────────────────────

def compute_momentum(returns_df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """12-1 month momentum: sum of returns from t-lookback to t-1."""
    return returns_df.iloc[-lookback:].sum()


def compute_reversal(returns_df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Short-term reversal: last `lookback` days returns (negative = reversal factor)."""
    return -returns_df.iloc[-lookback:].sum()


def compute_volatility(returns_df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Rolling volatility over last `lookback` days."""
    return returns_df.iloc[-lookback:].std()


# ── Fama-MacBeth regression ────────────────────────────────────────────────────

def fama_macbeth(signal_series: list, realized_series: list) -> dict:
    """
    Fama-MacBeth regression:
      Each period t: regress realized_{t} on signal_{t} cross-sectionally
      Then average the time-series of regression coefficients.

    signal_series  : list of pd.Series (one per rebalance period), the model score
    realized_series: list of pd.Series (one per rebalance period), actual next-period returns

    Returns: dict with mean coefficient, t-stat, p-value, R-squared
    """
    from scipy import stats

    coefs, r2s = [], []

    for sig, real in zip(signal_series, realized_series):
        common = sig.index.intersection(real.index)
        if len(common) < 20:
            continue
        x = sig[common].values
        y = real[common].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 10:
            continue

        # OLS: y = a + b*x
        slope, intercept, r, p, se = stats.linregress(x, y)
        coefs.append(slope)
        r2s.append(r ** 2)

    if not coefs:
        return {"mean_coef": None, "t_stat": None, "p_value": None, "mean_r2": None}

    coefs = np.array(coefs)
    t_stat = float(np.mean(coefs) / (np.std(coefs, ddof=1) / np.sqrt(len(coefs))))
    from scipy.stats import t as t_dist
    p_value = float(2 * t_dist.sf(abs(t_stat), df=len(coefs) - 1))

    return {
        "mean_coef": round(float(np.mean(coefs)), 6),
        "t_stat":    round(t_stat, 3),
        "p_value":   round(p_value, 4),
        "mean_r2":   round(float(np.mean(r2s)), 4),
        "n_periods": len(coefs),
    }


# ── Portfolio returns ──────────────────────────────────────────────────────────

def portfolio_returns(
    scores: pd.Series,
    next_returns: pd.DataFrame,
    top_pct: float,
    transaction_cost: float
) -> dict:
    """
    Given scores for each stock, form long (top X%) / short (bottom X%) portfolio.
    Returns equal-weighted long, short, and long-short returns for the next period.
    """
    n = len(scores)
    k = max(1, int(n * top_pct / 100))

    ranked = scores.rank(ascending=False)
    longs  = scores[ranked <= k].index
    shorts = scores[ranked > (n - k)].index

    # Equal-weighted returns for the holding period
    long_ret  = next_returns[longs].mean(axis=1).mean()   - transaction_cost
    short_ret = next_returns[shorts].mean(axis=1).mean()  - transaction_cost
    ls_ret    = long_ret - short_ret   # long-short spread

    return {"long": long_ret, "short": short_ret, "long_short": ls_ret}


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_performance(returns: pd.Series, label: str, freq: str = "monthly") -> dict:
    """Compute Sharpe, max drawdown, CAGR, win rate from a return series.
    freq: 'monthly' (default, rebalance=monthly) or 'daily'
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # Use correct annualisation factor based on rebalance frequency
    ann_factor = 12 if freq == "monthly" else 252
    total_ret  = float((1 + r).prod() - 1)
    n_years    = len(r) / ann_factor
    cagr       = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) if n_years > 0 else 0

    sharpe     = float(r.mean() / r.std() * np.sqrt(ann_factor)) if r.std() > 0 else 0

    # Max drawdown
    cum        = (1 + r).cumprod()
    peak       = cum.cummax()
    drawdown   = (cum - peak) / peak
    max_dd     = float(drawdown.min())

    win_rate   = float((r > 0).mean())

    print(f"\n  [{label}]")
    print(f"    CAGR        : {cagr*100:+.2f}%")
    print(f"    Sharpe      : {sharpe:.3f}")
    print(f"    Max Drawdown: {max_dd*100:.2f}%")
    print(f"    Win Rate    : {win_rate*100:.1f}%")
    print(f"    Total Return: {total_ret*100:+.2f}%")

    return {
        "label": label, "cagr_pct": round(cagr*100, 2),
        "sharpe": round(sharpe, 3), "max_dd_pct": round(max_dd*100, 2),
        "win_rate_pct": round(win_rate*100, 1), "total_ret_pct": round(total_ret*100, 2),
        "n_periods": len(r),
    }


# ── Main backtest loop ─────────────────────────────────────────────────────────

def run_backtest(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    print(f"\nLoading {args.data} ...")
    df_long = pd.read_csv(args.data, parse_dates=["date"])
    print(f"  Rows: {len(df_long):,}  |  Stocks: {df_long['stock'].nunique()}")
    # Convert to wide format for backtest
    df = df_long.pivot(index="date", columns="stock", values="return_1d")

    # Restrict to backtest window
    start = pd.Timestamp(args.start)
    df_bt = df[df.index >= start].copy()
    print(f"  Backtest window: {df_bt.index[0].date()} to {df_bt.index[-1].date()}")
    print(f"  Trading days   : {len(df_bt)}")

    # Build rebalance dates (monthly = first trading day of each month)
    if REBALANCE == "monthly":
        rebalance_dates = df_bt.resample("MS").first().dropna(how="all").index
        rebalance_dates = [d for d in rebalance_dates if d in df_bt.index or
                           df_bt.index[df_bt.index >= d].size > 0]
        # Get actual trading days closest to month start
        rebalance_dates = []
        months = df_bt.index.to_period("M").unique()
        for m in months:
            month_days = df_bt.index[df_bt.index.to_period("M") == m]
            if len(month_days) > 0:
                rebalance_dates.append(month_days[0])
    else:
        rebalance_dates = list(df_bt.index)

    print(f"  Rebalance dates: {len(rebalance_dates)}")

    # Storage
    port_returns_long  = {}
    port_returns_short = {}
    port_returns_ls    = {}
    signal_list        = []
    realized_list      = []

    print(f"\nRunning walk-forward backtest ...\n")

    for idx, reb_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[idx + 1]

        # All data UP TO rebalance date (no lookahead)
        history = df[df.index <= reb_date]

        # Data between this rebalance and next (what we're trying to predict)
        future = df_bt[(df_bt.index > reb_date) & (df_bt.index <= next_date)]
        horizon_days = min(len(future), args.horizon)

        if horizon_days == 0:
            continue

        # Score ALL stocks in one batched GPU call
        valid_stocks = []
        token_list   = []
        for stock in df.columns:
            series = history[stock].values
            tokens = tokenize_series(series)
            if len(tokens) < MIN_HISTORY:
                continue
            valid_stocks.append(stock)
            token_list.append(tokens)

        if not valid_stocks:
            continue

        print(f"  Month {idx+1}/{len(rebalance_dates)-1}: {reb_date.date()} | {len(valid_stocks)} stocks ...", flush=True)
        expected_returns = predict_all_stocks(
            model, token_list, horizon_days, N_SAMPLES, device
        )
        scores = dict(zip(valid_stocks, expected_returns.tolist()))

        if len(scores) < 20:
            continue

        scores_series = pd.Series(scores)

        # Actual realized returns over the next period
        realized = future.sum()   # sum of daily returns = period return

        # Portfolio returns
        pr = portfolio_returns(scores_series, future, args.top_pct, TRANSACTION_COST)
        port_returns_long[reb_date]  = pr["long"]
        port_returns_short[reb_date] = pr["short"]
        port_returns_ls[reb_date]    = pr["long_short"]

        # Store for Fama-MacBeth
        signal_list.append(scores_series)
        realized_list.append(realized)

        print(f"  {reb_date.date()} -> {next_date.date()} | "
              f"stocks={len(scores)} | "
              f"L/S={pr['long_short']*100:+.2f}%", flush=True)

        # Free fragmented VRAM between months to prevent OOM accumulation
        torch.cuda.empty_cache()

    # ── Results ──────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    ls_series    = pd.Series(port_returns_ls)
    long_series  = pd.Series(port_returns_long)
    short_series = pd.Series(port_returns_short)

    perf_ls    = compute_performance(ls_series,     "Long-Short Portfolio", freq="monthly")
    perf_long  = compute_performance(long_series,   "Long-Only Portfolio",  freq="monthly")
    perf_short = compute_performance(-short_series, "Short-Only Portfolio", freq="monthly")

    # Fama-MacBeth
    print("\n" + "="*60)
    print("FAMA-MACBETH REGRESSION")
    print("="*60)
    fm = fama_macbeth(signal_list, realized_list)
    print(f"  Mean coefficient : {fm['mean_coef']}")
    print(f"  t-statistic      : {fm['t_stat']}")
    pv = fm['p_value']
    pv_label = "p<0.001 (highly significant)" if pv is not None and pv < 0.001 else ("p<0.01 (significant)" if pv is not None and pv < 0.01 else ("p<0.05 (significant)" if pv is not None and pv < 0.05 else "(not significant)"))
    print(f"  p-value          : {pv}  {pv_label}")
    print(f"  Mean R-squared   : {fm['mean_r2']}")
    print(f"  Periods          : {fm['n_periods']}")

    # Factor alpha
    print("\n" + "="*60)
    print("FACTOR ANALYSIS (vs Momentum, Reversal, Volatility)")
    print("="*60)
    _factor_alpha_analysis(signal_list, realized_list, df)

    # Save results
    today = datetime.now().strftime("%Y%m%d")
    results_df = pd.DataFrame({
        "date":       ls_series.index,
        "long_ret":   long_series.values,
        "short_ret":  short_series.values,
        "ls_ret":     ls_series.values,
    })
    results_df.to_csv(f"backtest_results_{today}.csv", index=False)

    summary = pd.DataFrame([perf_ls, perf_long, perf_short])
    summary.to_csv(f"backtest_summary_{today}.csv", index=False)

    fm_df = pd.DataFrame([fm])
    fm_df.to_csv(f"fama_macbeth_{today}.csv", index=False)

    print(f"\nFiles saved:")
    print(f"  backtest_results_{today}.csv")
    print(f"  backtest_summary_{today}.csv")
    print(f"  fama_macbeth_{today}.csv")


def _factor_alpha_analysis(signal_list, realized_list, df_full):
    """
    For each rebalance period, compute momentum/reversal/vol factors
    and run regression of realized returns on both model signal + factors.
    Shows whether model adds alpha beyond simple factors.
    """
    try:
        from scipy import stats
    except ImportError:
        print("  scipy not installed, skipping factor analysis.")
        return

    model_coefs, mom_coefs, rev_coefs, vol_coefs = [], [], [], []

    for sig, real in zip(signal_list, realized_list):
        common = sig.index.intersection(real.index)
        if len(common) < 30:
            continue

        # Find the date corresponding to this signal
        # Use signal values + build factor values for same stocks
        stocks = list(common)
        y = real[stocks].values

        x_model = sig[stocks].values

        # Build factors from recent history at the time of signal
        # (approximate: use full history, which is fine for factor construction)
        sub = df_full[stocks]
        mom = compute_momentum(sub, 20)[stocks].values
        rev = compute_reversal(sub, 5)[stocks].values
        vol = compute_volatility(sub, 20)[stocks].values

        # Stack features: [model, momentum, reversal, volatility]
        X = np.column_stack([x_model, mom, rev, vol])
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        if len(y) < 20:
            continue

        # OLS with multiple regressors (manual)
        try:
            Xt = X.T
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
            model_coefs.append(coef[1])
            mom_coefs.append(coef[2])
            rev_coefs.append(coef[3])
            vol_coefs.append(coef[4])
        except Exception:
            continue

    def summarize(coefs, name):
        if not coefs:
            return
        arr = np.array(coefs)
        t = float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(len(arr)))) if np.std(arr) > 0 else 0
        sig = "(p<0.01)" if abs(t) > 2.58 else ("(p<0.05)" if abs(t) > 1.96 else ("(p<0.10)" if abs(t) > 1.65 else ""))
        print(f"  {name:15s}: mean coef = {np.mean(arr):+.6f}  t = {t:+.2f} {sig}")

    summarize(model_coefs, "StockGPT signal")
    summarize(mom_coefs,   "Momentum")
    summarize(rev_coefs,   "Reversal")
    summarize(vol_coefs,   "Volatility")
    print("  (p<0.01 = highly significant, p<0.05 = significant, p<0.10 = marginal)")
    print("  If StockGPT signal is significant even after controlling for factors,")
    print("  it means the model adds REAL alpha beyond simple rules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockGPT Backtest")
    parser.add_argument("--checkpoint", type=str,  default="stockgpt_final.pt")
    parser.add_argument("--data",       type=str,  default="dataset.csv")
    parser.add_argument("--start",      type=str,  default="2022-01-01",
                        help="Backtest start date (use post-training data only)")
    parser.add_argument("--top_pct",    type=float, default=TOP_PCT,
                        help="Long top X%%, short bottom X%% of stocks")
    parser.add_argument("--n_samples",  type=int,  default=N_SAMPLES)
    parser.add_argument("--horizon",    type=int,  default=5,
                        help="Scoring horizon in days (5=fast, 20=full)")
    args = parser.parse_args()
    run_backtest(args)
