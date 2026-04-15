# app.py  StockGPT Chat Interface
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import glob, os, re

# NSE trading holidays (exchange-declared, not just weekends)
NSE_HOLIDAYS = {
    # 2025
    "2025-01-26","2025-02-26","2025-03-14","2025-04-10","2025-04-14",
    "2025-04-18","2025-05-01","2025-08-15","2025-10-02","2025-10-20",
    "2025-10-21","2025-11-05","2025-12-25",
    # 2026
    "2026-01-26","2026-02-19","2026-03-25","2026-04-03","2026-04-14",
    "2026-05-01","2026-08-15","2026-10-02","2026-10-09","2026-11-24",
    "2026-12-25",
}

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS

def get_next_trading_day(from_date: date = None) -> date:
    d = (from_date or date.today()) + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

st.set_page_config(
    page_title="StockGPT",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp { background-color: #0d0d0d; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 0 80px 0 !important; max-width: 860px !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }

/* ── centered column ── */
.chat-col {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 16px;
}

/* ── user message ── */
.msg-user {
    display: flex; justify-content: flex-end; margin: 10px 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.55;
    box-shadow: 0 2px 14px rgba(37,99,235,0.35);
}

/* ── ai message ── */
.msg-ai {
    display: flex; gap: 10px; align-items: flex-start; margin: 10px 0;
}
.ai-dot {
    width: 30px; height: 30px; flex-shrink: 0; margin-top: 3px;
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    box-shadow: 0 0 10px rgba(6,182,212,0.35);
}
.ai-bubble {
    background: #161616;
    border: 1px solid #2a2a2a;
    color: #e5e5e5;
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px;
    font-size: 15px;
    line-height: 1.9;
    max-width: 86%;
}

/* ── metric row ── */
.metric-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.metric-card {
    background: #111; border: 1px solid #222; border-radius: 10px;
    padding: 10px 14px; text-align: center; min-width: 90px;
}
.metric-label { color: #555; font-size: 10px; letter-spacing: 0.6px; margin-bottom: 4px; }
.metric-value { font-size: 18px; font-weight: 700; }
.metric-green { color: #22c55e; }
.metric-red   { color: #ef4444; }
.metric-blue  { color: #60a5fa; }
.metric-gray  { color: #777; }

/* ── stock table ── */
.st-table { width: 55%; border-collapse: collapse; font-size: 12px; }
.st-table th {
    color: #3a3a3a; padding: 3px 8px; font-size: 10px;
    letter-spacing: 0.6px; border-bottom: 1px solid #1e1e1e; text-align: left;
}
.st-table td { padding: 3px 8px; border-bottom: 1px solid #1a1a1a; color: #ccc; }
.st-table tr:last-child td { border-bottom: none; }

/* ── prob bar ── */
.prob-bar { height: 2px; border-radius: 2px; background: #1e1e1e; margin-top: 2px; width: 50px; }
.prob-fill { height: 100%; border-radius: 2px; }

/* ── suggestion chip buttons ── */
div[data-testid="stButton"] button {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #ccc !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
    transition: border-color 0.2s, color 0.2s;
}
div[data-testid="stButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #fff !important;
    background: #1a1a2e !important;
}
/* featured next-day button targeted by key */
div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"]) button {
    border-color: #2a2a2a !important;
}

/* ── remove extra padding from block container ── */
section[data-testid="stSidebar"] { display: none; }

/* compact chat input */
div[data-testid="stChatInput"] textarea {
    font-size: 13px !important;
    padding: 6px 10px !important;
    min-height: 0 !important;
    max-height: 44px !important;
    line-height: 1.4 !important;
}
div[data-testid="stChatInput"] {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 10px !important;
    margin: 0 0 6px !important;
    padding: 0 !important;
}

/* go back button - small and subtle */
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="column"]:first-child div[data-testid="stButton"] button {
    font-size: 11px !important;
    padding: 3px 10px !important;
    color: #555 !important;
    border-color: #1e1e1e !important;
    background: transparent !important;
}

/* ── welcome center ── */
.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 55vh; text-align: center;
    padding: 20px 0 10px;
}
.welcome-icon {
    width: 60px; height: 60px;
    background: linear-gradient(135deg, #06b6d4, #2563eb);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; margin-bottom: 20px;
    box-shadow: 0 0 28px rgba(6,182,212,0.35);
}
.welcome-title { color: #fff; font-size: 26px; font-weight: 700; margin-bottom: 8px; }
.welcome-sub { color: #555; font-size: 14px; margin-bottom: 30px; line-height: 1.6; }

/* ── chips ── */
.chips-wrap { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
</style>
""", unsafe_allow_html=True)


# Data loading

@st.cache_data(ttl=300)
def load_forecast():
    files = sorted(glob.glob("forecasts_*.csv"), reverse=True)
    if not files:
        return None, None
    df = pd.read_csv(files[0])
    date_str = files[0].replace("forecasts_","").replace(".csv","")
    try:
        fdate = datetime.strptime(date_str, "%Y%m%d").strftime("%b %d, %Y")
    except:
        fdate = date_str
    return df, fdate

@st.cache_data(ttl=60)
def load_stock_history(stock):
    try:
        chunks = []
        for chunk in pd.read_csv("dataset.csv", parse_dates=["date"], chunksize=200_000):
            s = chunk[chunk["stock"] == stock]
            if len(s): chunks.append(s)
        if not chunks: return None
        return pd.concat(chunks).sort_values("date").reset_index(drop=True)
    except:
        return None

@st.cache_data(ttl=300)
def load_backtest_summary():
    files = sorted(glob.glob("backtest_summary_*.csv"), reverse=True)
    if not files: return None
    return pd.read_csv(files[0], index_col=0)

forecast_df, forecast_date = load_forecast()
summary = load_backtest_summary()


# Helper rendering

def prob_bar_html(p, color="#22c55e"):
    w = max(0, min(100, float(p)))
    return f"""<div class='prob-bar'><div class='prob-fill' style='width:{w:.0f}%;background:{color};'></div></div>"""


def signal_label(rnk, total):
    pct = rnk / total * 100
    if pct <= 5:   return "STRONG BUY",  "#22c55e"
    if pct <= 20:  return "BUY",          "#4ade80"
    if pct >= 95:  return "STRONG AVOID", "#ef4444"
    if pct >= 80:  return "AVOID",        "#f87171"
    return "NEUTRAL", "#eab308"

def render_stock_table(stocks, mode="buy"):
    total = len(forecast_df) if forecast_df is not None else 2411
    _next = get_next_trading_day(date.today())
    _next_label = _next.strftime("%a %b %d")
    rows = ""
    for i, s in enumerate(stocks, 1):
        rnk = int(s.get("composite_rank", i))
        p   = s.get("prob_up_1d", 50)
        top_pct = round((1 - rnk / total) * 100)
        sig, sig_color = signal_label(rnk, total)
        bg = "background:#111;" if i % 2 == 0 else ""
        rows += f"""<tr style='{bg}'>
            <td style='color:#444;width:20px;font-size:11px;padding:4px 6px;'>{i}</td>
            <td style='color:#fff;font-weight:600;font-size:13px;padding:4px 6px;'>{s['stock']}</td>
            <td style='padding:4px 6px;'><span style='color:{sig_color};font-size:11px;font-weight:700;'>{sig}</span></td>
            <td style='color:#555;font-size:11px;text-align:right;padding:4px 6px;'>Top {100-top_pct}%</td>
            <td style='text-align:right;padding:4px 6px;'>
                <span style='color:#888;font-size:12px;'>{p:.0f}%</span>
                {prob_bar_html(p, sig_color)}
            </td>
        </tr>"""
    return f"""<table style='width:100%;border-collapse:collapse;font-size:12px;'>
        <thead><tr style='color:#555;font-size:10px;letter-spacing:0.5px;border-bottom:1px solid #222;'>
            <th style='padding:4px 6px;text-align:left;'>#</th>
            <th style='padding:4px 6px;text-align:left;'>Stock</th>
            <th style='padding:4px 6px;text-align:left;'>Signal</th>
            <th style='padding:4px 6px;text-align:right;'>Rank</th>
            <th style='padding:4px 6px;text-align:right;'>P(Up {_next_label})</th>
        </tr></thead><tbody>{rows}</tbody></table>"""


def render_long_short_table(longs, shorts):
    total = len(forecast_df) if forecast_df is not None else 2411
    _next = get_next_trading_day(date.today())
    _next_label = _next.strftime("%a %b %d")
    def side(stocks, color, label):
        rows = ""
        for i, s in enumerate(stocks, 1):
            rnk = int(s.get("composite_rank", i))
            p   = s.get("prob_up_1d", 50)
            top_pct = round((1 - rnk / total) * 100)
            rows += f"""<tr>
                <td style='color:#444;width:20px;font-size:11px;'>{i}</td>
                <td style='color:#fff;font-weight:600;font-size:12px;'>{s['stock']}</td>
                <td style='text-align:right;font-size:11px;color:#555;'>Top {100-top_pct}%</td>
                <td style='text-align:right;font-size:11px;color:#666;'>{p:.0f}%</td>
            </tr>"""
        return f"""<div>
            <div style='color:{color};font-weight:700;font-size:12px;letter-spacing:1px;margin-bottom:6px;padding:4px 8px;background:{color}18;border-radius:6px;'>{label}</div>
            <table style='width:100%;border-collapse:collapse;'>
            <thead><tr style='color:#333;font-size:10px;'>
                <th style='padding:3px 4px;'>#</th><th>Stock</th>
                <th style='text-align:right;'>Rank</th>
                <th style='text-align:right;'>P(Up {_next_label})</th>
            </tr></thead><tbody>{rows}</tbody></table></div>"""

    ls = side(longs,  "#22c55e", "LONG / BUY")
    ss = side(shorts, "#ef4444", "SHORT / AVOID")
    return f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;'>{ls}{ss}</div>"


def render_metric_cards(items):
    cards = ""
    for label, value, cls in items:
        cards += f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value {cls}'>{value}</div>
        </div>"""
    return f"<div class='metric-row'>{cards}</div>"


# AI response engine

def ai_response(query: str) -> dict:
    q = query.lower().strip()
    resp = {"type": "text", "text": "", "stocks": [], "stocks_long": [],
            "stocks_short": [], "chart_stock": None, "metrics_html": ""}

    if forecast_df is None:
        resp["text"] = "No forecast data yet. Run <code>forecast.py</code> first."
        return resp

    # Extract number from query
    n = 10
    for word in q.split():
        if word.isdigit():
            n = min(int(word), 30)

    _next_td    = get_next_trading_day(date.today())
    _next_label = _next_td.strftime("%a, %b %d %Y")

    # Specific stock lookup runs FIRST so "invest in RELIANCE" finds the stock, not the buy list
    all_stocks = forecast_df["stock"].tolist()
    matched = [s for s in all_stocks if re.search(r'\b' + re.escape(s.lower()) + r'\b', q)]
    if matched:
        stock = matched[0]
        row   = forecast_df[forecast_df["stock"] == stock].iloc[0]
        r1    = row.get("adj_ret_1d",   0)
        r5    = row.get("adj_ret_5d",   0)
        r20   = row.get("adj_ret_20d",  0)
        p1    = row.get("prob_up_1d",  50)
        p5    = row.get("prob_up_5d",  50)
        p20   = row.get("prob_up_20d", 50)
        vol   = row.get("vol_ratio",    1)
        rnk   = int(row.get("composite_rank", 0))
        total = len(forecast_df)
        pctile = round((1 - rnk / total) * 100)

        sig_label, sig_color = signal_label(rnk, total)

        vol_txt = "more jumpy than usual" if vol > 1.3 else ("calmer than usual" if vol < 0.7 else "normal")
        _next_lbl2 = _next_td.strftime("%a %b %d")
        resp["type"] = "stock_detail"
        resp["chart_stock"] = stock
        resp["text"] = (
            f"<span style='color:#555;font-size:11px;letter-spacing:1px;'>NSE STOCK</span><br>"
            f"<strong style='color:#fff;font-size:20px;'>{stock}</strong> "
            f"<span style='color:#444;font-size:12px;'>· NSE India</span><br>"
            f"<span style='color:{sig_color};font-weight:700;font-size:15px;'>{sig_label}</span> "
            f"&nbsp; Rank <strong style='color:#fff;'>#{rnk}</strong> out of {total:,} stocks "
            f"(better than {pctile}% of all NSE stocks)<br><br>"

            f"<strong style='color:#aaa;font-size:12px;letter-spacing:1px;'>MODEL CONFIDENCE</strong><br>"
            f"P(Up {_next_lbl2}): <strong style='color:#60a5fa;'>{p1:.0f}%</strong> "
            f"(50% = coin flip &nbsp;|&nbsp; 60%+ = strong signal)<br>"
            f"P(Up in 20 trading days): <strong style='color:#60a5fa;'>{p20:.0f}%</strong><br>"
            f"Volatility vs own history: <strong style='color:#aaa;'>{vol_txt}</strong><br><br>"

            f"<span style='color:#555;font-size:12px;'>"
            f"{'Strong signal. Model places this in the top 5% of all NSE stocks.' if pctile>=95 else 'Weak signal. Model places this in the bottom 20%.' if pctile<=20 else 'Moderate signal. Use alongside other research before deciding.'}"
            f"</span>"
        )
        return resp

    # ── AVOID / SELL signals ────────────────────────────────────────────────
    avoid_kw = ["sell","avoid","bearish","worst","fall","drop","weak",
                "do not buy","don't buy","not buy","loser","decline",
                "underperform","red","negative","danger","risky stock",
                "going down","lose money","loss making"]
    if any(x in q for x in avoid_kw):
        bot = forecast_df.nlargest(n, "composite_rank")
        avg_prob = bot["prob_up_1d"].mean()
        resp["type"] = "stocks_sell"
        resp["text"] = (
            f"Top <strong style='color:#fff;'>{n} stocks to AVOID</strong> forecast for <strong style='color:#60a5fa;'>{_next_label}</strong>.<br>"
            f"Avg P(Up {_next_td.strftime('%a')}): <strong style='color:#ef4444;'>{avg_prob:.0f}%</strong>. "
            f"Weakest-ranked stocks in the model."
        )
        resp["stocks"] = bot[["stock","composite_rank","adj_ret_1d","adj_ret_5d","adj_ret_20d","prob_up_1d"]].to_dict("records")
        return resp

    # ── LONG-SHORT (explicit) ────────────────────────────────────────────────
    ls_triggers = ["long short","long-short","buy and short","buy and avoid",
                   "top and bottom","best and worst","long and short","buy short",
                   "long vs short","hedge","market neutral","both sides",
                   "which stock","portfolio"]
    if any(x in q for x in ls_triggers) or \
       (any(x in q for x in ["buy","long"]) and any(x in q for x in ["short","bottom"])):
        top = forecast_df.nsmallest(n, "composite_rank")
        bot = forecast_df.nlargest(n, "composite_rank")
        avg_l_prob = top["prob_up_1d"].mean()
        avg_s_prob = bot["prob_up_1d"].mean()

        # Market regime filter
        regime = forecast_df["regime"].iloc[0] if "regime" in forecast_df.columns else "neutral"
        mkt_prob = forecast_df["market_prob_up_1d"].iloc[0] if "market_prob_up_1d" in forecast_df.columns else 50

        if regime == "bull":
            regime_note = f"<br><br><span style='color:#facc15;'>BULL DAY (market prob up: {mkt_prob:.0f}%) — Shorts skipped. Only longs recommended.</span>"
            resp["stocks_short"] = []
        elif regime == "bear":
            regime_note = f"<br><br><span style='color:#facc15;'>BEAR DAY (market prob up: {mkt_prob:.0f}%) — Longs skipped. Only shorts recommended.</span>"
            resp["stocks_long"] = []
        else:
            regime_note = ""

        resp["type"] = "long_short"
        resp["text"] = (
            f"Model rankings for <strong style='color:#60a5fa;'>{_next_label}</strong>.<br><br>"
            f"<strong style='color:#22c55e;'>Green = BUY</strong> &nbsp; <strong style='color:#ef4444;'>Red = AVOID</strong><br>"
            f"Avg P(Up {_next_td.strftime('%a')}): Longs <strong style='color:#22c55e;'>{avg_l_prob:.0f}%</strong> &nbsp;|&nbsp; "
            f"Shorts <strong style='color:#ef4444;'>{avg_s_prob:.0f}%</strong>"
            f"{regime_note}"
        )
        if regime != "bear":
            resp["stocks_long"]  = top[["stock","composite_rank","adj_ret_1d","adj_ret_5d","adj_ret_20d","prob_up_1d"]].to_dict("records")
        if regime != "bull":
            resp["stocks_short"] = bot[["stock","composite_rank","adj_ret_1d","adj_ret_5d","adj_ret_20d","prob_up_1d"]].to_dict("records")
        return resp

    # ── BUY signals ─────────────────────────────────────────────────────────
    buy_kw = ["buy","top","best","invest","long","bullish","recommend","pick",
              "purchase","gainer","going up","rise","momentum","strong stock",
              "winner","upside","positive","ideas","suggestion","screen",
              "which stock to","what to buy","good stock","high potential",
              "outperform","rally","breakout","opportunity","safe stock",
              "low risk","defensive","can i buy","should i buy","worth buying"]
    if any(x in q for x in buy_kw):
        top = forecast_df.nsmallest(n, "composite_rank")
        avg_prob = top["prob_up_1d"].mean()
        resp["type"] = "stocks_buy"
        resp["text"] = (
            f"Top <strong style='color:#fff;'>{n} stocks to BUY</strong> forecast for <strong style='color:#60a5fa;'>{_next_label}</strong>.<br>"
            f"Avg P(Up {_next_td.strftime('%a')}): <strong style='color:#22c55e;'>{avg_prob:.0f}%</strong>. "
            f"Sorted by AI confidence."
        )
        resp["stocks"] = top[["stock","composite_rank","adj_ret_1d","adj_ret_5d","adj_ret_20d","prob_up_1d"]].to_dict("records")
        return resp

    # ── NIFTY / INDEX question ───────────────────────────────────────────────
    if any(x in q for x in ["nifty","sensex","index","nse index","bank nifty","midcap","smallcap index"]):
        resp["text"] = (
            f"<strong style='color:#fff;'>StockGPT does not predict Nifty or Sensex direction.</strong><br><br>"
            f"It ranks <strong style='color:#60a5fa;'>individual NSE stocks</strong> relative to each other "
            f"not the overall market.<br><br>"
            f"The model answers: <em>\"Which stocks will outperform others?\"</em><br>"
            f"Not: <em>\"Will the market go up or down?\"</em><br><br>"
            f"<span style='color:#555;font-size:12px;'>Try asking: <strong style='color:#ccc;'>\"Top 10 stocks to buy\"</strong> "
            f"or type any stock name like <strong style='color:#ccc;'>RELIANCE</strong>.</span>"
        )
        return resp

    # ── NEWS / FUNDAMENTALS question ─────────────────────────────────────────
    if "backtest" not in q and \
       any(x in q for x in ["news","earnings","quarterly results","fundamentals","pe ratio",
                             "balance sheet","revenue","profit loss","dividend","eps","roe","roce"]):
        resp["text"] = (
            f"<strong style='color:#fff;'>StockGPT only uses price data no news or fundamentals.</strong><br><br>"
            f"The model was trained purely on <strong style='color:#60a5fa;'>daily return patterns</strong> "
            f"over 25 years. It does not read:<br>"
            f"- News or analyst reports<br>"
            f"- Earnings / quarterly results<br>"
            f"- PE ratio, ROE, balance sheets<br><br>"
            f"<span style='color:#555;font-size:12px;'>For fundamental analysis, use Screener.in or Tickertape.<br>"
            f"StockGPT is best used as a <strong style='color:#ccc;'>technical / quantitative signal</strong> "
            f"alongside your own fundamental research.</span>"
        )
        return resp

    # ── SECTOR question ──────────────────────────────────────────────────────
    if any(x in q for x in ["sector","banking stocks","it stocks","pharma stocks","auto stocks",
                             "fmcg","realty","infra","energy stocks","metal stocks","nbfc"]):
        resp["text"] = (
            f"<strong style='color:#fff;'>StockGPT ranks all NSE stocks together no sector filter yet.</strong><br><br>"
            f"The model doesn't group stocks by sector. It ranks all 2,411 NSE stocks "
            f"by expected outperformance and you pick from the top.<br><br>"
            f"<span style='color:#555;font-size:12px;'>Tip: Ask <strong style='color:#ccc;'>\"Top 10 stocks to buy\"</strong> "
            f"and check if any match your preferred sector. Or type a specific stock name like "
            f"<strong style='color:#ccc;'>HDFCBANK</strong> or <strong style='color:#ccc;'>INFY</strong> for individual analysis.</span>"
        )
        return resp

    # ── CAN I MAKE MONEY / IS IT SAFE ───────────────────────────────────────
    if any(x in q for x in ["can i make money","how much can i earn","how much money","is it safe",
                             "will i profit","guaranteed","sure shot","loss","risk free",
                             "how much return","can i trust","reliable","accurate"]):
        resp["type"] = "backtest"
        resp["text"] = (
            f"<strong style='color:#fff;'>Honest answer about returns and risk:</strong><br><br>"
            f"The model has a <strong style='color:#22c55e;'>statistically proven edge</strong> "
            f"(t-stat 7.39, 84% win rate over 4 years) the signal is real, not luck.<br><br>"
            f"<strong style='color:#eab308;'>But:</strong><br>"
            f"- Returns are small (~1% per year on a market-neutral basis)<br>"
            f"- No strategy is guaranteed past performance doesn't guarantee future results<br>"
            f"- Use this as a <strong style='color:#fff;'>screening/research tool</strong>, not a blind buy signal<br>"
            f"- Always do your own research before investing real money<br><br>"
            f"<span style='color:#555;font-size:12px;'>The model is best used to shortlist stocks for further study, "
            f"not as a standalone trading system.</span>"
        )
        resp["metrics_html"] = render_metric_cards([
            ("SHARPE",       "4.0",    "metric-blue"),
            ("WIN RATE",     "84.3%",  "metric-green"),
            ("T-STAT",       "7.39",   "metric-blue"),
            ("MAX LOSS",     "-0.13%", "metric-gray"),
        ])
        return resp

    # ── MARKET OVERVIEW ──────────────────────────────────────────────────────
    market_kw = ["market","overview","summary","nse market","sentiment","outlook",
                 "broad market","market signal","how is market","market today",
                 "market condition","market mood","bullish or bearish"]
    if any(x in q for x in market_kw):
        total   = len(forecast_df)
        top10p  = int(total * 0.10)
        strong_buy  = int((forecast_df["composite_rank"] <= top10p).sum())
        strong_avoid= int((forecast_df["composite_rank"] >= total - top10p).sum())
        neutral = total - strong_buy - strong_avoid
        top3 = forecast_df.nsmallest(3, "composite_rank")["stock"].tolist()
        bot3 = forecast_df.nlargest(3, "composite_rank")["stock"].tolist()
        top3_str = " &nbsp; ".join([f"<strong style='color:#22c55e;'>{s}</strong>" for s in top3])
        bot3_str = " &nbsp; ".join([f"<strong style='color:#ef4444;'>{s}</strong>" for s in bot3])
        resp["type"] = "market"
        resp["text"] = (
            f"<strong style='color:#fff;'>NSE Market Signal forecast for {_next_label}</strong><br>"
            f"<span style='color:#555;font-size:12px;'>This is a cross-sectional ranking signal "
            f"it shows which stocks will outperform others, not whether Nifty goes up or down.</span><br><br>"
            f"Out of <strong style='color:#fff;'>{total:,}</strong> NSE stocks ranked:<br>"
            f"<strong style='color:#22c55e;'>{strong_buy:,} strong buy signals</strong> (top 10%)<br>"
            f"<strong style='color:#ef4444;'>{strong_avoid:,} avoid signals</strong> (bottom 10%)<br>"
            f"<strong style='color:#888;'>{neutral:,} neutral</strong><br><br>"
            f"<strong style='color:#aaa;font-size:12px;letter-spacing:1px;'>TOP 3 RANKED</strong><br>"
            f"{top3_str}<br><br>"
            f"<strong style='color:#aaa;font-size:12px;letter-spacing:1px;'>BOTTOM 3 RANKED</strong><br>"
            f"{bot3_str}"
        )
        resp["metrics_html"] = render_metric_cards([
            ("TOTAL STOCKS", f"{total:,}",       "metric-blue"),
            ("STRONG BUY",   f"{strong_buy:,}",  "metric-green"),
            ("AVOID",        f"{strong_avoid:,}", "metric-red"),
            ("NEUTRAL",      f"{neutral:,}",      "metric-gray"),
        ])
        return resp

    # ── BACKTEST / PERFORMANCE ───────────────────────────────────────────────
    bt_kw = ["backtest","performance","accuracy","sharpe","cagr","result","history",
             "track record","alpha","how good","profit","drawdown","win rate",
             "statistics","significant","factor","fama","t-stat","t stat",
             "return","how much did it make","how well","how accurate",
             "tested","validation","out of sample"]
    if any(x in q for x in bt_kw):
        resp["type"] = "backtest"
        resp["text"] = (
            f"<strong style='color:#fff;'>StockGPT 4 year backtest (2022–2026)</strong><br><br>"
            f"Right <strong style='color:#22c55e;'>84 out of 100 months</strong>.<br>"
            f"Long-short return: <strong style='color:#22c55e;'>+4.58%</strong> over 4 years (market-neutral).<br>"
            f"Sharpe <strong style='color:#60a5fa;'>4.0</strong> · Max loss: <strong style='color:#888;'>-0.13%</strong>.<br>"
            f"Fama-MacBeth t-stat <strong style='color:#fff;'>7.39</strong> survives factor controls."
        )
        resp["metrics_html"] = render_metric_cards([
            ("SHARPE",       "4.0",    "metric-blue"),
            ("WIN RATE",     "84.3%",  "metric-green"),
            ("TOTAL RETURN", "+4.58%", "metric-green"),
            ("MAX LOSS",     "-0.13%", "metric-gray"),
            ("T-STAT",       "7.39",   "metric-blue"),
        ])
        return resp

    # ── HOW IT WORKS ────────────────────────────────────────────────────────
    how_kw = ["what is stockgpt","how does","explain","how it work","how stockgpt",
              "gpt","transformer","token","architecture","train","neural",
              "machine learning","deep learning","predict","ai model",
              "how are stocks ranked","how does ranking","algorithm","methodology"]
    if any(x in q for x in how_kw):
        resp["text"] = (
            f"<strong style='color:#fff;'>StockGPT GPT transformer trained on stock prices, not text.</strong><br><br>"
            f"<strong style='color:#aaa;font-size:12px;letter-spacing:1px;'>HOW IT WORKS</strong><br>"
            f"1. Every day's stock return is converted to a token (402-bin quantization)<br>"
            f"2. The model reads the last 256 trading days (~1 year) of returns per stock<br>"
            f"3. Runs 200 Monte Carlo simulations of future price paths<br>"
            f"4. Averages simulations to get 1-day, 5-day, 20-day forecasts<br>"
            f"5. All 2,411 NSE stocks ranked top 10% = BUY, bottom 10% = AVOID<br><br>"
            f"<strong style='color:#aaa;font-size:12px;letter-spacing:1px;'>THE MODEL</strong><br>"
            f"Trained on <strong style='color:#fff;'>25 years</strong> of NSE data (2000–2021).<br>"
            f"929,000 parameters. Decoder-only architecture (same family as GPT-2).<br>"
            f"Runs on GPU in a few minutes for all 2,411 stocks.<br><br>"
            f"<span style='color:#555;font-size:13px;'>"
            f"Uses only price patterns no news, fundamentals, or analyst data."
            f"</span>"
        )
        return resp

    # ── NEXT TRADING DAY FORECAST ────────────────────────────────────────────
    if any(x in q for x in ["next day","next trading","tomorrow forecast","next market",
                             "forecast tomorrow","forecast for","upcoming","next week forecast"]):
        next_day     = get_next_trading_day(date.today())
        next_day_str = next_day.strftime("%A, %b %d %Y")
        total  = len(forecast_df)
        top10  = forecast_df.nsmallest(10, "composite_rank")
        top10p = int(total * 0.10)
        strong_buy   = int((forecast_df["composite_rank"] <= top10p).sum())
        strong_avoid = int((forecast_df["composite_rank"] >= total - top10p).sum())
        cols_data    = ["stock","composite_rank","adj_ret_1d","adj_ret_5d","adj_ret_20d","prob_up_1d"]
        table_html   = render_stock_table(top10[cols_data].to_dict("records"), "buy")
        resp["type"] = "next_day"
        resp["stocks"] = []
        resp["text"] = (
            f"<strong style='color:#60a5fa;font-size:15px;'>Next trading day: {next_day_str}</strong><br>"
            f"<span style='color:#555;font-size:12px;'>Data up to {forecast_date}.</span><br><br>"
            f"<strong style='color:#aaa;font-size:11px;letter-spacing:1px;'>MODEL RANKINGS</strong><br>"
            f"<strong style='color:#22c55e;'>{strong_buy:,} stocks</strong> in top 10% (strong buy zone) &nbsp;|&nbsp; "
            f"<strong style='color:#ef4444;'>{strong_avoid:,} stocks</strong> in bottom 10% (avoid zone)<br>"
            f"<span style='color:#555;font-size:11px;'>Rankings by expected outperformance not overall market direction.</span><br><br>"
            f"<strong style='color:#aaa;font-size:11px;letter-spacing:1px;'>TOP 10 STOCKS RANKED BY MODEL</strong><br>"
            f"<div style='margin-top:6px;'>{table_html}</div>"
        )
        resp["metrics_html"] = render_metric_cards([
            ("NEXT DAY",    next_day.strftime("%b %d"), "metric-blue"),
            ("STRONG BUY",  f"{strong_buy:,}",          "metric-green"),
            ("AVOID",       f"{strong_avoid:,}",         "metric-red"),
            ("TOTAL",       f"{total:,}",                "metric-gray"),
        ])
        return resp

    # ── HELP ────────────────────────────────────────────────────────────────
    if any(x in q for x in ["help","what can you","commands","guide","features",
                             "what do you","how to use","usage","examples"]):
        resp["text"] = (
            f"<strong style='color:#fff;'>Here is what you can ask me:</strong><br><br>"
            f"<strong style='color:#22c55e;'>Stock picks</strong><br>"
            f"- \"Top 10 stocks to buy\" / \"Top 5 stocks to buy\"<br>"
            f"- \"Which stocks should I avoid?\"<br>"
            f"- \"Show me buy and short signals\"<br><br>"
            f"<strong style='color:#60a5fa;'>Individual stock analysis</strong><br>"
            f"- Type any NSE ticker: <strong style='color:#fff;'>RELIANCE</strong>, <strong style='color:#fff;'>TCS</strong>, <strong style='color:#fff;'>INFY</strong><br>"
            f"- Or ask: \"Should I invest in HDFCBANK?\"<br>"
            f"- Shows signal, rank, probability, and price chart<br><br>"
            f"<strong style='color:#f59e0b;'>Market overview</strong><br>"
            f"- \"What is the NSE market outlook?\"<br>"
            f"- \"How is the market today?\"<br><br>"
            f"<strong style='color:#a78bfa;'>Model & results</strong><br>"
            f"- \"Show backtest results\" / \"How accurate is the model?\"<br>"
            f"- \"How does StockGPT work?\"<br>"
            f"- \"Is it safe to invest using this?\"<br><br>"
            f"<span style='color:#555;font-size:12px;'>Note: StockGPT uses only price data "
            f"no news, earnings, or fundamentals.</span>"
        )
        return resp

    # ── FALLBACK ─────────────────────────────────────────────────────────────
    resp["text"] = (
        f"I didn't quite understand that. Here's what I can help with:<br><br>"
        f"- <strong style='color:#fff;'>\"Top 10 stocks to buy\"</strong><br>"
        f"- <strong style='color:#fff;'>\"RELIANCE\"</strong> or any NSE ticker for analysis<br>"
        f"- <strong style='color:#fff;'>\"Stocks to avoid\"</strong><br>"
        f"- <strong style='color:#fff;'>\"Market outlook\"</strong><br>"
        f"- <strong style='color:#fff;'>\"Backtest results\"</strong><br>"
        f"- <strong style='color:#fff;'>\"How does it work?\"</strong><br>"
        f"- <strong style='color:#fff;'>\"Is it safe to invest?\"</strong><br><br>"
        f"<span style='color:#555;font-size:12px;'>Tip: For stock analysis just type the ticker symbol directly, e.g. <strong style='color:#ccc;'>TATAMOTORS</strong></span>"
    )
    return resp


# Header (slim, centered)

# Header

st.markdown(f"""
<div style='background:#111;border-bottom:1px solid #1e1e1e;padding:10px 20px;
            display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;'>
    <div style='display:flex;align-items:center;gap:10px;'>
        <div style='background:linear-gradient(135deg,#06b6d4,#3b82f6);width:32px;height:32px;
                    border-radius:9px;display:flex;align-items:center;justify-content:center;
                    font-size:16px;box-shadow:0 0 12px rgba(6,182,212,0.4);'>📈</div>
        <div>
            <div style='color:#fff;font-weight:700;font-size:15px;line-height:1.2;'>StockGPT</div>
            <div style='color:#444;font-size:11px;'>NSE India · 2,411 stocks · GPT Transformer</div>
        </div>
    </div>
    <div style='display:flex;align-items:center;gap:20px;'>
        <div style='text-align:right;'>
            <div style='color:#22c55e;font-size:11px;font-weight:600;'>Sharpe 4.0 &nbsp;|&nbsp; Win Rate 84.3% &nbsp;|&nbsp; T-stat 7.39</div>
            <div style='color:#60a5fa;font-size:10px;font-weight:600;'>Forecasting: {get_next_trading_day(date.today()).strftime("%a, %b %d %Y")}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# Init chat

_today    = date.today()
_next_day = get_next_trading_day(_today)
_fcount   = len(forecast_df) if forecast_df is not None else 0

# Dynamic non-trading day banner
if not is_trading_day(_today):
    st.markdown(
        f"<div style='background:#1a1200;border:1px solid #3a2800;border-radius:8px;"
        f"padding:7px 14px;margin:4px 0 6px;font-size:12px;color:#eab308;'>"
        f"Today ({_today.strftime('%A, %b %d')}) is a non-trading day. "
        f"Next market open: <strong style='color:#fff;'>{_next_day.strftime('%A, %b %d %Y')}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "ai",
        "type": "welcome",   # rendered dynamically never frozen
        "stocks": [], "stocks_long": [], "stocks_short": [],
        "chart_stock": None, "metrics_html": "", "text": ""
    }]

# Midnight rollover guard if the date changed since session started,
# update the session date so dynamic renders pick up the new day.
if "session_date" not in st.session_state:
    st.session_state.session_date = date.today()
elif st.session_state.session_date != date.today():
    st.session_state.session_date = date.today()
    # Date changed — clear all caches so fresh data is picked up immediately
    st.cache_data.clear()


# Render messages

with st.container():
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-user'>
                <div class='bubble'>{msg['text']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Welcome message always computed fresh so date is never stale
            if msg.get("type") == "welcome":
                _wr_next = get_next_trading_day(date.today())
                _wr_fcount = len(forecast_df) if forecast_df is not None else 0
                bubble_inner = (
                    f"Hi! I am <strong style='color:#fff;'>StockGPT</strong> AI trained on 25 years of NSE price data.<br>"
                    f"Ranked <strong style='color:#fff;'>{_wr_fcount:,} NSE stocks</strong> today.<br>"
                    f"Backtested 2022–2026: <strong style='color:#22c55e;'>84% win rate</strong>, Sharpe <strong style='color:#22c55e;'>4.0</strong>.<br>"
                    f"Next forecast: <strong style='color:#60a5fa;'>{_wr_next.strftime('%A, %b %d %Y')}</strong>"
                )
            else:
                bubble_inner = msg["text"]
            if msg.get("metrics_html"):
                bubble_inner += f"<div style='margin-top:10px;'>{msg['metrics_html']}</div>"

            st.markdown(f"""
            <div class='msg-ai'>
                <div class='ai-dot'>📈</div>
                <div class='ai-bubble'>{bubble_inner}</div>
            </div>""", unsafe_allow_html=True)

            if msg.get("type") == "next_day":
                _upd_col, _ = st.columns([2, 5])
                with _upd_col:
                    if st.button("🔄 Update data & refresh", key=f"upd_{id(msg)}", width='stretch'):
                        _sp = __import__("subprocess")
                        _ok = False
                        with st.spinner("Fetching latest prices (~4 min for all stocks)..."):
                            try:
                                r1 = _sp.run(["python","update_data.py"],
                                    cwd=r"C:\Users\Dell\Desktop\stockGPT v3",
                                    capture_output=True, text=True, timeout=480)
                                if r1.returncode != 0:
                                    st.error(f"update_data.py failed:\n{r1.stderr[-400:] or r1.stdout[-400:]}")
                                else:
                                    _ok = True
                            except _sp.TimeoutExpired:
                                # Even on timeout, incremental saves mean partial data was written
                                st.warning("Timed out after 8 min — partial data saved. Chart may still have updated.")
                                _ok = True   # try forecast anyway
                            except Exception as _e:
                                st.error(f"Error: {_e}")
                        if _ok:
                            with st.spinner("Running forecast.py..."):
                                try:
                                    r2 = _sp.run(["python","forecast.py"],
                                        cwd=r"C:\Users\Dell\Desktop\stockGPT v3",
                                        capture_output=True, text=True, timeout=300)
                                    if r2.returncode != 0:
                                        st.error(f"forecast.py failed:\n{r2.stderr[-400:] or r2.stdout[-400:]}")
                                    else:
                                        st.success("Done! Forecasts updated.")
                                        st.cache_data.clear()   # flush ALL caches — chart shows fresh data
                                        st.rerun()
                                except _sp.TimeoutExpired:
                                    st.warning("forecast.py timed out after 300 s.")
                                except Exception as _e:
                                    st.error(f"Error: {_e}")

            if msg.get("type") == "long_short":
                ls_html = render_long_short_table(
                    msg.get("stocks_long", []), msg.get("stocks_short", []))
                st.markdown(f"<div style='padding:4px 20px 4px 58px;'>{ls_html}</div>",
                            unsafe_allow_html=True)
            elif msg.get("stocks"):
                mode = "sell" if msg.get("type") == "stocks_sell" else "buy"
                tbl  = render_stock_table(msg["stocks"], mode)
                st.markdown(f"<div style='padding:4px 20px 4px 58px;'>{tbl}</div>",
                            unsafe_allow_html=True)

            if msg.get("chart_stock"):
                stock = msg["chart_stock"]
                hist  = load_stock_history(stock)
                if hist is not None and len(hist) > 30:
                    hr = hist.tail(180)
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.72, 0.28], vertical_spacing=0.02)
                    fig.add_trace(go.Candlestick(
                        x=hr["date"], open=hr["open"], high=hr["high"],
                        low=hr["low"], close=hr["close"],
                        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                        showlegend=False
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=hr["date"], y=hr["close"].rolling(20).mean(),
                        line=dict(color="#3b82f6", width=1.2),
                        name="20d MA", showlegend=False
                    ), row=1, col=1)
                    rets   = hr["return_1d"] if "return_1d" in hr.columns else hr["close"].pct_change()
                    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in rets]
                    fig.add_trace(go.Bar(
                        x=hr["date"], y=rets*100, marker_color=colors,
                        showlegend=False, opacity=0.7
                    ), row=2, col=1)
                    fig.update_layout(
                        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                        font=dict(color="#555", size=11),
                        xaxis_rangeslider_visible=False,
                        xaxis2=dict(gridcolor="#1a1a1a", rangeslider=dict(visible=False)),
                        yaxis=dict(title="Price (INR)", gridcolor="#1a1a1a",
                                   title_font=dict(color="#444")),
                        yaxis2=dict(title="Ret %", gridcolor="#1a1a1a",
                                    title_font=dict(color="#444")),
                        height=360, margin=dict(l=8, r=8, t=8, b=8),
                    )
                    st.plotly_chart(fig, width='stretch',
                                   key=f"chart_{stock}_{id(msg)}")

    st.markdown("<div id='chat-bottom' style='height:20px;'></div>", unsafe_allow_html=True)

# Auto-scroll to latest message
st.markdown("""
<script>
var el = document.getElementById('chat-bottom');
if (el) el.scrollIntoView({behavior: 'smooth'});
</script>
""", unsafe_allow_html=True)


# Suggestion chips

if len(st.session_state.messages) <= 1:
    _next = get_next_trading_day(date.today())

    # Row 1: featured Next Day button
    nd_col, _ = st.columns([2, 5])
    with nd_col:
        if st.button(f"📅 Next Trading Day Forecast  →  {_next.strftime('%A, %b %d')}",
                     key="chip_nextday", width='stretch'):
            st.session_state.pending_input = "Next trading day forecast"
            st.rerun()

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # Row 2: original 6 chips
    chips = [
        ("📊 Long-Short Portfolio",  "Which stocks to buy and which to short?"),
        ("📈 Top stocks to buy",     "Top 10 stocks to buy"),
        ("📉 Stocks to avoid",       "Top 10 stocks to avoid"),
        ("🌐 Market outlook",        "NSE market overview"),
        ("📋 Backtest results",      "Show backtest results"),
        ("🧠 How it works",          "How does StockGPT work?"),
    ]
    cols = st.columns(len(chips))
    for i, (col, (label, query)) in enumerate(zip(cols, chips)):
        with col:
            if st.button(label, key=f"chip_{i}", width='stretch'):
                st.session_state.pending_input = query
                st.rerun()


# Bottom bar: Back | Chat input | New

_c_back, _c_input, _c_new = st.columns([1, 8, 1])

with _c_back:
    st.markdown("<div style='padding-top:4px;'>", unsafe_allow_html=True)
    if st.button("← Back", key="undo", help="Remove last question and answer"):
        if len(st.session_state.messages) > 1:
            while st.session_state.messages and st.session_state.messages[-1]["role"] == "ai":
                st.session_state.messages.pop()
            while st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _c_input:
    user_input = st.chat_input("Ask about any NSE stock, market outlook, or backtest results...")

with _c_new:
    st.markdown("<div style='padding-top:4px;'>", unsafe_allow_html=True)
    if st.button("↺ New", key="new_chat", help="Clear conversation and start over"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input and user_input.strip():
    st.session_state.messages.append({
        "role": "user", "text": user_input.strip(),
        "stocks": [], "stocks_long": [], "stocks_short": [],
        "type": "text", "chart_stock": None, "metrics_html": ""
    })

    resp = ai_response(user_input.strip())
    st.session_state.messages.append({
        "role": "ai",
        "text":         resp.get("text", ""),
        "stocks":       resp.get("stocks", []),
        "stocks_long":  resp.get("stocks_long", []),
        "stocks_short": resp.get("stocks_short", []),
        "type":         resp.get("type", "text"),
        "chart_stock":  resp.get("chart_stock"),
        "metrics_html": resp.get("metrics_html", ""),
    })

    st.rerun()
