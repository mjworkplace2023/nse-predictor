"""
Streamlit dashboard for NSE Stock Predictor.

Run with:
    streamlit run dashboard/app.py

Features:
  - Live prediction run button
  - Score leaderboard table
  - Top gainers / losers bar chart
  - Candlestick + indicator chart for any selected stock
  - Score breakdown radar chart
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
import ta as ta_lib
from dotenv import load_dotenv

load_dotenv()

from dashboard.auth import require_login, render_top_session_bar, current_user, LOGO_PATH
from dashboard.admin_page import render_admin_page
from dashboard.user_store import is_admin

from data.nifty50_symbols import NIFTY50_SYMBOLS, get_display_name
from predictor.scorer import (
    run_prediction,
    result_to_dataframe,
    PredictionResult,
    _is_intraday_result,
)
from predictor.intraday_scorer import run_intraday_prediction, get_intraday_config
from predictor.trade_levels import format_entry_range
from predictor.market_hours import is_market_open, market_status_message, now_ist
from alerts.telegram_alert import send_prediction_alert

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSE Nifty50 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_login()

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Dashboard help panels
# ---------------------------------------------------------------------------

SIGNAL_GUIDE = pd.DataFrame([
    {"Score range": "≥ +50", "Signal": "🚀 STRONG BUY"},
    {"Score range": "+20 to +49", "Signal": "📈 BUY"},
    {"Score range": "-19 to +19", "Signal": "➡️ NEUTRAL"},
    {"Score range": "-20 to -49", "Signal": "📉 SELL"},
    {"Score range": "≤ -50", "Signal": "🔴 STRONG SELL"},
])

DAILY_COLUMN_GUIDE = pd.DataFrame([
    {"Column": "Score", "Meaning": "Overall swing outlook (-100 bearish to +100 bullish)"},
    {"Column": "Signal", "Meaning": "BUY / SELL label based on Score"},
    {"Column": "Technical", "Meaning": "RSI, MACD, EMA, Bollinger Bands, Volume (max ±40)"},
    {"Column": "Sentiment", "Meaning": "News headline tone via VADER analysis (max ±30)"},
    {"Column": "Momentum", "Meaning": "Price trend strength across recent days (max ±30)"},
    {"Column": "1D %", "Meaning": "% price change over the last 1 trading day"},
    {"Column": "5D %", "Meaning": "% price change over the last 5 trading days"},
    {"Column": "20D %", "Meaning": "% price change over the last 20 trading days (~1 month)"},
    {"Column": "RSI", "Meaning": "14-day momentum; below 30 = oversold, above 70 = overbought"},
    {"Column": "News Count", "Meaning": "Number of news headlines matched to this stock"},
])

INTRADAY_COLUMN_GUIDE = pd.DataFrame([
    {"Column": "Score", "Meaning": "Overall outlook today (-100 bearish to +100 bullish)"},
    {"Column": "Signal", "Meaning": "BUY / SELL label based on Score"},
    {"Column": "Technical", "Meaning": "Chart indicators + VWAP strength (max ±45)"},
    {"Column": "Sentiment", "Meaning": "Today's news tone (max ±15)"},
    {"Column": "Momentum", "Meaning": "How strongly price is moving today (max ±40)"},
    {"Column": "Open %", "Meaning": "% change since today's 9:15 AM market open"},
    {"Column": "30m %", "Meaning": "% change in the last 30 minutes"},
    {"Column": "1h %", "Meaning": "% change in the last 1 hour"},
    {"Column": "RSI", "Meaning": "Short-term momentum; below 30 = oversold, above 70 = overbought"},
    {"Column": "Action", "Meaning": "LONG = buy setup, SHORT = sell setup, WAIT = no clear trade"},
    {"Column": "Entry Range", "Meaning": "Suggested price zone to enter the trade (₹ low – ₹ high)"},
    {"Column": "Intraday Target", "Meaning": "Same-day profit target from ATR + session high/low (intraday only)"},
    {"Column": "Stop Loss", "Meaning": "Suggested exit if trade goes wrong — limits loss"},
    {"Column": "R:R", "Meaning": "Risk-to-reward ratio (reward ÷ risk). Above 1.5 is healthier"},
])

# Leaderboard cell background colours for Signal / Action
_STYLE_BUY = "background-color: #c8e6c9; color: #1b5e20; font-weight: bold"
_STYLE_SELL = "background-color: #ffcdd2; color: #b71c1c; font-weight: bold"
_STYLE_NEUTRAL = "background-color: #ffe0b2; color: #e65100; font-weight: bold"


def _style_signal_or_action(val) -> str:
    """Return CSS for Signal / Action cells: green buy/long, red sell/short, orange neutral/wait."""
    if not isinstance(val, str):
        return ""
    upper = val.upper()
    if "BUY" in upper or upper == "LONG":
        return _STYLE_BUY
    if "SELL" in upper or upper == "SHORT":
        return _STYLE_SELL
    if upper in ("NEUTRAL", "WAIT", "➡️ NEUTRAL"):
        return _STYLE_NEUTRAL
    return _STYLE_NEUTRAL


def _build_leaderboard_dataframe(
    result: PredictionResult,
    *,
    show_intraday_ui: bool,
    has_intraday_trade_levels: bool,
) -> pd.DataFrame:
    """Build leaderboard table; intraday price levels only from real intraday runs."""
    df = result_to_dataframe(
        result,
        intraday_view=show_intraday_ui,
        include_trade_levels=has_intraday_trade_levels,
    )

    if "Target" in df.columns and "Intraday Target" not in df.columns:
        df = df.rename(columns={"Target": "Intraday Target"})

    if show_intraday_ui and not has_intraday_trade_levels:
        drop_cols = [
            c for c in ("Entry Range", "Intraday Target", "Stop Loss", "R:R")
            if c in df.columns
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    return df


def _columns_in(df: pd.DataFrame, *names: str) -> list[str]:
    """Return column names that exist in df (avoids Styler KeyError on subset)."""
    return [c for c in names if c in df.columns]


def _leaderboard_column_config(df: pd.DataFrame) -> dict:
    """Column config for leaderboard — pin Symbol & Company while scrolling."""
    config = {}
    if "Symbol" in df.columns:
        config["Symbol"] = st.column_config.TextColumn("Symbol", pinned="left", width="small")
    if "Company" in df.columns:
        config["Company"] = st.column_config.TextColumn("Company", pinned="left", width="medium")
    for col in ("Action", "Entry Range", "Intraday Target", "Stop Loss", "R:R"):
        if col in df.columns:
            config[col] = st.column_config.TextColumn(col, width="medium")
    return config


_TRADE_LEVEL_FIELDS = (
    "trade_action",
    "entry_low",
    "entry_high",
    "target_price",
    "stop_loss",
    "risk_reward",
)


def _trade_field(stock, name: str, default=None):
    """Safe accessor — old cached results may lack trade-level fields."""
    return getattr(stock, name, default)


def _patch_legacy_intraday_scores(result: PredictionResult) -> bool:
    """Backfill missing trade-level attrs on old session-cached StockScore objects."""
    if result is None or result.mode != "intraday" or not result.all_scores:
        return False
    patched = False
    for stock in result.all_scores:
        if hasattr(stock, "trade_action"):
            continue
        patched = True
        for field_name in _TRADE_LEVEL_FIELDS:
            if not hasattr(stock, field_name):
                setattr(stock, field_name, None)
    return patched


def _resolve_trade_action(stock) -> str:
    action = _trade_field(stock, "trade_action")
    if action:
        return action
    if stock.signal in ("BUY", "STRONG BUY"):
        return "LONG"
    if stock.signal in ("SELL", "STRONG SELL"):
        return "SHORT"
    return "WAIT"


def _render_intraday_trade_plan(stock) -> None:
    """Show entry / target / stop-loss box for a scored stock."""
    action = _resolve_trade_action(stock)

    entry_low = _trade_field(stock, "entry_low")
    entry_high = _trade_field(stock, "entry_high")
    target = _trade_field(stock, "target_price")
    stop_loss = _trade_field(stock, "stop_loss")
    rr = _trade_field(stock, "risk_reward")

    if action == "LONG" and target and stop_loss:
        st.success(
            f"**LONG** · Entry: {format_entry_range(entry_low, entry_high)} · "
            f"Target: ₹{target:,.2f} · Stop Loss: ₹{stop_loss:,.2f}"
            + (f" · R:R **{rr:.1f}**" if rr else "")
        )
    elif action == "SHORT" and target and stop_loss:
        st.error(
            f"**SHORT** · Entry: {format_entry_range(entry_low, entry_high)} · "
            f"Target: ₹{target:,.2f} · Stop Loss: ₹{stop_loss:,.2f}"
            + (f" · R:R **{rr:.1f}**" if rr else "")
        )
    elif action == "WAIT" and entry_low and entry_high:
        st.info(
            f"**WAIT** — No clear trade. Today's range: "
            f"₹{entry_low:,.2f} – ₹{entry_high:,.2f}"
        )
    elif action in ("LONG", "SHORT"):
        st.caption(
            f"**{action}** signal — re-run prediction to load entry, target, and stop-loss prices."
        )


def _render_trade_levels_explainer() -> None:
    with st.expander("📍 Trade Levels — Entry, Target & Stop Loss", expanded=False):
        st.markdown(
            "Levels are **rule-based estimates** from today's session data "
            "(VWAP, session high/low, ATR). They are **not guaranteed**.\n\n"
            "**LONG** (BUY signal):\n"
            "- **Entry Range** — buy between VWAP pullback and current price\n"
            "- **Target** — current price + 2× ATR (or session high)\n"
            "- **Stop Loss** — below price − 1× ATR or session low\n\n"
            "**SHORT** (SELL signal):\n"
            "- **Entry Range** — sell/short between current price and VWAP rally zone\n"
            "- **Target** — current price − 2× ATR (or session low)\n"
            "- **Stop Loss** — above price + 1× ATR or session high\n\n"
            "**R:R** = Reward ÷ Risk. Example: R:R 2.0 means potential gain is 2× the risk."
        )


def _render_daily_sidebar_help() -> None:
    st.sidebar.markdown("**📖 How Daily Works**")
    st.sidebar.markdown(
        "Scores all Nifty 50 stocks using **daily closing prices** "
        "(3 months of history).\n\n"
        "- Best for **swing trades** held days to weeks\n"
        "- Run once each morning before market open\n"
        "- Combines charts, news, and multi-day momentum\n\n"
        "**Score range:** -100 (bearish) to +100 (bullish)"
    )


def _render_daily_welcome_banner() -> None:
    st.info(
        "**ℹ️ Daily Swing Mode** — Which stocks look strong or weak "
        "over the **coming days and weeks**?\n\n"
        "1. Click **Run Prediction Now** (~60 seconds)\n"
        "2. Check the **Leaderboard** — all 50 stocks ranked by score\n"
        "3. Review **1D %**, **5D %**, **20D %** for price trend context\n"
        "4. Optional: enable **Telegram alert** in the sidebar after each run\n\n"
        "⚠️ _Not financial advice. Always do your own research._"
    )


def _render_signal_guide() -> None:
    with st.expander("📊 Signal Guide", expanded=False):
        st.dataframe(SIGNAL_GUIDE, hide_index=True, use_container_width=True)


def _render_daily_score_explainer() -> None:
    with st.expander("What makes up the Score?", expanded=False):
        st.markdown(
            "**Technical (±40)** — RSI, MACD, EMA crossovers, Bollinger Bands, Volume\n"
            "- RSI below 30 = oversold (bullish signal)\n"
            "- RSI above 70 = overbought (bearish signal)\n\n"
            "**Sentiment (±30)** — News headlines matched to each stock, "
            "scored with VADER sentiment analysis\n\n"
            "**Momentum (±30)** — Multi-day price trends:\n"
            "- **1D %** = last 1 trading day (weight: ±15)\n"
            "- **5D %** = last 5 trading days (weight: ±10)\n"
            "- **20D %** = last 20 trading days (weight: ±5)"
        )


def _render_daily_column_guide() -> None:
    with st.expander("📋 Leaderboard — Column Guide", expanded=False):
        st.dataframe(DAILY_COLUMN_GUIDE, hide_index=True, use_container_width=True)
        st.caption(
            "💡 Quick tip: Look for **BUY** stocks with positive **1D %** "
            "AND positive **5D %** — short- and medium-term momentum aligned."
        )


def _render_intraday_welcome_banner(interval: str) -> None:
    st.info(
        f"**ℹ️ Intraday Mode ({interval})** — Which stocks look strong or weak "
        f"**right now today**?\n\n"
        "1. Select **Intraday (15-min)** in the sidebar\n"
        "2. Click **Run Prediction Now** (~90 seconds)\n"
        "3. Check the **Leaderboard** — score, entry range, target & stop loss\n"
        "4. Optional: enable **Auto-refresh** for live updates every 5 min\n\n"
        "⚠️ _Not financial advice. Always use your own stop-loss._"
    )


def _render_intraday_score_explainer() -> None:
    with st.expander("What makes up the Score?", expanded=False):
        st.markdown(
            "**Technical (±45)** — RSI, MACD, EMA, Bollinger Bands, VWAP, Volume  \n"
            "VWAP = average price traded today; price above VWAP is bullish\n\n"
            "**Sentiment (±15)** — Today's news headlines (lower weight for intraday)\n\n"
            "**Momentum (±40)** — How price is moving today:\n"
            "- **Open %** = change since 9:15 AM market open\n"
            "- **30m %** = change in the last 30 minutes\n"
            "- **1h %** = change in the last 1 hour"
        )


def _render_intraday_column_guide() -> None:
    with st.expander("📋 Leaderboard — Column Guide", expanded=False):
        st.dataframe(INTRADAY_COLUMN_GUIDE, hide_index=True, use_container_width=True)
        st.caption(
            "💡 Quick tip: Look for **BUY** stocks with positive **Open %** "
            "AND positive **30m %** — momentum continuing through the session."
        )


# ---------------------------------------------------------------------------
# Top session bar (all authenticated pages)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .home-header-row { display: flex; align-items: center; gap: 1rem; }
    .home-logo img { max-height: 72px; width: auto; display: block; }
    .home-brand-main {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: #1a237e;
        margin: 0;
        line-height: 1.2;
    }
    .home-brand-sub {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: #546e7a;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin: 0.2rem 0 0 0;
    }
    .brand-credit {
        text-align: center !important;
        font-size: 0.75rem !important;
        color: #78909c !important;
        margin: 0.1rem 0 0 0 !important;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_logo_col, _title_col, _session_col = st.columns([1.4, 3.6, 1.2])
with _logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown(
            '<p class="home-brand-main">Butterfly</p>'
            '<p class="home-brand-sub">Investment</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="brand-credit">Created By : MJ Workplace</p>',
            unsafe_allow_html=True,
        )
with _title_col:
    st.markdown(
        '<p class="home-brand-main">🇮🇳 NSE Nifty50</p>'
        '<p class="home-brand-sub">Stock Predictor</p>',
        unsafe_allow_html=True,
    )
with _session_col:
    render_top_session_bar()
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

_nav_options = ["📊 Predictor"]
if is_admin(current_user()):
    _nav_options.append("👤 Admin")
_page = st.sidebar.radio("Page", _nav_options)

if _page == "👤 Admin":
    render_admin_page()
    st.stop()

prediction_mode = st.sidebar.radio(
    "Prediction mode",
    options=["Daily (Swing)", "Intraday (15-min)"],
    index=0,
    help="Daily = end-of-day swing view. 15-min = intraday signals with less noise.",
)
top_n = st.sidebar.slider("Top N stocks", min_value=3, max_value=10, value=5)
is_intraday = prediction_mode == "Intraday (15-min)"
intraday_interval = "15m"

if is_intraday:
    st.sidebar.success(
        "Intraday (15-min) selected — leaderboard includes "
        "Action, Entry Range, Intraday Target, Stop Loss"
    )
else:
    st.sidebar.info(
        "Daily mode — switch to **Intraday (15-min)** for trade-level columns"
    )

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh every 5 min",
    value=False,
    disabled=not is_intraday,
    help="Only runs during NSE market hours (09:15–15:30 IST, weekdays).",
)
send_telegram = st.sidebar.checkbox("Send Telegram alert after run", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(market_status_message())

st.sidebar.markdown("---")
if not is_intraday:
    _render_daily_sidebar_help()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "last_auto_refresh" not in st.session_state:
    st.session_state.last_auto_refresh = None
if "last_prediction_mode" not in st.session_state:
    st.session_state.last_prediction_mode = prediction_mode

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
if is_intraday:
    st.caption(
        f"Intraday ({intraday_interval}): RSI · MACD · EMA · Bollinger · VWAP · Volume · "
        "Sentiment · Open/30m/1h Momentum"
    )
else:
    st.caption(
        "Daily: RSI · MACD · EMA · Bollinger Bands · Volume · VADER Sentiment · Price Momentum"
    )

col_run, col_info = st.columns([2, 5])
with col_run:
    run_button = st.button("🚀 Run Prediction Now", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run prediction (manual + auto-refresh)
# ---------------------------------------------------------------------------

def _run_prediction(intraday: bool, interval: str, n: int) -> PredictionResult:
    if intraday:
        return run_intraday_prediction(top_n=n, interval=interval)
    return run_prediction(top_n=n)


def _dispatch_alerts(res: PredictionResult, telegram: bool) -> None:
    if telegram:
        ok, detail = send_prediction_alert(res)
        if ok:
            st.success("✅ Telegram alert sent!")
        else:
            st.warning(f"⚠️ Telegram alert failed — {detail}")


if run_button:
    st.session_state.last_prediction_mode = prediction_mode
    spinner_msg = (
        f"Fetching {intraday_interval} intraday data — this takes ~90 s …"
        if is_intraday
        else "Fetching data and computing scores — this takes ~60 s …"
    )
    with st.spinner(spinner_msg):
        result: PredictionResult = _run_prediction(is_intraday, intraday_interval, top_n)
        st.session_state.result = result
    _dispatch_alerts(result, send_telegram)


@st.fragment(run_every=datetime.timedelta(minutes=5))
def intraday_auto_refresh():
    """Re-run intraday prediction every 5 minutes during market hours."""
    if not (is_intraday and auto_refresh):
        return
    if not is_market_open():
        st.sidebar.warning("Auto-refresh paused — market closed.")
        return

    with st.spinner(f"Auto-refreshing {intraday_interval} intraday data…"):
        refreshed = _run_prediction(True, intraday_interval, top_n)
        st.session_state.result = refreshed
        st.session_state.last_auto_refresh = now_ist().strftime("%H:%M:%S IST")

    st.toast(f"Refreshed at {st.session_state.last_auto_refresh}", icon="🔄")


intraday_auto_refresh()

# Clear cached results when user switches Daily ↔ Intraday mode
if st.session_state.last_prediction_mode != prediction_mode:
    st.session_state.result = None
    st.session_state.last_prediction_mode = prediction_mode
    st.session_state.last_auto_refresh = None

result: PredictionResult = st.session_state.result

# Normalize legacy results that predate mode="intraday" tagging
if result is not None and _is_intraday_result(result):
    result.mode = "intraday"

show_intraday_ui = is_intraday or _is_intraday_result(result)
has_intraday_trade_levels = _is_intraday_result(result)

if result is not None and is_intraday and not has_intraday_trade_levels:
    st.error(
        "**Intraday mode is selected** but your last run was **Daily (Swing)**. "
        "Intraday target, entry, and stop-loss columns appear only after you "
        "click **Run Prediction Now** in intraday mode."
    )

if _patch_legacy_intraday_scores(result):
    st.info(
        "Cached results loaded. Click **Run Prediction Now** to refresh "
        "entry, target, and stop-loss levels."
    )

if result is None:
    if is_intraday:
        _render_intraday_welcome_banner(intraday_interval)
        if auto_refresh:
            st.caption(
                "Auto-refresh is on — first run will start during market hours (09:15–15:30 IST), "
                "or click **Run Prediction Now** to run immediately."
            )
    else:
        _render_daily_welcome_banner()
    st.stop()

if st.session_state.last_auto_refresh:
    st.caption(f"Last auto-refresh: {st.session_state.last_auto_refresh}")

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Results — {result.run_timestamp}")

if result.mode == "intraday" or show_intraday_ui:
    interval_label = getattr(result, "interval", intraday_interval)
    st.caption(f"Intraday mode · {interval_label} candles · same-day trading signals")
    if getattr(result, "intraday_session_fallback", False) and result.intraday_session_date:
        st.info(
            f"Market just opened — using **{result.intraday_session_date}** session data "
            f"until enough {interval_label} bars accumulate today (need ~2 hours of trading)."
        )
    elif result.symbols_processed == 0:
        st.warning(
            "No intraday scores yet. This usually happens right after market open "
            "(before enough 15-min candles exist) or when Yahoo Finance returns no data. "
            "Try again after **11:00 IST**, or check market hours."
        )
else:
    st.caption("Daily swing mode · end-of-day data · multi-day outlook")

k1, k2, k3, k4 = st.columns(4)
if result.mode == "intraday":
    k1.metric(
        "Stocks analysed",
        result.symbols_processed,
        help="Nifty 50 stocks successfully scored in this run",
    )
    k2.metric(
        "Failed / skipped",
        result.symbols_failed,
        help="Stocks with missing or insufficient intraday candle data",
    )
    if result.top_gainers:
        k3.metric(
            "Top Gainer",
            result.top_gainers[0].symbol.replace(".NS", ""),
            f"{result.top_gainers[0].score:+.1f}",
            help="Highest intraday score right now — strongest bullish signals today",
        )
    if result.top_losers:
        k4.metric(
            "Top Loser",
            result.top_losers[0].symbol.replace(".NS", ""),
            f"{result.top_losers[0].score:+.1f}",
            help="Lowest intraday score right now — strongest bearish signals today",
        )
else:
    k1.metric(
        "Stocks analysed",
        result.symbols_processed,
        help="Nifty 50 stocks successfully scored using daily price data",
    )
    k2.metric(
        "Failed / skipped",
        result.symbols_failed,
        help="Stocks where daily price or indicator data could not be fetched",
    )
    if result.top_gainers:
        k3.metric(
            "Top Gainer",
            result.top_gainers[0].symbol.replace(".NS", ""),
            f"{result.top_gainers[0].score:+.1f}",
            help="Highest swing score — strongest bullish signals across indicators and news",
        )
    if result.top_losers:
        k4.metric(
            "Top Loser",
            result.top_losers[0].symbol.replace(".NS", ""),
            f"{result.top_losers[0].score:+.1f}",
            help="Lowest swing score — strongest bearish signals across indicators and news",
        )

_render_signal_guide()
if result.mode == "intraday":
    _render_intraday_score_explainer()
    _render_trade_levels_explainer()
else:
    _render_daily_score_explainer()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_overview, tab_gainers, tab_losers, tab_detail = st.tabs(
    ["📋 Leaderboard", "🏆 Top Gainers", "⚠️ Top Losers", "🔍 Stock Detail"]
)

# ---- Leaderboard -----------------------------------------------------------
with tab_overview:
    if show_intraday_ui:
        _render_intraday_column_guide()
    else:
        _render_daily_column_guide()

    df_all = _build_leaderboard_dataframe(
        result,
        show_intraday_ui=show_intraday_ui,
        has_intraday_trade_levels=has_intraday_trade_levels,
    )

    if df_all.empty:
        st.warning(
            "No stock data to display. "
            + (
                "Intraday needs enough 15-min candles — try after 11:00 IST, "
                "or wait for the session fallback banner above."
                if show_intraday_ui
                else "Click **Run Prediction Now** to refresh."
            )
        )
    else:
        mom_cols = _columns_in(
            df_all,
            *(
                ("Open %", "30m %", "1h %")
                if show_intraday_ui
                else ("1D %", "5D %", "20D %")
            ),
        )
        score_cols = _columns_in(
            df_all, "Score", "Technical", "Sentiment", "Momentum"
        )

        def color_score(val):
            if isinstance(val, (int, float)):
                if val >= 20:
                    return "color: #00cc44; font-weight: bold"
                elif val <= -20:
                    return "color: #ff4444; font-weight: bold"
            return ""

        def color_pct(val):
            if isinstance(val, (int, float)):
                return "color: #00cc44" if val > 0 else "color: #ff4444" if val < 0 else ""
            return ""

        format_dict = {
            col: fmt
            for col, fmt in {
                "Price (INR)": "₹{:,.2f}",
                "Score": "{:+.1f}",
                "Technical": "{:+.1f}",
                "Sentiment": "{:+.1f}",
                "Momentum": "{:+.1f}",
                "RSI": "{:.1f}",
                "Open %": "{:+.2f}%",
                "30m %": "{:+.2f}%",
                "1h %": "{:+.2f}%",
                "1D %": "{:+.2f}%",
                "5D %": "{:+.2f}%",
                "20D %": "{:+.2f}%",
                "Intraday Target": "₹{:,.2f}",
                "Target": "₹{:,.2f}",
                "Stop Loss": "₹{:,.2f}",
                "R:R": "{:.1f}",
            }.items()
            if col in df_all.columns
        }

        signal_action_cols = _columns_in(df_all, "Signal", "Action")

        styled = df_all.style.format(format_dict, na_rep="—")
        if score_cols:
            styled = styled.map(color_score, subset=score_cols)
        if mom_cols:
            styled = styled.map(color_pct, subset=mom_cols)
        if signal_action_cols:
            styled = styled.map(_style_signal_or_action, subset=signal_action_cols)

        column_config = _leaderboard_column_config(df_all)

        st.dataframe(
            styled,
            use_container_width=True,
            height=600,
            column_order=list(df_all.columns),
            column_config=column_config,
        )

    # Download button
    csv = df_all.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"nse_prediction_{datetime.date.today()}.csv",
        mime="text/csv",
    )

# ---- Top Gainers -----------------------------------------------------------
with tab_gainers:
    st.subheader(f"🏆 Top {top_n} Potential Gainers")
    if result.mode == "intraday":
        st.caption(
            "Stocks with the **highest intraday score** right now. "
            "A high score means bullish signals across technicals, momentum, and news. "
            "Expand any row for full details."
        )
    else:
        st.caption(
            "Stocks with the **highest swing score** — best candidates for "
            "multi-day holds. Strong technicals, positive news, and upward momentum. "
            "Expand any row for full details."
        )
    gainers = result.top_gainers

    # Horizontal bar chart
    fig_g = go.Figure()
    labels = [s.symbol.replace(".NS", "") for s in gainers]
    scores = [s.score for s in gainers]
    colors = ["#00cc44"] * len(gainers)

    fig_g.add_trace(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{sc:+.1f}" for sc in scores],
        textposition="outside",
    ))
    fig_g.update_layout(
        title="Composite Score",
        xaxis_title="Score (-100 to +100)",
        yaxis={"categoryorder": "total ascending"},
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # Score breakdown stacked bar
    fig_breakdown = go.Figure()
    cats = ["Technical", "Sentiment", "Momentum"]
    for cat, attr, color in zip(
        cats,
        ["technical_score", "sentiment_score", "momentum_score"],
        ["#4da6ff", "#ffa64d", "#66cc66"],
    ):
        fig_breakdown.add_trace(go.Bar(
            name=cat,
            x=labels,
            y=[getattr(s, attr) for s in gainers],
            marker_color=color,
        ))
    fig_breakdown.update_layout(
        barmode="stack",
        title="Score Breakdown",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    # Cards
    for s in gainers:
        with st.expander(f"📈 {s.symbol.replace('.NS','')} — {s.name}  |  Score: {s.score:+.1f}  |  {s.signal}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"₹{s.price:,.2f}")
            if result.mode == "intraday":
                c2.metric("Since Open", f"{s.change_1d:+.2f}%")
                c3.metric("30-Min Change", f"{s.change_5d:+.2f}%")
            else:
                c2.metric("1-Day Change", f"{s.change_1d:+.2f}%")
                c3.metric("5-Day Change", f"{s.change_5d:+.2f}%")
            c4.metric("RSI", f"{s.rsi:.1f}" if s.rsi else "—")
            st.write(f"Technical: {s.technical_score:+.1f} | Sentiment: {s.sentiment_score:+.1f} | Momentum: {s.momentum_score:+.1f}")
            st.write(f"News headlines matched: {s.sentiment_headline_count}")
            if result.mode == "intraday":
                _render_intraday_trade_plan(s)

# ---- Top Losers ------------------------------------------------------------
with tab_losers:
    st.subheader(f"⚠️ Top {top_n} Potential Losers")
    if result.mode == "intraday":
        st.caption(
            "Stocks with the **lowest intraday score** right now. "
            "A low score means bearish signals today — useful for avoiding weak names "
            "or spotting potential shorts (with your own analysis)."
        )
    else:
        st.caption(
            "Stocks with the **lowest swing score** — weakest technicals, "
            "negative news, and downward momentum. Useful for stocks to avoid "
            "or watch for further decline."
        )
    losers = result.top_losers

    fig_l = go.Figure()
    labels_l = [s.symbol.replace(".NS", "") for s in losers]
    scores_l = [s.score for s in losers]

    fig_l.add_trace(go.Bar(
        x=scores_l,
        y=labels_l,
        orientation="h",
        marker_color=["#ff4444"] * len(losers),
        text=[f"{sc:+.1f}" for sc in scores_l],
        textposition="outside",
    ))
    fig_l.update_layout(
        title="Composite Score",
        xaxis_title="Score (-100 to +100)",
        yaxis={"categoryorder": "total descending"},
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_l, use_container_width=True)

    for s in losers:
        with st.expander(f"📉 {s.symbol.replace('.NS','')} — {s.name}  |  Score: {s.score:+.1f}  |  {s.signal}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"₹{s.price:,.2f}")
            if result.mode == "intraday":
                c2.metric("Since Open", f"{s.change_1d:+.2f}%")
                c3.metric("30-Min Change", f"{s.change_5d:+.2f}%")
            else:
                c2.metric("1-Day Change", f"{s.change_1d:+.2f}%")
                c3.metric("5-Day Change", f"{s.change_5d:+.2f}%")
            c4.metric("RSI", f"{s.rsi:.1f}" if s.rsi else "—")
            st.write(f"Technical: {s.technical_score:+.1f} | Sentiment: {s.sentiment_score:+.1f} | Momentum: {s.momentum_score:+.1f}")
            if result.mode == "intraday":
                _render_intraday_trade_plan(s)

# ---- Stock Detail ----------------------------------------------------------
with tab_detail:
    st.subheader("🔍 Detailed Stock Analysis")
    if result.mode == "intraday":
        chart_iv = getattr(result, "interval", intraday_interval)
        st.caption(
            f"Pick a stock to see its **{chart_iv} chart**, RSI, volume, and score breakdown. "
            "Use this to confirm a Leaderboard pick before acting."
        )
    else:
        st.caption(
            "Pick a stock to see its **3-month daily chart**, EMA, Bollinger Bands, "
            "RSI, volume, and score breakdown. Use this to validate a Leaderboard pick."
        )

    all_syms = [s.symbol for s in result.all_scores]
    selected = st.selectbox(
        "Select a stock",
        options=all_syms,
        format_func=lambda x: f"{x.replace('.NS','')} — {get_display_name(x)}",
    )

    if selected:
        # Find score object
        score_obj = next((s for s in result.all_scores if s.symbol == selected), None)

        if score_obj:
            st.markdown(
                f"### {score_obj.name}  "
                f"{'🚀' if 'BUY' in score_obj.signal else '🔴' if 'SELL' in score_obj.signal else '➡️'} "
                f"**{score_obj.signal}** — Score: **{score_obj.score:+.1f}**"
            )

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Price", f"₹{score_obj.price:,.2f}")
            if result.mode == "intraday":
                mc2.metric("Since Open", f"{score_obj.change_1d:+.2f}%")
                mc3.metric("30m %", f"{score_obj.change_5d:+.2f}%")
                mc4.metric("1h %", f"{score_obj.change_20d:+.2f}%")
            else:
                mc2.metric("1D %", f"{score_obj.change_1d:+.2f}%")
                mc3.metric("5D %", f"{score_obj.change_5d:+.2f}%")
                mc4.metric("20D %", f"{score_obj.change_20d:+.2f}%")
            mc5.metric("RSI", f"{score_obj.rsi:.1f}" if score_obj.rsi else "—")

            if result.mode == "intraday":
                st.markdown("#### 📍 Intraday Trade Plan")
                _render_intraday_trade_plan(score_obj)

        # Fetch chart data fresh
        with st.spinner("Loading chart data..."):
            try:
                ticker = yf.Ticker(selected)
                if result.mode == "intraday":
                    chart_interval = getattr(result, "interval", "5m")
                    cfg = get_intraday_config(chart_interval)
                    chart_df = ticker.history(
                        period=cfg.period, interval=chart_interval, auto_adjust=True
                    )
                    chart_title = f"{selected} — {chart_interval} Intraday Chart"
                    ema_fast, ema_slow = cfg.ema_fast, cfg.ema_slow
                else:
                    chart_df = ticker.history(period="3mo", interval="1d", auto_adjust=True)
                    chart_title = f"{selected} — 3-Month Chart"
                    ema_fast, ema_slow = 20, 50

                if not chart_df.empty:
                    # Candlestick
                    fig_candle = go.Figure()

                    fig_candle.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df["Open"],
                        high=chart_df["High"],
                        low=chart_df["Low"],
                        close=chart_df["Close"],
                        name="OHLC",
                        increasing_line_color="#00cc44",
                        decreasing_line_color="#ff4444",
                    ))

                    ema20 = ta_lib.trend.EMAIndicator(chart_df["Close"], window=ema_fast).ema_indicator()
                    ema50 = ta_lib.trend.EMAIndicator(chart_df["Close"], window=ema_slow).ema_indicator()
                    if ema20 is not None:
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=ema20,
                            name=f"EMA {ema_fast}", line=dict(color="#4da6ff", width=1.5)
                        ))
                    if ema50 is not None:
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=ema50,
                            name=f"EMA {ema_slow}", line=dict(color="#ffa64d", width=1.5)
                        ))

                    # Bollinger Bands
                    bb = ta_lib.volatility.BollingerBands(chart_df["Close"], window=20, window_dev=2)
                    bb_upper = bb.bollinger_hband()
                    bb_lower = bb.bollinger_lband()
                    if bb_upper is not None:
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=bb_upper,
                            name="BB Upper", line=dict(color="gray", dash="dot", width=1)
                        ))
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=bb_lower,
                            name="BB Lower", line=dict(color="gray", dash="dot", width=1),
                            fill="tonexty", fillcolor="rgba(128,128,128,0.05)",
                        ))

                    fig_candle.update_layout(
                        title=chart_title,
                        xaxis_title="Date",
                        yaxis_title="Price (INR)",
                        xaxis_rangeslider_visible=False,
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_candle, use_container_width=True)

                    # Volume bar chart
                    fig_vol = px.bar(
                        x=chart_df.index,
                        y=chart_df["Volume"],
                        title="Volume",
                        color=chart_df["Close"].diff().apply(lambda x: "Up" if x >= 0 else "Down"),
                        color_discrete_map={"Up": "#00cc44", "Down": "#ff4444"},
                    )
                    fig_vol.update_layout(
                        height=200,
                        showlegend=False,
                        margin=dict(t=30, b=10),
                        xaxis_title="",
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    # RSI chart
                    if result.mode == "intraday":
                        rsi_window = get_intraday_config(
                            getattr(result, "interval", "5m")
                        ).rsi_window
                    else:
                        rsi_window = 14
                    rsi_series = ta_lib.momentum.RSIIndicator(chart_df["Close"], window=rsi_window).rsi()
                    if rsi_series is not None:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=chart_df.index, y=rsi_series,
                            name=f"RSI {rsi_window}", line=dict(color="#b366ff"),
                        ))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(
                            title=f"RSI ({rsi_window})",
                            height=200,
                            yaxis=dict(range=[0, 100]),
                            margin=dict(t=30, b=10),
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)

            except Exception as exc:
                st.error(f"Chart error: {exc}")

        # Score breakdown
        if score_obj:
            st.markdown("#### Score Breakdown")
            if result.mode == "intraday":
                max_vals = [45, 15, 40]
            else:
                max_vals = [40, 30, 30]
            breakdown_data = {
                "Component": ["Technical", "Sentiment", "Momentum"],
                "Score": [score_obj.technical_score, score_obj.sentiment_score, score_obj.momentum_score],
                "Max": max_vals,
            }
            bd_df = pd.DataFrame(breakdown_data)
            bd_df["% of Max"] = (bd_df["Score"] / bd_df["Max"] * 100).round(1)

            fig_radar = go.Figure(go.Bar(
                x=bd_df["Component"],
                y=bd_df["Score"],
                marker_color=["#4da6ff", "#ffa64d", "#66cc66"],
                text=bd_df["Score"].apply(lambda x: f"{x:+.1f}"),
                textposition="outside",
            ))
            fig_radar.update_layout(
                yaxis=dict(range=[-40, 40]),
                height=300,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚡ NSE Stock Predictor — Data: Yahoo Finance · Indicators: pandas-ta · "
    "Sentiment: VADER · Built with Streamlit · Not financial advice."
)
