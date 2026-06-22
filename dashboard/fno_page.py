"""F&O Intraday Trading dashboard — Nifty 50, Bank Nifty, Sensex."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from fno.config import FNO_INDICES, PCR_BEARISH, PCR_BULLISH
from fno.formatting import format_index
from fno.models import lstm_available
from fno.option_trades import OptionTradeRecommendation, option_trades_to_dataframe
from fno.predictor import run_fno_intraday_prediction, results_to_dataframe

_CACHE_TTL_SEC = 300  # 5 min — avoids re-training on every click


@st.cache_data(ttl=_CACHE_TTL_SEC, show_spinner=False)
def _cached_fno_prediction(include_options: bool):
    return run_fno_intraday_prediction(include_options=include_options)


def _signal_color(signal: str) -> str:
    return {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}.get(signal, "#6b7280")


def _action_color(action: str) -> str:
    return {"CALL": "#16a34a", "PUT": "#dc2626", "WAIT": "#6b7280"}.get(action, "#6b7280")


def render_fno_page() -> None:
    index_names = ", ".join(i.name for i in FNO_INDICES)
    st.title("📈 F&O Intraday Trading")
    st.caption(
        f"ML + options analytics for **{index_names}**. "
        "Decision support only — always use stop-loss and paper trade first."
    )

    with st.expander("⚠️ Risk warning", expanded=False):
        st.markdown(
            "- Model predictions are probabilistic — never 100% accurate\n"
            "- Always use stop-loss with every trade\n"
            "- F&O carries unlimited loss risk — use position sizing\n"
            "- Treat as decision support, not auto-trading signal\n"
            "- Backtest and paper trade before live deployment"
        )

    col1, col2 = st.columns(2)
    with col1:
        include_options = st.checkbox("Include options chain (PCR, max pain, IV skew)", value=True)
    with col2:
        st.caption(f"LSTM: {'available' if lstm_available() else 'not installed'}")

    st.markdown(
        f"**Indices:** {index_names} · "
        f"**Options (NSE):** Nifty 50 & Bank Nifty · "
        f"**Options rules:** PCR > {PCR_BULLISH} bullish · PCR < {PCR_BEARISH} bearish"
    )
    st.caption("Results cached 5 min. Sensex uses ML only (options chain is on BSE).")

    force_refresh = st.button("🔄 Run F&O prediction", type="primary", use_container_width=True)

    if force_refresh:
        _cached_fno_prediction.clear()
        with st.spinner("Fetching index data & options (usually 15–30 sec)…"):
            results, option_trades = _cached_fno_prediction(include_options=include_options)
            st.session_state["fno_results"] = results
            st.session_state["fno_option_trades"] = option_trades
            st.session_state["fno_options_on"] = include_options

    results = st.session_state.get("fno_results", [])
    option_trades = st.session_state.get("fno_option_trades", [])

    if not results:
        st.info("Click **Run F&O prediction** to scan Nifty 50, Bank Nifty, and Sensex.")
        st.markdown("#### Pipeline")
        st.markdown(
            "1. **Data** — yfinance 15m OHLCV (parallel fetch)\n"
            "2. **Options** — NSE v3 API for NIFTY & BANKNIFTY (PCR, max pain, IV skew)\n"
            "3. **Features** — EMA, MACD, RSI, Stochastic, BB, ATR, OBV, VWAP\n"
            "4. **Labels** — BUY/HOLD/SELL from forward 15m returns (±0.5%)\n"
            "5. **Models** — fast Random Forest (time-series split)\n"
            "6. **Levels** — Entry range, target, stop-loss from session VWAP + ATR"
        )
        return

    df = results_to_dataframe(results)
    st.subheader("F&O Intraday — Index Signals")
    st.caption(f"As of {results[0].as_of}")

    options_ok = sum(1 for r in results if r.pcr is not None)
    if include_options and options_ok < 2:
        st.warning(
            "Options chain loaded for "
            f"{options_ok}/2 NSE indices. "
            "During off-market hours or if NSE blocks the server IP, only ML signals are shown."
        )

    def _style_signal(val: str) -> str:
        return f"color: {_signal_color(val)}; font-weight: 600"

    styled = df.style.map(_style_signal, subset=["Signal", "ML Signal", "Options"])
    # Index prices and levels — whole numbers only
    int_cols = ["Price", "Entry Low", "Entry High", "Target", "Stop Loss", "Max Pain"]
    for col in int_cols:
        if col in df.columns:
            styled = styled.format({col: lambda v: "—" if pd.isna(v) else f"{int(v):,}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Intraday Option Chain — Strike Recommendations")
    st.caption(
        "CALL on BUY signal · PUT on SELL signal · Target/SL on option premium (LTP). "
        "Sensex strike from spot (BSE chain not linked)."
    )
    if option_trades:
        opt_df = option_trades_to_dataframe(option_trades)
        opt_styled = opt_df.style.map(
            lambda v: f"color: {_action_color(v)}; font-weight: 600",
            subset=["Action"],
        )
        for col in ["Strike"]:
            if col in opt_df.columns:
                opt_styled = opt_styled.format({col: lambda v: "—" if pd.isna(v) else f"{int(v):,}"})
        for col in ["Entry (₹)", "Target (₹)", "Stop Loss (₹)"]:
            if col in opt_df.columns:
                opt_styled = opt_styled.format(
                    {col: lambda v: "—" if pd.isna(v) else f"{float(v):.1f}"}
                )
        st.dataframe(opt_styled, use_container_width=True, hide_index=True)
        with st.expander("Option trade notes"):
            for t in option_trades:
                st.markdown(f"**{t.index}** — {t.action} {t.strike:,}")
                st.caption(t.notes)
    else:
        st.info("Run prediction to generate option strike recommendations.")

    st.markdown("---")
    st.subheader("Signal distribution")
    counts = df["Signal"].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=[_signal_color(s) for s in counts.index],
            )
        ]
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed notes per index"):
        for r in results:
            st.markdown(f"**{r.symbol}** — {r.combined_signal} @ {format_index(r.price)}")
            if r.expiry_date and r.expiry_day:
                dte = f"{r.days_to_expiry} day(s) away" if r.days_to_expiry is not None else ""
                st.caption(f"Options expiry: **{r.expiry_date}** ({r.expiry_day}) {dte}")
            st.caption(r.notes)

    if st.session_state.get("fno_options_on"):
        st.markdown("---")
        st.subheader("Options snapshot")
        for r in results:
            st.markdown(f"**{r.symbol}**")
            if r.expiry_date and r.expiry_day:
                dte_text = f"{r.days_to_expiry} days to expiry" if r.days_to_expiry is not None else ""
                st.caption(f"Expiry: **{r.expiry_date}** · **{r.expiry_day}** · {dte_text}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Spot", format_index(r.price))
            c2.metric("PCR", f"{r.pcr:.2f}" if r.pcr is not None else "—")
            c3.metric("Max Pain", format_index(r.max_pain))
            c4.metric("IV Skew", f"{r.iv_skew:.2f}" if r.iv_skew is not None else "—")
            c5.metric("Options bias", r.options_signal)
