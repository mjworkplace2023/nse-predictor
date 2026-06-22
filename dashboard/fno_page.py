"""F&O Intraday Trading dashboard — Nifty 50, Bank Nifty, Sensex."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from fno.config import FNO_INDICES, PCR_BEARISH, PCR_BULLISH
from fno.models import lstm_available
from fno.predictor import run_fno_intraday_prediction, results_to_dataframe


def _signal_color(signal: str) -> str:
    return {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}.get(signal, "#6b7280")


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
        f"**Options rules:** PCR > {PCR_BULLISH} bullish · PCR < {PCR_BEARISH} bearish · "
        "Max pain pull · IV skew · OI buildup"
    )

    if st.button("🔄 Run F&O prediction", type="primary", use_container_width=True):
        with st.spinner("Fetching index data & training models for Nifty, Bank Nifty, Sensex…"):
            results = run_fno_intraday_prediction(include_options=include_options)
            st.session_state["fno_results"] = results
            st.session_state["fno_options_on"] = include_options

    results = st.session_state.get("fno_results", [])
    if not results:
        st.info("Click **Run F&O prediction** to scan Nifty 50, Bank Nifty, and Sensex.")
        st.markdown("#### Pipeline")
        st.markdown(
            "1. **Data** — yfinance OHLCV (15m/1h/1d) for each index\n"
            "2. **Options** — nsepython chain per index (NIFTY / BANKNIFTY / SENSEX)\n"
            "3. **Features** — EMA, MACD, RSI, Stochastic, BB, ATR, OBV, VWAP, volume\n"
            "4. **Labels** — BUY/HOLD/SELL from forward 15m returns (±0.5%)\n"
            "5. **Models** — XGBoost + LightGBM + RF + LR ensemble (time-series split)\n"
            "6. **Levels** — Entry range, target, stop-loss from session VWAP + ATR"
        )
        return

    df = results_to_dataframe(results)
    st.subheader("F&O Intraday — Index Signals")
    st.caption(f"As of {results[0].as_of}")

    def _style_signal(val: str) -> str:
        return f"color: {_signal_color(val)}; font-weight: 600"

    styled = df.style.map(_style_signal, subset=["Signal", "ML Signal", "Options"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

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
            st.markdown(f"**{r.symbol}** — {r.combined_signal} @ {r.price:,.2f}")
            st.caption(r.notes)

    if st.session_state.get("fno_options_on"):
        st.markdown("---")
        st.subheader("Options snapshot")
        for r in results:
            st.markdown(f"**{r.symbol}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PCR", f"{r.pcr:.2f}" if r.pcr is not None else "—")
            c2.metric("Max Pain", f"{r.max_pain:,.0f}" if r.max_pain else "—")
            c3.metric("IV Skew", f"{r.iv_skew:.2f}" if r.iv_skew is not None else "—")
            c4.metric("Options bias", r.options_signal)
