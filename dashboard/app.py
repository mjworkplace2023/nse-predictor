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

from data.nifty50_symbols import NIFTY50_SYMBOLS, get_display_name
from predictor.scorer import run_prediction, result_to_dataframe, PredictionResult
from alerts.telegram_alert import send_prediction_alert

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSE Nifty50 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
top_n = st.sidebar.slider("Top N stocks", min_value=3, max_value=10, value=5)
send_telegram = st.sidebar.checkbox("Send Telegram alert after run", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "Scores NSE Nifty50 stocks using:\n"
    "- Technical indicators (±40)\n"
    "- News sentiment (±30)\n"
    "- Price momentum (±30)\n\n"
    "_Not financial advice._"
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🇮🇳 NSE Nifty50 Stock Predictor")
st.caption(
    "Combines RSI · MACD · EMA · Bollinger Bands · Volume · VADER Sentiment · Price Momentum"
)

col_run, col_info = st.columns([2, 5])
with col_run:
    run_button = st.button("🚀 Run Prediction Now", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if run_button:
    with st.spinner("Fetching data and computing scores — this takes ~60 s …"):
        result: PredictionResult = run_prediction(top_n=top_n)
        st.session_state.result = result

    if send_telegram:
        ok = send_prediction_alert(result)
        if ok:
            st.success("✅ Telegram alert sent!")
        else:
            st.warning("⚠️ Telegram alert failed — check credentials in .env")

result: PredictionResult = st.session_state.result

if result is None:
    st.info("👆 Click **Run Prediction Now** to start the analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Results — {result.run_timestamp}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stocks analysed", result.symbols_processed)
k2.metric("Failed / skipped", result.symbols_failed)
if result.top_gainers:
    k3.metric(
        "Top Gainer",
        result.top_gainers[0].symbol.replace(".NS", ""),
        f"{result.top_gainers[0].score:+.1f}",
    )
if result.top_losers:
    k4.metric(
        "Top Loser",
        result.top_losers[0].symbol.replace(".NS", ""),
        f"{result.top_losers[0].score:+.1f}",
    )

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_overview, tab_gainers, tab_losers, tab_detail = st.tabs(
    ["📋 Leaderboard", "🏆 Top Gainers", "⚠️ Top Losers", "🔍 Stock Detail"]
)

# ---- Leaderboard -----------------------------------------------------------
with tab_overview:
    df_all = result_to_dataframe(result)

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

    styled = (
        df_all.style
        .map(color_score, subset=["Score", "Technical", "Sentiment", "Momentum"])
        .map(color_pct, subset=["1D %", "5D %", "20D %"])
        .format({
            "Price (INR)": "₹{:,.2f}",
            "Score": "{:+.1f}",
            "Technical": "{:+.1f}",
            "Sentiment": "{:+.1f}",
            "Momentum": "{:+.1f}",
            "1D %": "{:+.2f}%",
            "5D %": "{:+.2f}%",
            "20D %": "{:+.2f}%",
            "RSI": "{:.1f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=600)

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
            c2.metric("1-Day Change", f"{s.change_1d:+.2f}%")
            c3.metric("5-Day Change", f"{s.change_5d:+.2f}%")
            c4.metric("RSI", f"{s.rsi:.1f}" if s.rsi else "—")
            st.write(f"Technical: {s.technical_score:+.1f} | Sentiment: {s.sentiment_score:+.1f} | Momentum: {s.momentum_score:+.1f}")
            st.write(f"News headlines matched: {s.sentiment_headline_count}")

# ---- Top Losers ------------------------------------------------------------
with tab_losers:
    st.subheader(f"⚠️ Top {top_n} Potential Losers")
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
            c2.metric("1-Day Change", f"{s.change_1d:+.2f}%")
            c3.metric("5-Day Change", f"{s.change_5d:+.2f}%")
            c4.metric("RSI", f"{s.rsi:.1f}" if s.rsi else "—")
            st.write(f"Technical: {s.technical_score:+.1f} | Sentiment: {s.sentiment_score:+.1f} | Momentum: {s.momentum_score:+.1f}")

# ---- Stock Detail ----------------------------------------------------------
with tab_detail:
    st.subheader("🔍 Detailed Stock Analysis")

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
            mc2.metric("1D %", f"{score_obj.change_1d:+.2f}%")
            mc3.metric("5D %", f"{score_obj.change_5d:+.2f}%")
            mc4.metric("20D %", f"{score_obj.change_20d:+.2f}%")
            mc5.metric("RSI", f"{score_obj.rsi:.1f}" if score_obj.rsi else "—")

        # Fetch chart data fresh
        with st.spinner("Loading chart data..."):
            try:
                ticker = yf.Ticker(selected)
                chart_df = ticker.history(period="3mo", interval="1d", auto_adjust=True)

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

                    # EMA 20 and 50
                    ema20 = ta_lib.trend.EMAIndicator(chart_df["Close"], window=20).ema_indicator()
                    ema50 = ta_lib.trend.EMAIndicator(chart_df["Close"], window=50).ema_indicator()
                    if ema20 is not None:
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=ema20,
                            name="EMA 20", line=dict(color="#4da6ff", width=1.5)
                        ))
                    if ema50 is not None:
                        fig_candle.add_trace(go.Scatter(
                            x=chart_df.index, y=ema50,
                            name="EMA 50", line=dict(color="#ffa64d", width=1.5)
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
                        title=f"{selected} — 3-Month Chart",
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
                    rsi_series = ta_lib.momentum.RSIIndicator(chart_df["Close"], window=14).rsi()
                    if rsi_series is not None:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=chart_df.index, y=rsi_series,
                            name="RSI 14", line=dict(color="#b366ff"),
                        ))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(
                            title="RSI (14)",
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
            breakdown_data = {
                "Component": ["Technical", "Sentiment", "Momentum"],
                "Score": [score_obj.technical_score, score_obj.sentiment_score, score_obj.momentum_score],
                "Max": [40, 30, 30],
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
