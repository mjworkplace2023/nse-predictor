# 🇮🇳 NSE Nifty50 Stock Predictor

A fully automated, demo-ready NSE stock prediction tool that scores all 50 Nifty stocks on a **-100 to +100 scale** using technical indicators, news sentiment, and price momentum — then sends Telegram alerts and displays a live Streamlit dashboard.

**No database required. Everything runs in memory.**

---

## Features

| Feature | Details |
|---|---|
| Data source | Yahoo Finance via `yfinance` (free, no API key) |
| Technical indicators | RSI 14, MACD 12/26/9, EMA 20/50, Bollinger Bands 20/2, Volume ratio |
| Sentiment | VADER on live RSS feeds (Moneycontrol, ET, LiveMint, Business Standard) |
| Scoring | Technical ±40 + Sentiment ±30 + Momentum ±30 = **Total ±100** |
| Alerts | Telegram bot message with top gainers/losers |
| Dashboard | Streamlit app with candlestick, RSI, volume charts |
| Automation | GitHub Actions cron job (9:30 AM IST weekdays) |
| Storage | **None** — pure in-memory processing |

---

## Quick Start (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/nse-predictor.git
cd nse-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Telegram credentials (optional for demo)
```

### 3. Run prediction (CLI)

```bash
python run_daily.py
```

Output:
```
TOP 5 POTENTIAL GAINERS
1. RELIANCE    Reliance Industries    Score=+42.3  ₹2,943.50  1D=+1.20%  BUY
...

TOP 5 POTENTIAL LOSERS
1. COALINDIA   Coal India             Score=-38.7  ₹430.10   1D=-2.10%  SELL
...
```

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) and click **Run Prediction Now**.

---

## Telegram Setup (Optional)

1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → copy your token
2. Message [@userinfobot](https://t.me/userinfobot) → copy your Chat ID
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:AAF...
   TELEGRAM_CHAT_ID=987654321
   ```

---

## GitHub Actions Setup

1. Push this repo to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. The workflow runs automatically at **9:30 AM IST every weekday**
5. Trigger manually via **Actions → NSE Daily Stock Prediction → Run workflow**

---

## Scoring Methodology

### Technical Score (max ±40)

| Indicator | Weight | Signal |
|---|---|---|
| RSI 14 | ±10 | < 30 = oversold (+), > 70 = overbought (-) |
| MACD histogram | ±10 | Positive + growing = bullish |
| EMA 20 vs EMA 50 | ±10 | Price > EMA20 > EMA50 = bullish |
| Bollinger Bands | ±5 | Near lower band = oversold (+) |
| Volume ratio (5d/20d) | ±5 | Spike = +, Drought = - |

### Sentiment Score (max ±30)

- Fetches headlines from 4 free RSS feeds
- Matches each headline to the stock by name/symbol keywords
- Applies VADER sentiment analysis
- Mean compound score × 30

### Momentum Score (max ±30)

| Period | Weight |
|---|---|
| 1-day return | ±15 |
| 5-day return | ±10 |
| 20-day return | ±5 |

### Signal Labels

| Score | Signal |
|---|---|
| ≥ +50 | STRONG BUY |
| +20 to +49 | BUY |
| -19 to +19 | NEUTRAL |
| -20 to -49 | SELL |
| ≤ -50 | STRONG SELL |

---

## Project Structure

```
nse-predictor/
├── run_daily.py              # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   └── nifty50_symbols.py   # Nifty 50 tickers + company names
├── predictor/
│   ├── fetch_data.py        # yfinance OHLCV downloader
│   ├── indicators.py        # Technical indicator scorers
│   ├── sentiment.py         # RSS + VADER sentiment
│   └── scorer.py            # Master scoring pipeline
├── alerts/
│   └── telegram_alert.py    # Telegram bot sender
├── dashboard/
│   └── app.py               # Streamlit dashboard
└── .github/
    └── workflows/
        └── daily_prediction.yml
```

---

## CLI Options

```
python run_daily.py --help

  --top N          Show top/bottom N stocks (default: 5)
  --no-alert       Skip Telegram notification
  --schedule       Run daily at 9:30 AM IST (weekdays, blocks)
```

---

## Disclaimer

**This tool is for educational and research purposes only. It is NOT financial advice. Past performance of technical signals does not guarantee future results. Always do your own research before investing.**

---

## License

MIT — free to use, modify, and distribute.
