# Crypto Research Bot (BTC + ETH)

A beginner-friendly Streamlit dashboard for Android/mobile browsers.

## Features
- Public BTCUSDT and ETHUSDT candles from Binance (no account or API key)
- Price chart with EMA 20 and EMA 50
- RSI, MACD, volatility and transparent rule-based research score
- Market scanner
- Simplified historical backtest with configurable holding period, fees and slippage
- Session-only hypothetical paper-trade ledger

## Important
This is a research/education tool only. It does not execute trades, connect to exchange accounts, or provide calibrated probability forecasts. Backtest results are simplified and do not guarantee future performance. The paper ledger is stored only for the current app session and resets when the session ends.

## Deploy using an Android phone

1. Download and extract `crypto_research_bot.zip`.
2. In your browser, sign in to GitHub and create a new **public** repository, for example `crypto-research-bot`.
3. Upload `app.py` and `requirements.txt` to the repository root.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository, branch `main`, and main file path `app.py`.
6. Deploy. When deployment finishes, open the Streamlit URL on your Android phone and use **Add to Home screen** in Chrome if desired.

## Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data source
Binance public REST endpoint: `https://api.binance.com/api/v3/klines`
