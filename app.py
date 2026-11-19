import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Crypto Research Bot", page_icon="📊", layout="wide")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = {"Bitcoin (BTC)": "BTCUSDT", "Ethereum (ETH)": "ETHUSDT"}
INTERVALS = {"1 hour": "1h", "4 hours": "4h", "1 day": "1d"}

st.title("📊 Crypto Research Bot")
st.caption("BTC + ETH • Market scanner • Rule-based research signals • Backtesting • Paper trading")
st.warning(
    "Research and simulation only. This app does not place trades or connect to an exchange account. "
    "Signals are rule-based scores, not calibrated probabilities or guarantees of future returns."
)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    response = requests.get(
        BINANCE_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
        headers={"User-Agent": "CryptoResearchBot/1.0"},
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError("The market-data provider returned no candle data.")
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["return"] = d["close"].pct_change()
    d["volatility"] = d["return"].rolling(24).std() * np.sqrt(24)
    # Transparent heuristic score from -1 to +1, mapped to a 0-100 research score.
    trend = np.tanh(((d["ema20"] / d["ema50"]) - 1) * 100)
    momentum = np.tanh((d["rsi"] - 50) / 12)
    macd_component = np.tanh((d["macd"] - d["macd_signal"]) / d["close"] * 5000)
    d["score"] = 50 + 20 * trend + 20 * momentum + 10 * macd_component
    d["score"] = d["score"].clip(0, 100)
    d["signal"] = np.select(
        [d["score"] >= 65, d["score"] <= 35],
        ["Bullish watch", "Bearish watch"],
        default="Neutral"
    )
    return d

def backtest(d: pd.DataFrame, hold_bars: int, fee_pct: float, slippage_pct: float):
    # Signal is observed at candle close; simulated entry is next candle close.
    # Exit is hold_bars candles after that. This avoids same-candle look-ahead.
    x = d.copy()
    x["entry"] = x["close"].shift(-1)
    x["exit"] = x["close"].shift(-(hold_bars + 1))
    x["future_return"] = x["exit"] / x["entry"] - 1
    x["position"] = np.where(x["score"] >= 65, 1, np.where(x["score"] <= 35, -1, 0))
    cost = 2 * (fee_pct + slippage_pct) / 100
    x["strategy_return"] = x["position"] * x["future_return"] - np.where(x["position"] != 0, cost, 0)
    trades = x[(x["position"] != 0) & x["strategy_return"].notna()].copy()
    if trades.empty:
        return trades, {}
    wins = (trades["strategy_return"] > 0).mean() * 100
    avg = trades["strategy_return"].mean() * 100
    total = ((1 + trades["strategy_return"]).prod() - 1) * 100
    equity = (1 + trades["strategy_return"]).cumprod()
    drawdown = (equity / equity.cummax() - 1).min() * 100
    stats = {
        "trades": len(trades),
        "win_rate": wins,
        "avg_trade": avg,
        "total_return": total,
        "max_drawdown": drawdown,
    }
    return trades, stats

with st.sidebar:
    st.header("Bot settings")
    selected = st.multiselect("Assets", list(SYMBOLS.keys()), default=list(SYMBOLS.keys()))
    interval_label = st.selectbox("Candle interval", list(INTERVALS.keys()), index=0)
    hold_bars = st.slider("Backtest holding period (candles)", 1, 24, 6)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, max_value=2.0, value=0.10, step=0.01)
    slippage_pct = st.number_input("Slippage per side (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    st.caption("Data refreshes every 5 minutes. Backtest settings are illustrative.")

if not selected:
    st.info("Choose at least one asset in the sidebar.")
    st.stop()

tabs = st.tabs(["Live research", "Market scanner", "Backtesting", "Paper trading", "How it works"])

asset_data = {}
errors = {}
for asset in selected:
    try:
        raw = fetch_klines(SYMBOLS[asset], INTERVALS[interval_label])
        asset_data[asset] = add_indicators(raw)
    except Exception as e:
        errors[asset] = str(e)

for asset, err in errors.items():
    st.error(f"{asset}: Could not load market data. {err}")

if not asset_data:
    st.error("No market data loaded. Check your internet connection and try again later.")
    st.stop()

with tabs[0]:
    for asset, d in asset_data.items():
        latest = d.iloc[-1]
        st.subheader(asset)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest close", f"${latest['close']:,.2f}")
        c2.metric("Research score", f"{latest['score']:.1f}/100")
        c3.metric("RSI (14)", f"{latest['rsi']:.1f}")
        c4.metric("Rule-based status", latest["signal"])
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=d["time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=d["time"], y=d["ema20"], name="EMA 20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=d["time"], y=d["ema50"], name="EMA 50", line=dict(width=1)))
        fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last candle timestamp: {latest['time'].strftime('%Y-%m-%d %H:%M UTC')}. The latest candle may still be forming.")

with tabs[1]:
    st.subheader("Market scanner")
    rows = []
    for asset, d in asset_data.items():
        r = d.iloc[-1]
        rows.append({
            "Asset": asset,
            "Price (USD)": r["close"],
            "Score / 100": round(r["score"], 1),
            "RSI": round(r["rsi"], 1),
            "24-candle volatility": round(r["volatility"] * 100, 2) if pd.notna(r["volatility"]) else np.nan,
            "Status": r["signal"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("The scanner ranks rule-based conditions; it does not establish a profitable trading edge.")

with tabs[2]:
    st.subheader("Historical strategy test")
    st.write("A simple long/short research rule: score ≥65 = long; score ≤35 = short; otherwise no position.")
    for asset, d in asset_data.items():
        trades, stats = backtest(d, hold_bars, fee_pct, slippage_pct)
        st.markdown(f"**{asset}**")
        if not stats:
            st.info("Not enough completed signals to calculate results.")
            continue
        a,b,c,dcol,e = st.columns(5)
        a.metric("Trades", stats["trades"])
        b.metric("Win rate", f"{stats['win_rate']:.1f}%")
        c.metric("Average trade", f"{stats['avg_trade']:.2f}%")
        dcol.metric("Compound return", f"{stats['total_return']:.2f}%")
        e.metric("Max drawdown", f"{stats['max_drawdown']:.2f}%")
        st.dataframe(
            trades[["time", "position", "score", "entry", "exit", "strategy_return"]]
            .tail(15).rename(columns={"position":"Direction (1 long, -1 short)", "strategy_return":"Net return"}),
            use_container_width=True, hide_index=True
        )
    st.warning("This is an educational, simplified backtest. It can overstate performance due to overlapping trades, limited history, regime changes and imperfect execution assumptions. Test on unseen data before trusting it.")

with tabs[3]:
    st.subheader("Paper trading simulator")
    st.write("This is a manual, hypothetical ledger stored only for this browser session. No real orders are sent.")
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []
    starting_balance = st.number_input("Virtual starting balance (USD)", min_value=10.0, value=100.0, step=10.0)
    asset_choice = st.selectbox("Asset for hypothetical trade", list(asset_data.keys()), key="paper_asset")
    direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
    stake = st.number_input("Hypothetical stake (USD)", min_value=1.0, value=10.0, step=1.0)
    entry_price = float(asset_data[asset_choice].iloc[-1]["close"])
    st.write(f"Reference price: **${entry_price:,.2f}**")
    if st.button("Record paper entry"):
        if stake > starting_balance:
            st.error("Stake cannot exceed the virtual starting balance.")
        else:
            st.session_state.paper_trades.append({
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "asset": asset_choice,
                "direction": direction,
                "stake": float(stake),
                "entry": entry_price,
            })
            st.success("Hypothetical entry recorded in this session.")
    if st.session_state.paper_trades:
        ledger = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        if st.button("Clear paper ledger"):
            st.session_state.paper_trades = []
            st.rerun()
    else:
        st.info("No hypothetical trades recorded yet. Entries reset when the app session ends.")

with tabs[4]:
    st.subheader("How this bot works")
    st.markdown("""
    - **Data:** Public Binance spot-market candle data for BTCUSDT and ETHUSDT. No account or API key is used.
    - **Indicators:** EMA 20/50, RSI 14, MACD and rolling volatility.
    - **Research score:** A transparent weighted heuristic combining trend, momentum and MACD. It is **not** a trained model or a calibrated probability.
    - **Scanner:** Summarises the latest score and indicators across selected assets.
    - **Backtest:** Applies the rule to historical candles, enters on the next candle close, and estimates fees and slippage.
    - **Paper ledger:** Records hypothetical entries only; it does not calculate live unrealised P&L or persist after a session ends.

    **Limitations:** Public data can be delayed, unavailable or rate-limited. Historical results do not predict future returns. The strategy may lose money. No financial advice or live trading is provided.
    """)
    st.markdown("[Binance public market-data documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data)")
    st.markdown("[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud)")
