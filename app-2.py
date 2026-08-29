import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# ============================================================
# 👑 ANGEL KING CRYPTO AI TRADER V5.5
# Compact Signal Box (Green = BUY / Red = SELL)
# ============================================================

st.set_page_config(
    page_title="Angel King Crypto AI Trader V5.5",
    page_icon="👑",
    layout="wide"
)

BINANCE_BASE = "https://api.binance.us"
INTERVAL = "4h"

TRADE_CAPITAL = 50.00
LEVERAGE = 10
RISK_PERCENT = 1.0
TP_ATR_MULTIPLIER = 2.5
SL_ATR_MULTIPLIER = 1.4

@st.cache_data(ttl=50)
def get_klines(symbol="BTCUSDT", limit=200):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("No data")
    df = pd.DataFrame([{
        "time": pd.to_datetime(x[0], unit="ms"),
        "open": float(x[1]),
        "high": float(x[2]),
        "low": float(x[3]),
        "close": float(x[4]),
        "volume": float(x[5])
    } for x in data])
    return df

def indicators(df):
    d = df.copy()
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    prev = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev).abs(),
        (d["low"] - prev).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["vol_sma"] = d["volume"].rolling(20).mean()
    return d

def analyze_trend_consistency(df, lookback=8):
    if len(df) < lookback + 5:
        return "Neutral", "gray"
    recent = df.iloc[-lookback:]
    closes = recent["close"].values
    last_close = df["close"].iloc[-1]
    ema21 = df["ema21"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]

    higher = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
    lower = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])

    up = 0
    if last_close > ema21 > ema50: up += 2
    if higher >= lookback * 0.6: up += 2
    if last_close > df["close"].iloc[-lookback]: up += 1

    down = 0
    if last_close < ema21 < ema50: down += 2
    if lower >= lookback * 0.6: down += 2
    if last_close < df["close"].iloc[-lookback]: down += 1

    if up >= 4 and up > down:
        return "Uptrend", "green"
    elif down >= 4 and down > up:
        return "Downtrend", "red"
    return "Neutral", "gray"

def classify(row, previous=None, mode="Strict"):
    bull = bear = 0
    reasons = []
    price = row.close

    if price > row.ema9 > row.ema21 > row.ema50:
        bull += 3
        reasons.append("Strong bullish EMA stack")
    elif price < row.ema9 < row.ema21 < row.ema50:
        bear += 3
        reasons.append("Strong bearish EMA stack")
    elif price > row.ema21:
        bull += 1
        reasons.append("Price above EMA21")
    elif price < row.ema21:
        bear += 1
        reasons.append("Price below EMA21")

    if previous is not None:
        if row.ema9 > previous.ema9 and row.ema21 > previous.ema21:
            bull += 1
            reasons.append("EMAs rising")
        elif row.ema9 < previous.ema9 and row.ema21 < previous.ema21:
            bear += 1
            reasons.append("EMAs falling")

    rsi = row.rsi
    if 45 <= rsi <= 65: bull += 1; reasons.append(f"RSI healthy ({rsi:.1f})")
    elif 35 <= rsi <= 55: bear += 1; reasons.append(f"RSI healthy ({rsi:.1f})")
    elif rsi >= 70: bear += 1; reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi <= 30: bull += 1; reasons.append(f"RSI oversold ({rsi:.1f})")

    if row.macd > row.macd_signal and row.macd_hist > 0:
        bull += 1; reasons.append("MACD bullish")
    elif row.macd < row.macd_signal and row.macd_hist < 0:
        bear += 1; reasons.append("MACD bearish")

    if row.volume > row.vol_sma * 1.2:
        if bull > bear: bull += 1; reasons.append("Volume confirmation")
        elif bear > bull: bear += 1; reasons.append("Volume confirmation")

    score = bull - bear
    long_th, short_th = (4, -4) if mode == "Strict" else (2, -2)

    if score >= long_th:
        signal = "LONG"
        confidence = "High" if score >= long_th + 2 else "Medium"
    elif score <= short_th:
        signal = "SHORT"
        confidence = "High" if score <= short_th - 2 else "Medium"
    else:
        signal = "NEUTRAL"
        confidence = "Low"

    return {
        "signal": signal, "confidence": confidence, "score": score,
        "reasons": reasons, "rsi": rsi, "atr": row.atr, "close": price
    }

def calculate_trade_plan(signal_data, capital, leverage, risk_pct):
    if signal_data["signal"] == "NEUTRAL":
        return None
    atr = signal_data["atr"]
    price = signal_data["close"]
    stop_dist = atr * SL_ATR_MULTIPLIER
    tp_dist = atr * TP_ATR_MULTIPLIER

    if signal_data["signal"] == "LONG":
        entry, sl, tp = price, price - stop_dist, price + tp_dist
    else:
        entry, sl, tp = price, price + stop_dist, price - tp_dist

    risk_amount = capital * (risk_pct / 100)
    qty = risk_amount / stop_dist
    return {
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "quantity": qty, "notional": qty * entry,
        "margin": (qty * entry) / leverage,
        "risk_amount": risk_amount, "rr": tp_dist / stop_dist
    }

# ============================================================
# MAIN
# ============================================================

st.title("👑 Angel King Crypto AI Trader V5.5")
st.caption("Compact Signal Box • 4H • Auto-refresh")

symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
mode = st.sidebar.radio("Signal Mode", ["Strict", "Active"], index=0)
capital = st.sidebar.number_input("Capital (USDT)", value=TRADE_CAPITAL, min_value=10.0)
leverage = st.sidebar.slider("Leverage", 1, 20, LEVERAGE)
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, RISK_PERCENT, 0.25)

try:
    df = get_klines(symbol)
    df = indicators(df)
    current = df.iloc[-1]
    previous = df.iloc[-2]

    signal_data = classify(current, previous, mode)
    plan = calculate_trade_plan(signal_data, capital, leverage, risk_pct)
    trend_text, trend_color = analyze_trend_consistency(df)

    # ========== SMALL COMPACT SIGNAL BOX ==========
    signal = signal_data["signal"]

    if signal == "LONG":
        box_color = "#00c853"          # Bright green
        box_text = "BUY"
        text_color = "white"
    elif signal == "SHORT":
        box_color = "#ff1744"          # Bright red
        box_text = "SELL"
        text_color = "white"
    else:
        box_color = "#616161"          # Gray
        box_text = "NEUTRAL"
        text_color = "white"

    st.markdown(f"""
    <div style="
        display: inline-block;
        background-color: {box_color};
        padding: 10px 28px;
        border-radius: 8px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        <span style="color: {text_color}; font-size: 22px; font-weight: 700; letter-spacing: 1px;">
            {box_text}
        </span>
    </div>
    """, unsafe_allow_html=True)
    # ==============================================

    # Small trend info
    st.caption(f"Market Structure: {trend_text}")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${signal_data['close']:,.2f}")
    c2.metric("Signal", signal_data["signal"])
    c3.metric("Confidence", signal_data["confidence"])
    c4.metric("Score", f"{signal_data['score']:+d}")

    st.markdown(f"**Mode:** {mode}")
    st.markdown("---")

    st.subheader("Reasons")
    for r in signal_data["reasons"]:
        st.write(f"• {r}")

    if plan:
        st.subheader("Trade Plan")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Entry", f"${plan['entry']:,.2f}")
        r2.metric("Stop Loss", f"${plan['stop_loss']:,.2f}")
        r3.metric("Take Profit", f"${plan['take_profit']:,.2f}")
        r4.metric("R:R", f"1:{plan['rr']:.2f}")

        r5, r6, r7 = st.columns(3)
        r5.metric("Quantity", f"{plan['quantity']:.4f}")
        r6.metric("Notional", f"${plan['notional']:,.0f}")
        r7.metric("Margin", f"${plan['margin']:,.2f}")
        st.success(f"Risking ${plan['risk_amount']:.2f}")
    else:
        st.info("No clear setup right now.")

    # Chart
    st.subheader("4H Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"]
    )])
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema9"], name="EMA9", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema21"], name="EMA21", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(width=2)))
    fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Last update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

except Exception as e:
    st.error(f"Error: {e}")

time.sleep(60)
st.rerun()
