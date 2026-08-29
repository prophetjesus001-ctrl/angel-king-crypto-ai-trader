import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# ============================================================
# 👑 ANGEL KING CRYPTO AI TRADER V6.0
# Multi-Timeframe: 1m / 15m / 1h
# ============================================================

st.set_page_config(page_title="Angel King V6.0", page_icon="👑", layout="wide")

BINANCE_BASE = "https://api.binance.us"

TRADE_CAPITAL = 50.0
LEVERAGE = 10
RISK_PERCENT = 1.0
TP_ATR_MULTIPLIER = 2.5
SL_ATR_MULTIPLIER = 1.4

@st.cache_data(ttl=20)
def get_klines(symbol="BTCUSDT", interval="1m", limit=300):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame([{
        "time": pd.to_datetime(x[0], unit="ms"),
        "open": float(x[1]),
        "high": float(x[2]),
        "low": float(x[3]),
        "close": float(x[4]),
        "volume": float(x[5])
    } for x in data])

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
    tr = pd.concat([d["high"]-d["low"], (d["high"]-prev).abs(), (d["low"]-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["vol_sma"] = d["volume"].rolling(20).mean()
    return d

def find_swing_points(df, left=3, right=3):
    highs = df["high"].values
    lows = df["low"].values
    swing_high = swing_low = None
    for i in range(left, len(df)-right):
        if highs[i] == max(highs[i-left:i+right+1]):
            swing_high = highs[i]
        if lows[i] == min(lows[i-left:i+right+1]):
            swing_low = lows[i]
    return swing_high, swing_low

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

    return {"signal": signal, "confidence": confidence, "score": score,
            "reasons": reasons, "rsi": rsi, "atr": row.atr, "close": price}

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
        "margin": (qty * entry)/leverage,
        "risk_amount": risk_amount, "rr": tp_dist / stop_dist
    }

# ============================================================
# MAIN
# ============================================================

st.title("👑 Angel King V6.0")
st.caption("Multi-Timeframe • 1m / 15m / 1h")

# Sidebar controls
symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
timeframe = st.sidebar.radio("Timeframe", ["1m", "15m", "1h"], index=0)
mode = st.sidebar.radio("Mode", ["Strict", "Active"], index=0)
capital = st.sidebar.number_input("Capital", value=TRADE_CAPITAL, min_value=10.0)
leverage = st.sidebar.slider("Leverage", 1, 20, LEVERAGE)
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, RISK_PERCENT, 0.25)

try:
    df = get_klines(symbol, interval=timeframe, limit=300)
    df = indicators(df)
    current = df.iloc[-1]
    previous = df.iloc[-2]

    signal_data = classify(current, previous, mode)
    plan = calculate_trade_plan(signal_data, capital, leverage, risk_pct)
    swing_high, swing_low = find_swing_points(df)

    # Signal Badge
    signal = signal_data["signal"]
    if signal == "LONG":
        bg, txt = "#00c853", "BUY"
    elif signal == "SHORT":
        bg, txt = "#ff1744", "SELL"
    else:
        bg, txt = "#616161", "NEUTRAL"

    st.markdown(f"""
    <div style="background-color:{bg}; padding:8px 16px; border-radius:6px; display:inline-block; margin-bottom:12px;">
        <span style="color:white; font-size:16px; font-weight:700;">{txt}</span>
    </div>
    &nbsp;&nbsp;
    <span style="font-size:18px; font-weight:600;">${signal_data['close']:,.2f}</span>
    <span style="color:gray; font-size:14px;"> &nbsp; Score: {signal_data['score']:+d} | TF: {timeframe}</span>
    """, unsafe_allow_html=True)

    # Two separate small boxes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:8px 10px; font-size:13px;">
            <b>Swing Levels</b><br>
            High: <b>{f'${swing_high:,.1f}' if swing_high else 'N/A'}</b><br>
            Low: &nbsp;<b>{f'${swing_low:,.1f}' if swing_low else 'N/A'}</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if plan:
            st.markdown(f"""
            <div style="background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:8px 10px; font-size:13px;">
                <b>Trade Plan</b><br>
                Entry: ${plan['entry']:,.2f}<br>
                SL: ${plan['stop_loss']:,.2f}<br>
                TP: ${plan['take_profit']:,.2f}<br>
                R:R 1:{plan['rr']:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:8px 10px; font-size:13px;">
                <b>Trade Plan</b><br>
                No active plan
            </div>
            """, unsafe_allow_html=True)

    with st.expander("More details & Reasons"):
        if plan:
            st.write(f"Quantity: {plan['quantity']:.4f} | Margin: ${plan['margin']:.2f} | Risk: ${plan['risk_amount']:.2f}")
        for r in signal_data["reasons"]:
            st.write("• " + r)

    # Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"]
    )])
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema21"], name="EMA21", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(width=2)))

    if swing_high:
        fig.add_hline(y=swing_high, line_dash="dot", line_color="red", annotation_text="High")
    if swing_low:
        fig.add_hline(y=swing_low, line_dash="dot", line_color="green", annotation_text="Low")

    fig.update_layout(height=460, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Timeframe: {timeframe} | Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

except Exception as e:
    st.error(str(e))

time.sleep(60)
st.rerun()
