import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# ============================================================
# 👑 ANGEL KING CRYPTO AI TRADER V5
# Based on your original code
# 4-Hour timeframe • Improved signals • Risk management
# Trading OFF
# ============================================================

st.set_page_config(
    page_title="Angel King Crypto AI Trader V5",
    page_icon="👑",
    layout="wide"
)

# ============================================================
# CONFIGURATION (same style as your original)
# ============================================================

BINANCE_BASE = "https://api.binance.us"
INTERVAL = "4h"

# ============================================================
# TRADE SETTINGS
# ============================================================

TRADE_CAPITAL = 50.00
LEVERAGE = 10
RISK_PERCENT = 1.0          # Risk 1% of capital per trade
TP_ATR_MULTIPLIER = 2.5
SL_ATR_MULTIPLIER = 1.4

# ============================================================
# BINANCE MARKET DATA (same style as your original)
# ============================================================

@st.cache_data(ttl=55)
def get_klines(symbol="BTCUSDT", limit=200):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError("Binance returned no market data.")

    df = pd.DataFrame([
        {
            "time": pd.to_datetime(x[0], unit="ms"),
            "open": float(x[1]),
            "high": float(x[2]),
            "low": float(x[3]),
            "close": float(x[4]),
            "volume": float(x[5])
        }
        for x in data
    ])
    return df


def get_current_price(symbol):
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["price"])


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):
    d = df.copy()

    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    # ATR 14
    previous_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - previous_close).abs(),
        (d["low"] - previous_close).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Volume
    d["vol_sma"] = d["volume"].rolling(20).mean()

    return d


# ============================================================
# IMPROVED SIGNAL ENGINE (based on your original style)
# ============================================================

def classify(row, previous=None):
    bull = 0
    bear = 0
    reasons = []

    price = row.close
    ema9 = row.ema9
    ema21 = row.ema21
    ema50 = row.ema50
    rsi = row.rsi

    # 1. EMA Trend (strongest)
    if price > ema9 > ema21 > ema50:
        bull += 3
        reasons.append("Strong bullish EMA stack (9>21>50)")
    elif price < ema9 < ema21 < ema50:
        bear += 3
        reasons.append("Strong bearish EMA stack (9<21<50)")
    elif price > ema21:
        bull += 1
        reasons.append("Price above EMA21")
    elif price < ema21:
        bear += 1
        reasons.append("Price below EMA21")

    # 2. EMA Direction
    if previous is not None:
        if row.ema9 > previous.ema9 and row.ema21 > previous.ema21:
            bull += 1
            reasons.append("EMAs rising")
        elif row.ema9 < previous.ema9 and row.ema21 < previous.ema21:
            bear += 1
            reasons.append("EMAs falling")

    # 3. RSI
    if 45 <= rsi <= 65:
        bull += 1
        reasons.append(f"RSI healthy zone ({rsi:.1f})")
    elif 35 <= rsi <= 55:
        bear += 1
        reasons.append(f"RSI healthy zone ({rsi:.1f})")
    elif rsi >= 70:
        bear += 1
        reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi <= 30:
        bull += 1
        reasons.append(f"RSI oversold ({rsi:.1f})")

    # 4. MACD
    if row.macd > row.macd_signal and row.macd_hist > 0:
        bull += 1
        reasons.append("MACD bullish")
    elif row.macd < row.macd_signal and row.macd_hist < 0:
        bear += 1
        reasons.append("MACD bearish")

    # 5. Volume
    if row.volume > row.vol_sma * 1.2:
        if bull > bear:
            bull += 1
            reasons.append("Volume confirmation")
        elif bear > bull:
            bear += 1
            reasons.append("Volume confirmation")

    # Final decision
    score = bull - bear

    if score >= 4:
        signal = "LONG"
        confidence = "High" if score >= 6 else "Medium"
    elif score <= -4:
        signal = "SHORT"
        confidence = "High" if score <= -6 else "Medium"
    else:
        signal = "NEUTRAL"
        confidence = "Low"

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "bull": bull,
        "bear": bear,
        "reasons": reasons,
        "rsi": rsi,
        "atr": row.atr,
        "close": price
    }


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_trade_plan(signal_data, capital, leverage, risk_pct):
    if signal_data["signal"] == "NEUTRAL":
        return None

    atr = signal_data["atr"]
    price = signal_data["close"]

    stop_distance = atr * SL_ATR_MULTIPLIER
    tp_distance = atr * TP_ATR_MULTIPLIER

    if signal_data["signal"] == "LONG":
        entry = price
        stop_loss = entry - stop_distance
        take_profit = entry + tp_distance
    else:
        entry = price
        stop_loss = entry + stop_distance
        take_profit = entry - tp_distance

    risk_amount = capital * (risk_pct / 100)
    quantity = risk_amount / stop_distance
    notional = quantity * entry
    margin = notional / leverage

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "quantity": quantity,
        "notional": notional,
        "margin": margin,
        "risk_amount": risk_amount,
        "rr": tp_distance / stop_distance
    }


# ============================================================
# MAIN APP
# ============================================================

st.title("👑 Angel King Crypto AI Trader V5")
st.caption("4-Hour Timeframe • Improved Signals • Risk Management • Trading OFF")

symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
capital = st.sidebar.number_input("Capital (USDT)", value=TRADE_CAPITAL, min_value=10.0)
leverage = st.sidebar.slider("Leverage", 1, 20, LEVERAGE)
risk_pct = st.sidebar.slider("Risk % per trade", 0.5, 3.0, RISK_PERCENT, 0.25)

try:
    df = get_klines(symbol, limit=200)
    df = indicators(df)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    signal_data = classify(current, previous)
    plan = calculate_trade_plan(signal_data, capital, leverage, risk_pct)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${signal_data['close']:,.2f}")
    col2.metric("Signal", signal_data["signal"])
    col3.metric("Confidence", signal_data["confidence"])
    col4.metric("Score", f"{signal_data['score']:+d}")

    st.markdown("---")

    # Reasons
    st.subheader(f"Signal: {signal_data['signal']}")
    for reason in signal_data["reasons"]:
        st.write(f"• {reason}")

    # Trade Plan
    if plan:
        st.subheader("Trade Plan")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"${plan['entry']:,.2f}")
        c2.metric("Stop Loss", f"${plan['stop_loss']:,.2f}")
        c3.metric("Take Profit", f"${plan['take_profit']:,.2f}")
        c4.metric("Risk : Reward", f"1 : {plan['rr']:.2f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Quantity", f"{plan['quantity']:.4f}")
        c6.metric("Notional", f"${plan['notional']:,.0f}")
        c7.metric("Margin Needed", f"${plan['margin']:,.2f}")

        st.success(f"Risking ${plan['risk_amount']:.2f} ({risk_pct}% of capital)")
    else:
        st.info("No clear high-quality setup right now.")

    # Simple Chart
    st.subheader("4H Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema9"], name="EMA9", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema21"], name="EMA21", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(width=2)))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Last update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | Data: Binance.US")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("If this keeps happening, the server IP is restricted. Run the app locally for best results.")
