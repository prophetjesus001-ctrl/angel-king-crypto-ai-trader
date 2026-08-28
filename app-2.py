import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# ============================================================
# 👑 ANGEL KING CRYPTO AI TRADER V5 – HIGH CONVICTION
# Binance Futures • 4H Execution + Daily Bias
# Trading OFF – Professional Signals + Risk Management
# ============================================================

st.set_page_config(
    page_title="Angel King V5 – High Conviction 4H",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIG
# ============================================================

FUTURES_BASE = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👑 Angel King V5")
st.sidebar.markdown("**High Conviction • 4H + Daily Bias**")

symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)
capital = st.sidebar.number_input("Capital (USDT)", min_value=20.0, value=100.0, step=10.0)
leverage = st.sidebar.slider("Leverage", 1, 25, 8)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.success("Trading is OFF – Signals only")
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=True)

# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=50)
def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    url = f"{FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame([{
        "time": pd.to_datetime(x[0], unit="ms", utc=True),
        "open": float(x[1]),
        "high": float(x[2]),
        "low": float(x[3]),
        "close": float(x[4]),
        "volume": float(x[5])
    } for x in data])
    return df

@st.cache_data(ttl=30)
def get_ticker(symbol: str) -> dict:
    r = requests.get(f"{FUTURES_BASE}/fapi/v1/ticker/24hr", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def get_funding(symbol: str) -> float:
    r = requests.get(f"{FUTURES_BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json().get("lastFundingRate", 0)) * 100

# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ema9"]  = d["close"].ewm(span=9,  adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    prev = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev).abs(),
        (d["low"] - prev).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["atr_sma"] = d["atr"].rolling(50).mean()

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Volume
    d["vol_sma"] = d["volume"].rolling(20).mean()

    # Simple swing structure (last 5 candles)
    d["swing_high"] = d["high"].rolling(5).max()
    d["swing_low"]  = d["low"].rolling(5).min()

    return d

# ============================================================
# HIGH CONVICTION SIGNAL ENGINE
# ============================================================

def generate_signal(df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> dict:
    if len(df_4h) < 80 or len(df_daily) < 50:
        return {"signal": "NEUTRAL", "confidence": "None", "score": 0, "reasons": ["Insufficient data"]}

    row = df_4h.iloc[-1]
    prev = df_4h.iloc[-2]
    daily = df_daily.iloc[-1]

    bull = 0
    bear = 0
    reasons = []

    # ---------- 1. DAILY BIAS (mandatory filter) ----------
    daily_bullish = daily.close > daily.ema50 and daily.ema21 > daily.ema50
    daily_bearish = daily.close < daily.ema50 and daily.ema21 < daily.ema50

    if daily_bullish:
        reasons.append("Daily bias: BULLISH")
    elif daily_bearish:
        reasons.append("Daily bias: BEARISH")
    else:
        reasons.append("Daily bias: NEUTRAL → no trade")
        return {"signal": "NEUTRAL", "confidence": "None", "score": 0, "reasons": reasons,
                "close": row.close, "atr": row.atr, "rsi": row.rsi}

    # ---------- 2. 4H EMA STRUCTURE (core) ----------
    if row.close > row.ema9 > row.ema21 > row.ema50:
        bull += 3
        reasons.append("4H strong bullish EMA stack")
    elif row.close < row.ema9 < row.ema21 < row.ema50:
        bear += 3
        reasons.append("4H strong bearish EMA stack")
    else:
        reasons.append("4H EMA structure not clean")

    # EMA slope
    if row.ema9 > prev.ema9 and row.ema21 > prev.ema21:
        bull += 1
        reasons.append("EMAs rising")
    elif row.ema9 < prev.ema9 and row.ema21 < prev.ema21:
        bear += 1
        reasons.append("EMAs falling")

    # ---------- 3. RSI ----------
    if 45 <= row.rsi <= 68:
        bull += 1
        reasons.append(f"RSI healthy ({row.rsi:.1f})")
    elif 32 <= row.rsi <= 55:
        bear += 1
        reasons.append(f"RSI healthy ({row.rsi:.1f})")
    elif row.rsi > 72:
        bear += 1
        reasons.append(f"RSI overbought ({row.rsi:.1f})")
    elif row.rsi < 28:
        bull += 1
        reasons.append(f"RSI oversold ({row.rsi:.1f})")

    # ---------- 4. MACD ----------
    if row.macd > row.macd_signal and row.macd_hist > 0 and row.macd_hist > prev.macd_hist:
        bull += 1
        reasons.append("MACD bullish & expanding")
    elif row.macd < row.macd_signal and row.macd_hist < 0 and row.macd_hist < prev.macd_hist:
        bear += 1
        reasons.append("MACD bearish & expanding")

    # ---------- 5. VOLUME ----------
    if row.volume > row.vol_sma * 1.15:
        if bull > bear:
            bull += 1
            reasons.append("Volume confirmation")
        elif bear > bull:
            bear += 1
            reasons.append("Volume confirmation")

    # ---------- 6. VOLATILITY FILTER ----------
    if row.atr < row.atr_sma * 0.75:
        reasons.append("Volatility too low → skip")
        return {"signal": "NEUTRAL", "confidence": "None", "score": 0, "reasons": reasons,
                "close": row.close, "atr": row.atr, "rsi": row.rsi}

    # ---------- 7. MARKET STRUCTURE (simple) ----------
    if row.close > row.swing_high.shift(1).iloc[-1] if not pd.isna(row.swing_high) else False:
        bull += 1
        reasons.append("Broke recent swing high")
    if row.close < row.swing_low.shift(1).iloc[-1] if not pd.isna(row.swing_low) else False:
        bear += 1
        reasons.append("Broke recent swing low")

    # ---------- FINAL DECISION (strict) ----------
    score = bull - bear

    # Only allow trade in direction of Daily bias
    if daily_bullish and score >= 4:
        signal = "LONG"
        confidence = "High" if score >= 6 else "Medium"
    elif daily_bearish and score <= -4:
        signal = "SHORT"
        confidence = "High" if score <= -6 else "Medium"
    else:
        signal = "NEUTRAL"
        confidence = "Low"
        reasons.append("Score not strong enough or against Daily bias")

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "bull": bull,
        "bear": bear,
        "reasons": reasons,
        "close": row.close,
        "atr": row.atr,
        "rsi": row.rsi,
        "ema9": row.ema9,
        "ema21": row.ema21,
        "ema50": row.ema50
    }

# ============================================================
# RISK CALCULATOR
# ============================================================

def calc_risk(sig: dict, capital: float, leverage: int, risk_pct: float):
    if sig["signal"] == "NEUTRAL":
        return None

    atr = sig["atr"]
    price = sig["close"]
    stop_dist = atr * 1.4          # slightly tighter for higher quality
    tp_dist   = atr * 2.8          # \~2R target

    if sig["signal"] == "LONG":
        entry = price
        sl = entry - stop_dist
        tp = entry + tp_dist
    else:
        entry = price
        sl = entry + stop_dist
        tp = entry - tp_dist

    risk_amount = capital * (risk_pct / 100)
    qty = risk_amount / stop_dist
    notional = qty * entry
    margin = notional / leverage

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": tp_dist / stop_dist,
        "qty": qty,
        "notional": notional,
        "margin": margin,
        "risk_amount": risk_amount
    }

# ============================================================
# CHART
# ============================================================

def make_chart(df: pd.DataFrame):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.58, 0.21, 0.21],
                        subplot_titles=("Price + EMAs", "RSI", "MACD"))

    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["ema9"],  name="EMA9",  line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema21"], name="EMA21", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(width=2.0)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["macd"], name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], name="Signal", line=dict(color="orange")), row=3, col=1)
    colors = ["green" if v >= 0 else "red" for v in df["macd_hist"]]
    fig.add_trace(go.Bar(x=df["time"], y=df["macd_hist"], name="Hist", marker_color=colors), row=3, col=1)

    fig.update_layout(height=820, xaxis_rangeslider_visible=False, template="plotly_dark",
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    return fig

# ============================================================
# MAIN
# ============================================================

st.title("👑 Angel King Crypto AI Trader V5")
st.caption("High-Conviction 4H System • Daily Bias Filter • Professional Risk Management")

try:
    with st.spinner(f"Loading {symbol} data..."):
        df_4h = add_indicators(get_klines(symbol, "4h", 200))
        df_d  = add_indicators(get_klines(symbol, "1d", 120))
        ticker = get_ticker(symbol)
        funding = get_funding(symbol)
        sig = generate_signal(df_4h, df_d)
        risk = calc_risk(sig, capital, leverage, risk_pct)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${sig['close']:,.2f}")
    c2.metric("24h Change", f"{float(ticker['priceChangePercent']):.2f}%")
    c3.metric("Funding", f"{funding:.4f}%")
    c4.metric("Signal", sig["signal"], delta=sig["confidence"])
    c5.metric("Score", f"{sig['score']:+d}")

    st.markdown("---")

    # Signal box
    color = {"LONG": "green", "SHORT": "red", "NEUTRAL": "gray"}[sig["signal"]]
    st.subheader(f"Signal: :{color}[{sig['signal']}]   |   Confidence: **{sig['confidence']}**")

    with st.expander("Full Reasoning", expanded=True):
        for r in sig["reasons"]:
            st.write("• " + r)
        st.write(f"Bull points: {sig.get('bull', 0)}   |   Bear points: {sig.get('bear', 0)}")

    # Risk plan
    if risk:
        st.subheader("Trade Plan & Risk")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Entry", f"${risk['entry']:,.2f}")
        r2.metric("Stop Loss", f"${risk['sl']:,.2f}")
        r3.metric("Take Profit", f"${risk['tp']:,.2f}")
        r4.metric("Risk : Reward", f"1 : {risk['rr']:.2f}")

        r5, r6, r7 = st.columns(3)
        r5.metric("Quantity", f"{risk['qty']:.4f}")
        r6.metric("Notional", f"${risk['notional']:,.0f}")
        r7.metric("Margin Required", f"${risk['margin']:,.2f}")

        st.success(f"Risking **${risk['risk_amount']:.2f}** ({risk_pct}% of capital)")
    else:
        st.warning("No high-conviction setup right now. Wait for better conditions.")

    # Chart
    st.subheader("4H Chart")
    st.plotly_chart(make_chart(df_4h), use_container_width=True)

    with st.expander("Notes"):
        st.write("• Only trades that align with Daily bias are allowed.")
        st.write("• Minimum score of ±4 required.")
        st.write("• Volatility filter blocks low-ATR environments.")
        st.write("• Always wait for 4H candle close before acting.")
        st.write("• This is not financial advice. Manage risk strictly.")

    st.caption(f"Last update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | Binance Futures")

except Exception as e:
    st.error(f"Error: {e}")

if auto_refresh:
    time.sleep(60)
    st.rerun()
