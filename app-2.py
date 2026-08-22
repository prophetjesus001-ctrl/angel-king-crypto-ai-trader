import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# 👑 ANGEL KING CRYPTO AI TRADER V3
# Binance market data • 1-minute scalping • Trading OFF
# EXISTING SIGNAL ENGINE LOCKED
# ============================================================

st.set_page_config(
    page_title="Angel King Crypto AI Trader V3",
    page_icon="👑",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_BASE = "https://api.binance.us"
INTERVAL = "1m"

# ============================================================
# BINANCE MARKET DATA
# ============================================================

@st.cache_data(ttl=4)
def get_klines(symbol="BTCUSDT", limit=300):

    url = f"{BINANCE_BASE}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit
    }

    r = requests.get(
        url,
        params=params,
        timeout=10
    )

    r.raise_for_status()

    data = r.json()

    if not data:
        raise ValueError("Binance returned no market data.")

    df = pd.DataFrame(
        [
            {
                "time": pd.to_datetime(x[0], unit="ms"),
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5])
            }
            for x in data
        ]
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    d = df.copy()

    # EMA 9
    d["ema9"] = d["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    # EMA 21
    d["ema21"] = d["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # RSI 14
    delta = d["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    d["rsi"] = 100 - (100 / (1 + rs))

    # ATR 14
    previous_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - previous_close).abs(),
            (d["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    # Candle direction
    d["green"] = d["close"] > d["open"]
    d["red"] = d["close"] < d["open"]

    # Recent candle counts
    d["green_count"] = d["green"].rolling(10).sum()
    d["red_count"] = d["red"].rolling(10).sum()

    return d


# ============================================================
# ORIGINAL SIGNAL ENGINE
# DO NOT CHANGE
# ============================================================

def classify(row, previous=None):

    bull = 0
    bear = 0
    reasons = []

    price = row.close
    ema9 = row.ema9
    ema21 = row.ema21
    rsi = row.rsi

    # --------------------------------------------------------
    # EMA TREND
    # --------------------------------------------------------

    if price > ema9 > ema21:
        bull += 2
        reasons.append("price above EMA9/EMA21")

    elif price < ema9 < ema21:
        bear += 2
        reasons.append("price below EMA9/EMA21")

    # --------------------------------------------------------
    # EMA DIRECTION
    # --------------------------------------------------------

    if previous is not None:

        if row.ema9 > previous.ema9:
            bull += 1
            reasons.append("EMA9 rising")

        elif row.ema9 < previous.ema9:
            bear += 1
            reasons.append("EMA9 falling")

        if row.ema21 > previous.ema21:
            bull += 1

        elif row.ema21 < previous.ema21:
            bear += 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if 50 <= rsi <= 70:
        bull += 1
        reasons.append("RSI bullish zone")

    elif 30 <= rsi < 50:
        bear += 1
        reasons.append("RSI bearish zone")

    # --------------------------------------------------------
    # CANDLE MOMENTUM
    # --------------------------------------------------------

    if row.green:
        bull += 1

    elif row.red:
        bear += 1

    # --------------------------------------------------------
    # RECENT CANDLE PRESSURE
    # --------------------------------------------------------

    green_count = row.green_count
    red_count = row.red_count

    if pd.notna(green_count) and pd.notna(red_count):

        if green_count > red_count + 2:
            bull += 1
            reasons.append("green candle pressure")

        elif red_count > green_count + 2:
            bear += 1
            reasons.append("red candle pressure")

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    total = bull + bear

    if bull >= 5 and bull > bear:

        signal = "BUY"

        confidence = min(
            95,
            50 + (bull - bear) * 8
        )

        reason = ", ".join(
            reasons[:4]
        ) if reasons else "Bullish conditions"

    elif bear >= 5 and bear > bull:

        signal = "SELL"

        confidence = min(
            95,
            50 + (bear - bull) * 8
        )

        reason = ", ".join(
            reasons[:4]
        ) if reasons else "Bearish conditions"

    else:

        signal = "WAIT"
        confidence = 50
        reason = "Mixed conditions"

    return signal, confidence, reason


# ============================================================
# 🆕 MARKET POWER INDICATOR
# SEPARATE FROM ORIGINAL SIGNAL ENGINE
# ============================================================

def market_power_indicator(d, signal):

    # --------------------------------------------------------
    # IMPORTANT:
    # If the original engine says WAIT,
    # Market Power stays completely blank.
    # --------------------------------------------------------

    if signal == "WAIT":
        return "", "blank"

    # We examine recent completed candles.
    recent = d.iloc[-8:].copy()

    if len(recent) < 8:
        return "", "blank"

    closes = recent["close"].values
    ema9 = recent["ema9"].values
    ema21 = recent["ema21"].values

    # Count directional candle movement
    up_candles = sum(
        recent["close"] > recent["open"]
    )

    down_candles = sum(
        recent["close"] < recent["open"]
    )

    # Overall price movement
    price_change = closes[-1] - closes[0]

    # EMA movement
    ema9_change = ema9[-1] - ema9[0]
    ema21_change = ema21[-1] - ema21[0]

    # --------------------------------------------------------
    # DOWNWARD POWER
    # --------------------------------------------------------

    down_power = (
        down_candles >= 5
        and price_change < 0
        and ema9_change < 0
        and ema21_change <= 0
        and ema9[-1] < ema21[-1]
    )

    # --------------------------------------------------------
    # UPWARD POWER
    # --------------------------------------------------------

    up_power = (
        up_candles >= 5
        and price_change > 0
        and ema9_change > 0
        and ema21_change >= 0
        and ema9[-1] > ema21[-1]
    )

    # --------------------------------------------------------
    # REVERSAL DETECTION
    # --------------------------------------------------------

    # Downtrend weakening and upward movement beginning
    upward_reversal = (
        down_candles >= 3
        and up_candles >= 3
        and price_change > 0
        and ema9_change > 0
    )

    # Uptrend weakening and downward movement beginning
    downward_reversal = (
        up_candles >= 3
        and down_candles >= 3
        and price_change < 0
        and ema9_change < 0
    )

    # --------------------------------------------------------
    # FINAL MARKET POWER STATE
    # --------------------------------------------------------

    if signal == "SELL":

        if upward_reversal:

            return (
                "🟡 MARKET REVERSING — UPWARD",
                "reversing_up"
            )

        if down_power:

            return (
                "🔴 MARKET POWER DOWN — CONSISTENT",
                "down"
            )

        return (
            "🔴 MARKET POWER DOWN",
            "down"
        )

    if signal == "BUY":

        if downward_reversal:

            return (
                "🟡 MARKET REVERSING — DOWNWARD",
                "reversing_down"
            )

        if up_power:

            return (
                "🟢 MARKET POWER UP — CONSISTENT",
                "up"
            )

        return (
            "🟢 MARKET POWER UP",
            "up"
        )

    return "", "blank"


# ============================================================
# MARKET POWER DISPLAY
# ============================================================

def show_market_power(text, state):

    if state == "blank":
        return

    if state == "down":

        st.error(
            f"### {text}"
        )

    elif state == "up":

        st.success(
            f"### {text}"
        )

    elif state == "reversing_up":

        st.warning(
            f"### {text}"
        )

    elif state == "reversing_down":

        st.warning(
            f"### {text}"
        )


# ============================================================
# SIGNAL DISPLAY
# ============================================================

def show_signal(signal, confidence, reason):

    if signal == "BUY":

        st.success(
            f"🟢 BUY  •  Confidence: {confidence:.0f}%"
        )

        st.markdown(
            f"""
            **Signal:** 🟢 BUY

            **Reason:** {reason}
            """
        )

    elif signal == "SELL":

        st.error(
            f"🔴 SELL  •  Confidence: {confidence:.0f}%"
        )

        st.markdown(
            f"""
            **Signal:** 🔴 SELL

            **Reason:** {reason}
            """
        )

    else:

        st.warning(
            f"🟡 WAIT  •  Confidence: {confidence:.0f}%"
        )

        st.markdown(
            f"""
            **Signal:** 🟡 WAIT

            **Reason:** {reason}
            """
        )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    # 👑 Angel King Crypto AI Trader V3

    **Binance market data • 1-minute scalping • live signal dashboard • trading OFF**
    """
)


# ============================================================
# LIVE MARKET SECTION
# ============================================================

st.markdown("### 📊 Binance Market")

assets = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "ADA/USDT": "ADAUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "LINK/USDT": "LINKUSDT",
    "LTC/USDT": "LTCUSDT"
}

selected_asset = st.selectbox(
    "Market",
    list(assets.keys()),
    index=0
)

SYMBOL = assets[selected_asset]


# ============================================================
# LIVE DATA + SIGNAL PROCESSING
# ============================================================

try:

    df = get_klines(
        symbol=SYMBOL,
        limit=300
    )

    d = indicators(df)

    latest = d.iloc[-1]
    previous = d.iloc[-2]

    # ========================================================
    # ORIGINAL V3 ENGINE
    # ========================================================

    signal, confidence, reason = classify(
        latest,
        previous
    )

    # ========================================================
    # 🆕 MARKET POWER INDICATOR
    # ========================================================

    market_power_text, market_power_state = (
        market_power_indicator(
            d,
            signal
        )
    )

    # ========================================================
    # FRONT / TOP MARKET POWER BOX
    # ========================================================

    st.markdown("---")

    show_market_power(
        market_power_text,
        market_power_state
    )

    # ========================================================
    # MAIN SIGNAL
    # ========================================================

    show_signal(
        signal,
        confidence,
        reason
    )

    st.markdown("---")

    # ========================================================
    # PRICE INFORMATION
    # ========================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            selected_asset,
            f"${latest.close:,.2f}"
        )

    with c2:

        st.metric(
            "Signal",
            signal
        )

    with c3:

        st.metric(
            "Confidence",
            f"{confidence:.0f}%"
        )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        f"{selected_asset} • 1-minute chart"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=d["time"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["time"],
            y=d["ema9"],
            mode="lines",
            name="EMA 9"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["time"],
            y=d["ema21"],
            mode="lines",
            name="EMA 21"
        )
    )

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # SIGNAL DETAILS
    # ========================================================

    st.markdown(
        f"""
        **Reason:** {reason}

        **RSI:** {latest.rsi:.2f}

        **EMA 9:** {latest.ema9:.2f}

        **EMA 21:** {latest.ema21:.2f}
        """
    )

    # ========================================================
    # TRADING STATUS
    # ========================================================

    st.info(
        "Execution is OFF. This app does not place Binance trades."
    )

    st.caption(
        "🔄 Live Binance market data refreshes automatically every 5 seconds."
    )

    st.caption(
        "⚠️ Experimental software. Backtests are not guarantees of future "
        "performance. Do not use real funds based solely on these signals."
    )


except requests.exceptions.HTTPError as e:

    st.error(
        f"Unable to retrieve Binance market data: {e}"
    )

except Exception as e:

    st.error(
        f"Unable to retrieve Binance market data: {e}"
    )


# ============================================================
# AUTOMATIC 5-SECOND REFRESH
# ============================================================

time.sleep(5)

st.rerun()
