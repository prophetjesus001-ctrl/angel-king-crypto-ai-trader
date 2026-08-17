import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

INTERVAL = "1m"

MARKETS = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "ADA/USDT": "ADAUSDT",
    "AVAX/USDT": "AVAXUSDT",
    "LINK/USDT": "LINKUSDT",
}

st.set_page_config(
    page_title="Angel King Crypto AI Trader V3",
    page_icon="👑",
    layout="wide",
)

def get_klines(symbol, limit=300):
    r = requests.get(
        "https://api.binance.us/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit
        },
        timeout=15,
    )

    r.raise_for_status()
    x = r.json()

    return pd.DataFrame([
        {
            "time": pd.to_datetime(a[0], unit="ms"),
            "open": float(a[1]),
            "high": float(a[2]),
            "low": float(a[3]),
            "close": float(a[4]),
            "volume": float(a[5]),
        }
        for a in x
    ])


# ============================================================
# EXISTING SIGNAL ENGINE — KEEPING THE CURRENT LOGIC
# ============================================================

def indicators(df):
    d = df.copy()

    d["ema9"] = d.close.ewm(
        span=9,
        adjust=False
    ).mean()

    d["ema21"] = d.close.ewm(
        span=21,
        adjust=False
    ).mean()

    delta = d.close.diff()

    gain = delta.clip(
        lower=0
    ).rolling(14).mean()

    loss = (-delta.clip(
        upper=0
    )).rolling(14).mean()

    rs = gain / loss.replace(
        0,
        np.nan
    )

    d["rsi"] = 100 - 100 / (1 + rs)

    tr = pd.concat(
        [
            d.high - d.low,
            (d.high - d.close.shift()).abs(),
            (d.low - d.close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    return d


def classify(row, prev):

    bull = 0
    bear = 0

    reasons = []

    # EMA TREND
    if row.close > row.ema9 > row.ema21:
        bull += 1
        reasons.append(
            "price above EMA9/EMA21"
        )

    if row.close < row.ema9 < row.ema21:
        bear += 1
        reasons.append(
            "price below EMA9/EMA21"
        )

    # EMA DIRECTION
    if (
        row.ema9 > prev.ema9
        and row.ema21 > prev.ema21
    ):
        bull += 1

    if (
        row.ema9 < prev.ema9
        and row.ema21 < prev.ema21
    ):
        bear += 1

    # RSI
    if 52 <= row.rsi <= 68:
        bull += 1
        reasons.append(
            "RSI bullish zone"
        )

    if 32 <= row.rsi <= 48:
        bear += 1
        reasons.append(
            "RSI bearish zone"
        )

    # CANDLE BODY
    rng = max(
        row.high - row.low,
        1e-9
    )

    body = abs(
        row.close - row.open
    )

    if body / rng >= 0.55:

        if row.close > row.open:
            bull += 1

        else:
            bear += 1

    # FINAL SIGNAL
    if bull >= 3 and bull > bear:

        return (
            "BUY",
            min(
                95,
                55 + 10 * bull
            ),
            "; ".join(reasons)
        )

    if bear >= 3 and bear > bull:

        return (
            "SELL",
            min(
                95,
                55 + 10 * bear
            ),
            "; ".join(reasons)
        )

    return (
        "WAIT",
        50,
        "Mixed conditions"
    )


def current_signal(df):

    d = indicators(df).dropna()

    if len(d) < 3:

        return (
            "WAIT",
            0,
            "Collecting data"
        )

    return classify(
        d.iloc[-1],
        d.iloc[-2]
    )


# ============================================================
# APP
# ============================================================

st.title(
    "👑 Angel King Crypto AI Trader V3"
)

st.caption(
    "Binance market data • "
    "1-minute scalping • "
    "live signal dashboard • "
    "trading OFF"
)


# MARKET SELECTOR
market = st.selectbox(
    "📊 Binance Market",
    list(MARKETS.keys()),
    index=0
)

symbol = MARKETS[market]


# GET LIVE BINANCE DATA
try:

    data = get_klines(
        symbol,
        300
    )

    signal, confidence, reason = current_signal(
        data
    )

except Exception as e:

    st.error(
        f"Market data error: {e}"
    )

    st.stop()


# ============================================================
# LIVE INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    market,
    f"${data.close.iloc[-1]:,.2f}"
)

col2.metric(
    "Signal",
    signal
)

col3.metric(
    "Confidence",
    f"{confidence}%"
)


# ============================================================
# CANDLESTICK CHART
# ============================================================

chart = go.Figure(
    go.Candlestick(
        x=data.time,
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close
    )
)

chart.update_layout(
    height=500,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    chart,
    use_container_width=True
)


# ============================================================
# SIGNAL DISPLAY
# ============================================================

if signal == "BUY":

    st.success(
        "🟢 BUY SIGNAL"
    )

elif signal == "SELL":

    st.error(
        "🔴 SELL SIGNAL"
    )

else:

    st.warning(
        "🟡 WAIT"
    )


st.write(
    "**Reason:**",
    reason
)


st.info(
    "Execution is OFF. "
    "This app does not place Binance trades."
)


st.caption(
    "🔄 Live Binance market data "
    "refreshes automatically every 5 seconds."
)


st.caption(
    "⚠️ Experimental software. "
    "Backtests are not guarantees of future performance. "
    "Do not use real funds based solely on these signals."
)


# ============================================================
# AUTOMATIC REFRESH
# ============================================================

time.sleep(5)

st.rerun()
