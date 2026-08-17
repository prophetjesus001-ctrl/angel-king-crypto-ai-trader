import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# ANGEL KING CRYPTO AI TRADER V3
# Binance market data • 1-minute scalping
# Trading execution intentionally OFF
# ============================================================

st.set_page_config(
    page_title="Angel King Crypto AI Trader V3",
    page_icon="👑",
    layout="wide"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

INTERVAL = "1m"

MARKETS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT"
]


# ============================================================
# BINANCE DATA
# ============================================================

def get_symbol(market):
    return market.replace("/", "")


def get_klines(symbol, limit=300):
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit
    }

    r = requests.get(
        url,
        params=params,
        timeout=15
    )

    r.raise_for_status()

    data = r.json()

    return pd.DataFrame(
        [
            {
                "time": pd.to_datetime(a[0], unit="ms"),
                "open": float(a[1]),
                "high": float(a[2]),
                "low": float(a[3]),
                "close": float(a[4]),
                "volume": float(a[5])
            }
            for a in data
        ]
    )


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

    gain = delta.clip(
        lower=0
    ).rolling(14).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(14).mean()

    rs = gain / loss.replace(
        0,
        np.nan
    )

    d["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # ATR 14
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close"].shift()).abs(),
            (d["low"] - d["close"].shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    # Candle information
    d["body"] = (
        d["close"] - d["open"]
    ).abs()

    d["range"] = (
        d["high"] - d["low"]
    )

    d["body_ratio"] = np.where(
        d["range"] > 0,
        d["body"] / d["range"],
        0
    )

    d["bull_candle"] = (
        (d["close"] > d["open"]) &
        (d["body_ratio"] >= 0.55)
    )

    d["bear_candle"] = (
        (d["close"] < d["open"]) &
        (d["body_ratio"] >= 0.55)
    )

    return d


# ============================================================
# EXISTING SIGNAL ENGINE
# ============================================================

def classify(row, prev):

    bull = 0
    bear = 0

    reasons = []

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if row.close > row.ema9 > row.ema21:

        bull += 1
        reasons.append(
            "price above EMA9 and EMA21"
        )

    elif row.close < row.ema9 < row.ema21:

        bear += 1
        reasons.append(
            "price below EMA9 and EMA21"
        )

    # --------------------------------------------------------
    # EMA DIRECTION
    # --------------------------------------------------------

    if prev is not None:

        if (
            row.ema9 > prev.ema9 and
            row.ema21 > prev.ema21
        ):

            bull += 1
            reasons.append(
                "EMA direction bullish"
            )

        elif (
            row.ema9 < prev.ema9 and
            row.ema21 < prev.ema21
        ):

            bear += 1
            reasons.append(
                "EMA direction bearish"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if 52 <= row.rsi <= 68:

        bull += 1
        reasons.append(
            "RSI supports bullish momentum"
        )

    elif 32 <= row.rsi <= 48:

        bear += 1
        reasons.append(
            "RSI supports bearish momentum"
        )

    # --------------------------------------------------------
    # CANDLE COLOR + STRENGTH
    # --------------------------------------------------------

    if row.bull_candle:

        bull += 1
        reasons.append(
            "strong bullish candle"
        )

    elif row.bear_candle:

        bear += 1
        reasons.append(
            "strong bearish candle"
        )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if bull >= 3 and bull > bear:

        signal = "BUY"

        confidence = min(
            95,
            55 + 10 * bull
        )

        reason = "; ".join(reasons)

    elif bear >= 3 and bear > bull:

        signal = "SELL"

        confidence = min(
            95,
            55 + 10 * bear
        )

        reason = "; ".join(reasons)

    else:

        signal = "WAIT"

        confidence = 50

        reason = "Mixed conditions"

    return signal, confidence, reason, bull, bear


# ============================================================
# ANALYZE MARKET
# ============================================================

def analyze_market(df):

    d = indicators(df)

    if len(d) < 3:

        return (
            d,
            "WAIT",
            0,
            "Collecting data",
            0,
            0
        )

    row = d.iloc[-1]

    prev = d.iloc[-2]

    signal, confidence, reason, bull, bear = classify(
        row,
        prev
    )

    return (
        d,
        signal,
        confidence,
        reason,
        bull,
        bear
    )


# ============================================================
# TOP SIGNAL INDICATOR
# ONLY UI CHANGE REQUESTED
# ============================================================

def show_top_signal(signal):

    if signal == "BUY":

        st.markdown(
            """
            <div style="
                background:#d9f7df;
                border-left:7px solid #16a34a;
                border-radius:14px;
                padding:16px 20px;
                margin:10px 0 18px 0;
                color:#15803d;
                font-size:27px;
                font-weight:700;
            ">
                🟢 BUY
            </div>
            """,
            unsafe_allow_html=True
        )

    elif signal == "SELL":

        st.markdown(
            """
            <div style="
                background:#ffe0e0;
                border-left:7px solid #dc2626;
                border-radius:14px;
                padding:16px 20px;
                margin:10px 0 18px 0;
                color:#b91c1c;
                font-size:27px;
                font-weight:700;
            ">
                🔴 SELL
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="
                background:#fff9c4;
                border-left:7px solid #eab308;
                border-radius:14px;
                padding:16px 20px;
                margin:10px 0 18px 0;
                color:#a16207;
                font-size:27px;
                font-weight:700;
            ">
                🟡 WAIT
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CHART
# ============================================================

def show_chart(df, market):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=market
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema9"],
            name="EMA 9",
            mode="lines"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema21"],
            name="EMA 21",
            mode="lines"
        )
    )

    fig.update_layout(
        height=480,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(df, fee=0.10, slippage=0.02, rr=2.0):

    d = indicators(df)

    trades = []

    for i in range(2, len(d) - 1):

        row = d.iloc[i]
        prev = d.iloc[i - 1]

        signal, confidence, reason, bull, bear = classify(
            row,
            prev
        )

        if signal not in ["BUY", "SELL"]:
            continue

        entry = float(
            d.iloc[i + 1]["open"]
        )

        next_close = float(
            d.iloc[i + 1]["close"]
        )

        gross_return = (
            (next_close - entry) / entry
        ) * 100

        if signal == "SELL":

            gross_return = -gross_return

        net_return = (
            gross_return
            - (fee * 2)
            - (slippage * 2)
        )

        outcome = (
            "WIN"
            if net_return > 0
            else "LOSS"
        )

        trades.append(
            {
                "time": d.iloc[i + 1]["time"],
                "signal": signal,
                "confidence": confidence,
                "entry": entry,
                "close": next_close,
                "return_pct": net_return,
                "outcome": outcome,
                "reason": reason
            }
        )

    return pd.DataFrame(trades)


# ============================================================
# APP
# ============================================================

st.title(
    "👑 Angel King Crypto AI Trader V3"
)

st.caption(
    "Binance market data • 1-minute scalping • "
    "live signal dashboard • trading OFF"
)


# ============================================================
# MARKET SELECTION
# ============================================================

market = st.selectbox(
    "📊 Binance Market",
    MARKETS,
    index=0
)

symbol = get_symbol(market)


# ============================================================
# LIVE DATA
# ============================================================

try:

    hist = get_klines(
        symbol,
        300
    )

    (
        data,
        signal,
        confidence,
        reason,
        bull,
        bear
    ) = analyze_market(hist)

except Exception as e:

    st.error(
        f"Unable to retrieve Binance market data: {e}"
    )

    st.stop()


# ============================================================
# TOP SIGNAL
# ============================================================

show_top_signal(signal)


# ============================================================
# CURRENT MARKET INFORMATION
# ============================================================

current_price = float(
    data.iloc[-1]["close"]
)

st.subheader(market)

st.metric(
    "Price",
    f"${current_price:,.2f}"
)


# ============================================================
# SIGNAL INFORMATION
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Signal",
        signal
    )

with c2:

    st.metric(
        "Confidence",
        f"{confidence:.0f}%"
    )


# ============================================================
# SIGNAL REASON
# ============================================================

if signal == "BUY":

    st.success(
        f"Reason: {reason}"
    )

elif signal == "SELL":

    st.error(
        f"Reason: {reason}"
    )

else:

    st.warning(
        f"Reason: {reason}"
    )


# ============================================================
# EXECUTION OFF
# ============================================================

st.info(
    "Execution is OFF. This app does not place Binance trades."
)


# ============================================================
# CHART
# ============================================================

show_chart(
    data.tail(100),
    market
)


# ============================================================
# AUTO REFRESH
# ============================================================

st.caption(
    "🔄 Live Binance market data refreshes automatically "
    "every 5 seconds."
)


# ============================================================
# BACKTEST
# ============================================================

st.divider()

tab1, tab2 = st.tabs(
    [
        "📡 Live Signal",
        "🧪 Backtest"
    ]
)


with tab1:

    st.subheader(
        f"Live {market} 1-minute signal"
    )

    st.write(
        f"Current signal: **{signal}**"
    )

    st.write(
        f"Confidence: **{confidence:.0f}%**"
    )

    st.write(
        f"Reason: **{reason}**"
    )


with tab2:

    st.subheader(
        f"Historical {market} 1-minute test"
    )

    n = st.slider(
        "Candles to test",
        200,
        1000,
        1000
    )

    fee = st.number_input(
        "Fee per side (%)",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

    slippage = st.number_input(
        "Slippage per side (%)",
        min_value=0.0,
        value=0.02,
        step=0.01
    )

    rr = st.select_slider(
        "Risk/Reward",
        options=[
            1.0,
            1.5,
            2.0,
            2.5,
            3.0
        ],
        value=2.0
    )

    if st.button(
        "Run backtest",
        type="primary"
    ):

        with st.spinner(
            "Downloading and testing historical data..."
        ):

            try:

                hist = get_klines(
                    symbol,
                    n
                )

                trades = backtest(
                    hist,
                    fee,
                    slippage,
                    rr
                )

            except Exception as e:

                st.error(
                    f"Backtest error: {e}"
                )

                st.stop()

        if trades.empty:

            st.warning(
                "No qualifying trades found."
            )

        else:

            wins = (
                trades["outcome"] == "WIN"
            ).sum()

            losses = (
                trades["outcome"] == "LOSS"
            ).sum()

            winrate = (
                wins / len(trades) * 100
            )

            gross_win = trades.loc[
                trades["return_pct"] > 0,
                "return_pct"
            ].sum()

            gross_loss = abs(
                trades.loc[
                    trades["return_pct"] < 0,
                    "return_pct"
                ].sum()
            )

            profit_factor = (
                gross_win / gross_loss
                if gross_loss > 0
                else np.inf
            )

            net_return = (
                trades["return_pct"].sum()
            )

            equity = (
                trades["return_pct"]
                .cumsum()
            )

            drawdown = (
                equity -
                equity.cummax()
            )

            max_drawdown = drawdown.min()

            a, b, c, d = st.columns(4)

            a.metric(
                "Trades",
                len(trades)
            )

            b.metric(
                "Win rate",
                f"{winrate:.1f}%"
            )

            c.metric(
                "Net return",
                f"{net_return:.2f}%"
            )

            d.metric(
                "Profit factor",
                "∞"
                if np.isinf(profit_factor)
                else f"{profit_factor:.2f}"
            )

            st.metric(
                "Max drawdown",
                f"{max_drawdown:.2f}%"
            )

            chart = go.Figure()

            chart.add_trace(
                go.Scatter(
                    x=trades["time"],
                    y=equity,
                    mode="lines",
                    name="Equity"
                )
            )

            chart.update_layout(
                height=400
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            st.dataframe(
                trades.tail(100),
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "⚠️ Experimental software. Backtests are not guarantees "
    "of future performance. Do not use real funds based "
    "solely on these signals."
)


# ============================================================
# AUTOMATIC 5-SECOND REFRESH
# ============================================================

time.sleep(5)
st.rerun()
