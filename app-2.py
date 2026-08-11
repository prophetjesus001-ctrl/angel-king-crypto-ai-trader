import time, json, threading
from collections import deque
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import websocket

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
st.set_page_config(page_title="Angel King Crypto AI Trader V2", page_icon="👑", layout="wide")

def get_klines(limit=1000):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=15)
    r.raise_for_status()
    x = r.json()
    return pd.DataFrame([{
        "time": pd.to_datetime(a[0], unit="ms"),
        "open": float(a[1]), "high": float(a[2]), "low": float(a[3]),
        "close": float(a[4]), "volume": float(a[5])
    } for a in x])

def indicators(df):
    d = df.copy()
    d["ema9"] = d.close.ewm(span=9, adjust=False).mean()
    d["ema21"] = d.close.ewm(span=21, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100/(1+rs)
    tr = pd.concat([(d.high-d.low), (d.high-d.close.shift()).abs(),
                    (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d

def classify(row, prev):
    bull = bear = 0
    reasons = []
    if row.close > row.ema9 > row.ema21:
        bull += 1; reasons.append("price above EMA9/EMA21")
    if row.close < row.ema9 < row.ema21:
        bear += 1; reasons.append("price below EMA9/EMA21")
    if row.ema9 > prev.ema9 and row.ema21 > prev.ema21:
        bull += 1
    if row.ema9 < prev.ema9 and row.ema21 < prev.ema21:
        bear += 1
    if 52 <= row.rsi <= 68:
        bull += 1; reasons.append("RSI bullish zone")
    if 32 <= row.rsi <= 48:
        bear += 1; reasons.append("RSI bearish zone")
    rng = max(row.high-row.low, 1e-9)
    body = abs(row.close-row.open)
    if body/rng >= .55:
        if row.close > row.open: bull += 1
        else: bear += 1
    if bull >= 3 and bull > bear:
        return "BUY", min(95, 55+10*bull), "; ".join(reasons)
    if bear >= 3 and bear > bull:
        return "SELL", min(95, 55+10*bear), "; ".join(reasons)
    return "WAIT", 50, "Mixed conditions"

def current_signal(df):
    d = indicators(df).dropna()
    if len(d) < 3:
        return ("WAIT", 0, "Collecting data")
    return classify(d.iloc[-1], d.iloc[-2])

def backtest(df, fee_pct, slippage_pct, rr):
    d = indicators(df).dropna().reset_index(drop=True)
    trades=[]; i=1
    while i < len(d)-2:
        sig, conf, reason = classify(d.iloc[i], d.iloc[i-1])
        if sig == "WAIT":
            i += 1; continue
        entry = d.iloc[i+1].open
        atr = d.iloc[i].atr
        if not np.isfinite(atr) or atr <= 0:
            i += 1; continue
        risk = atr * 0.8
        if sig == "BUY":
            sl, tp = entry-risk, entry+risk*rr
        else:
            sl, tp = entry+risk, entry-risk*rr
        outcome=None; exit_price=None; exit_time=None
        for j in range(i+1, min(i+31, len(d))):
            hi, lo = d.iloc[j].high, d.iloc[j].low
            if sig=="BUY":
                if lo <= sl:
                    outcome="LOSS"; exit_price=sl; exit_time=d.iloc[j].time; break
                if hi >= tp:
                    outcome="WIN"; exit_price=tp; exit_time=d.iloc[j].time; break
            else:
                if hi >= sl:
                    outcome="LOSS"; exit_price=sl; exit_time=d.iloc[j].time; break
                if lo <= tp:
                    outcome="WIN"; exit_price=tp; exit_time=d.iloc[j].time; break
        if outcome is None:
            exit_price=d.iloc[min(i+30,len(d)-1)].close
            exit_time=d.iloc[min(i+30,len(d)-1)].time
            raw=(exit_price-entry)/entry if sig=="BUY" else (entry-exit_price)/entry
            outcome="WIN" if raw>0 else "LOSS"
        raw=(exit_price-entry)/entry if sig=="BUY" else (entry-exit_price)/entry
        net=raw-fee_pct/100*2-slippage_pct/100*2
        trades.append({"time":d.iloc[i].time,"side":sig,"confidence":conf,
                       "entry":entry,"exit":exit_price,"outcome":outcome,"return_pct":net*100})
        i=max(i+1,j+1)
    t=pd.DataFrame(trades)
    return t

st.title("👑 Angel King Crypto AI Trader V2")
st.caption("BTC/USDT • 1-minute scalping • research + live signal dashboard • trading OFF")

tab1, tab2 = st.tabs(["📡 Live Signal", "🧪 Backtest"])

with tab1:
    if "live" not in st.session_state:
        try: st.session_state.live = get_klines(300)
        except Exception as e: st.error(str(e)); st.stop()
    d=st.session_state.live
    sig, conf, reason=current_signal(d)
    a,b,c=st.columns(3)
    a.metric("BTC/USDT", f"${d.close.iloc[-1]:,.2f}")
    b.metric("Signal", sig)
    c.metric("Confidence", f"{conf}%")
    chart=go.Figure(go.Candlestick(x=d.time,open=d.open,high=d.high,low=d.low,close=d.close))
    chart.update_layout(height=500,xaxis_rangeslider_visible=False)
    st.plotly_chart(chart,use_container_width=True)
    if sig=="BUY": st.success("🟢 BUY SIGNAL")
    elif sig=="SELL": st.error("🔴 SELL SIGNAL")
    else: st.warning("🟡 WAIT")
    st.write("**Reason:**",reason)
    st.info("Execution is OFF. This app does not place Binance trades.")

with tab2:
    st.subheader("Historical BTC/USDT 1-minute test")
    n=st.slider("Candles to test", 200, 1000, 1000, 100)
    fee=st.number_input("Fee per side (%)",0.0,1.0,0.10,0.01)
    slip=st.number_input("Slippage per side (%)",0.0,1.0,0.02,0.01)
    rr=st.select_slider("Risk/Reward", options=[1.0,1.25,1.5,2.0,2.5,3.0], value=2.0)
    if st.button("Run backtest", type="primary"):
        with st.spinner("Downloading and testing candles..."):
            hist=get_klines(n)
            trades=backtest(hist,fee,slip,rr)
        if trades.empty:
            st.warning("No qualifying trades found.")
        else:
            wins=(trades.outcome=="WIN").sum()
            losses=(trades.outcome=="LOSS").sum()
            winrate=wins/len(trades)*100
            gross_win=trades.loc[trades.return_pct>0,"return_pct"].sum()
            gross_loss=abs(trades.loc[trades.return_pct<0,"return_pct"].sum())
            pf=gross_win/gross_loss if gross_loss else np.inf
            equity=trades.return_pct.cumsum()
            dd=(equity-equity.cummax()).min()
            x,y,z,q=st.columns(4)
            x.metric("Trades",len(trades)); y.metric("Win rate",f"{winrate:.1f}%")
            z.metric("Net return",f"{trades.return_pct.sum():.2f}%")
            q.metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
            st.metric("Max drawdown",f"{dd:.2f}%")
            st.plotly_chart(go.Figure(go.Scatter(x=trades.time,y=equity,mode="lines",name="Equity")),use_container_width=True)
            st.dataframe(trades.tail(100),use_container_width=True)

st.caption("Experimental software. Backtests are not guarantees of future performance. Do not use real funds based solely on these signals.")
