import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import math
import pathlib

# ------------------ Bring in same JIT fallback ------------------
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not found — running in pure Python mode.")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# ------------------ CONFIG FOR PLOTTING ------------------
DAYS_TO_PLOT = [20]

CONFIG = {
    'DATA_DIR': '/data/quant14/EBY/',
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # INDICATORS
    'super_smoother_period': 30,
    'eama_er_period': 15,
    'eama_fast': 30,
    'eama_slow': 120,

    'eama_slow_er_period': 15,
    'eama_slow_fast': 60,
    'eama_slow_slow': 180,

    'hydra_window': 300,
    'hydra_std_period': 120,
    'std_warmup_seconds': 3600,

    # KAMA + Regime
    'kama_period': 15,
    'kama_fast': 25,
    'kama_slow': 100,

    'use_kama_regime_filter': True,
    'kama_regime_window': 120,
    'kama_regime_count_thresh': 20,
    'kama_regime_lockout': 60,
    'kama_regime_warmup_min': 30,

    # DAILY ACTIVATION
    'use_daily_activation': True,
    'activation_window_seconds': 1800,
    'activation_sigma_threshold': 0.15,

    # Strategy rules
    'strategy_start_tick': 1800,
}


# ================================================================
# NUMBA FUNCTIONS (SAME AS STRATEGY)
# ================================================================

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n=len(prices)
    ss=np.zeros(n)
    if n==0: return ss
    pi=3.141592653589793
    arg=1.414*pi/period
    a1=math.exp(-arg)
    b1=2*a1*math.cos(arg)
    c1=1-b1+a1*a1
    c2=b1
    c3=-a1*a1
    ss[0]=prices[0]
    if n>1: ss[1]=prices[1]
    for i in range(2,n):
        ss[i]=c1*prices[i]+c2*ss[i-1]+c3*ss[i-2]
    return ss

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, er_period, fast, slow):
    n=len(ss_values)
    eama=np.zeros(n)
    eama[0]=ss_values[0]
    f_sc=2/(fast+1)
    s_sc=2/(slow+1)
    for i in range(1,n):
        er=0.0
        if i>=er_period:
            direction=abs(ss_values[i]-ss_values[i-er_period])
            vol=0.0
            for j in range(er_period):
                vol+=abs(ss_values[i-j]-ss_values[i-j-1])
            if vol>1e-10:
                er=direction/vol
        sc=((er*(f_sc-s_sc))+s_sc)**2
        eama[i]=eama[i-1]+sc*(ss_values[i]-eama[i-1])
    return eama

@jit(nopython=True, fastmath=True)
def calculate_kama_numba(prices, period, fast, slow):
    n=len(prices)
    kama=np.zeros(n)
    kama[0]=prices[0]
    f_sc=2/(fast+1)
    s_sc=2/(slow+1)
    for i in range(1,n):
        er=0.0
        if i>=period:
            direction=abs(prices[i]-prices[i-period])
            vol=0.0
            for j in range(period):
                vol+=abs(prices[i-j]-prices[i-j-1])
            if vol>1e-10: er=direction/vol
        sc=((er*(f_sc-s_sc))+s_sc)**2
        kama[i]=kama[i-1]+sc*(prices[i]-kama[i-1])
    return kama

@jit(nopython=True, fastmath=True)
def calculate_rolling_std(series, window):
    n=len(series)
    out=np.zeros(n)
    for i in range(n):
        st=max(0, i-window+1)
        w=series[st:i+1]
        if len(w)>1: out[i]=np.std(w)
        else: out[i]=0
    return out

@jit(nopython=True, fastmath=True)
def calculate_diff_std_std(prices, window):
    n=len(prices)
    diff=np.zeros(n)
    for i in range(1,n):
        diff[i]=prices[i]-prices[i-1]
    diff_std=calculate_rolling_std(diff,window)
    diff_std_std=calculate_rolling_std(diff_std,window)
    return diff_std_std

@jit(nopython=True, fastmath=True)
def calculate_hydra_threshold(diff_std_std, warmup):
    if warmup<10: return 0.0
    c=diff_std_std[:warmup]
    valid=c[c>0]
    if len(valid)==0: return 0.0
    s=np.sort(valid)
    idx=int(len(s)*0.99)
    if idx>=len(s): idx=len(s)-1
    return s[idx]/2.5

@jit(nopython=True, fastmath=True)
def calculate_hydra_ma(eama, eama_slow, diff_std_std, threshold, window, warmup):
    n=len(eama)
    out=np.zeros(n)
    for i in range(n):
        if i<warmup:
            out[i]=eama_slow[i]
            continue
        st=max(0, i-window+1)
        c=0
        for j in range(st, i+1):
            if diff_std_std[j]>threshold: c+=1
        w=c/window
        out[i]=w*eama[i] + (1-w)*eama_slow[i]
    return out

@jit(nopython=True, fastmath=True)
def check_early_volatility_numba(timestamps, prices, window_seconds, sigma_thresh):
    n=len(prices)
    if n==0: return False, 0.0
    start=timestamps[0]
    cutoff=start+window_seconds
    s=0.0; c=0
    for i in range(n):
        if timestamps[i]>cutoff: break
        s+=prices[i]; c+=1
    if c<10: return False, 0.0
    mean=s/c
    var=0.0
    for i in range(c):
        d=prices[i]-mean
        var+=d*d
    sd=math.sqrt(var/c)
    return sd>sigma_thresh, sd

@jit(nopython=True, fastmath=True)
def calculate_kama_regime_numba(timestamps, kama, window, count_thresh, lockout, warmup_minutes):
    n=len(kama)
    regime=np.zeros(n, dtype=np.int8)
    if n<window: return regime, 0.0

    slope=np.zeros(n)
    for i in range(1,n): slope[i]=abs(kama[i]-kama[i-1])

    warm_sec=warmup_minutes*60.0
    start=timestamps[0]
    widx=0
    for i in range(n):
        if timestamps[i]-start>warm_sec:
            widx=i
            break

    thr=0.0003
    if widx>10:
        ws=np.sort(slope[:widx])
        idx=int(len(ws)*0.99)
        if idx>=len(ws): idx=len(ws)-1
        thr=ws[idx]/2

    is_high=(slope>thr).astype(np.int8)
    rs=0
    lock=0
    for i in range(n):
        rs+=is_high[i]
        if i>=window: rs-=is_high[i-window]
        raw = 1 if rs>count_thresh else 0
        if raw==1: lock=lockout
        if lock>0:
            regime[i]=1
            lock-=1
        else:
            regime[i]=0
    return regime, thr

@jit(nopython=True, fastmath=True)
def detect_crossover(a,b,i):
    if i<1: return 0
    eps=1e-10
    cd=a[i]-b[i]
    pd=a[i-1]-b[i-1]
    if pd<=eps and cd>eps: return 1
    if pd>=-eps and cd<-eps: return -1
    return 0


# ================================================================
# PLOT FUNCTION
# ================================================================
def plot_day(day, df, prices, eama, hydra, kama, signals, kama_regime, is_active, sigma):

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75,0.25],
        subplot_titles=[f"Day {day} — Price + HYDRA + EAMA + KAMA",
                        "KAMA Regime"]
    )

    # Row 1 — price + indicators
    fig.add_trace(go.Scattergl(x=df.index, y=prices, mode='lines',
                               name="Price", line=dict(width=1,color='white')), row=1,col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=eama, mode='lines',
                               name="EAMA", line=dict(width=1.2,color='cyan')), row=1,col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=hydra, mode='lines',
                               name="HYDRA MA", line=dict(width=1.2,color='orange')), row=1,col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=kama, mode='lines',
                               name="KAMA", line=dict(width=1.2,color='magenta')), row=1,col=1)

    # Signals
    buys=np.where(signals==1.0)[0]
    sells=np.where(signals==-1.0)[0]

    if len(buys)>0:
        fig.add_trace(go.Scatter(
            x=df.index[buys], y=prices[buys], mode='markers',
            marker=dict(symbol='triangle-up',size=12,color='lime'),
            name="BUY"
        ), row=1,col=1)

    if len(sells)>0:
        fig.add_trace(go.Scatter(
            x=df.index[sells], y=prices[sells], mode='markers',
            marker=dict(symbol='triangle-down',size=12,color='red'),
            name="SELL"
        ), row=1,col=1)

    # Row 2 — KAMA regime
    fig.add_trace(go.Scattergl(
        x=df.index, y=kama_regime, mode='lines',
        line=dict(width=1.2,color='yellow'),
        name="Regime (1 = No Trade)"
    ), row=2,col=1)

    # Layout
    fig.update_layout(
        template='plotly_dark',
        height=900,
        title=f"HYDRA Strategy Plot — Day {day} — Active: {is_active} (σ={sigma:.4f})",
        hovermode="x unified"
    )
    fig.show()


# ================================================================
# PROCESS DAY
# ================================================================
def process_day(day_num):

    f = os.path.join(CONFIG['DATA_DIR'], f"day{day_num}.parquet")
    if not os.path.exists(f):
        print(f"Missing parquet: {f}")
        return

    df = pd.read_parquet(f, columns=[CONFIG['TIME_COLUMN'], CONFIG['PRICE_COLUMN']])
    df = df.reset_index(drop=True)

    # Convert time → seconds
    df['Time_sec'] = pd.to_timedelta(df[CONFIG['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)

    # FFILL/BFILL like strategy
    df[CONFIG['PRICE_COLUMN']] = df[CONFIG['PRICE_COLUMN']].replace(0,np.nan).ffill().bfill()

    timestamps = df['Time_sec'].values.astype(float)
    prices = df[CONFIG['PRICE_COLUMN']].values.astype(float)

    # -------- Remove first 60 seconds --------
    cutoff = timestamps[0] + 60
    cut_idx = np.searchsorted(timestamps, cutoff, 'right')
    timestamps = timestamps[cut_idx:]
    prices = prices[cut_idx:]
    df = df.iloc[cut_idx:].copy()

    if len(prices)<200:
        print(f"Day {day_num}: Not enough ticks after cutoff.")
        return

    # -------- Daily Activation --------
    is_active, sigma = check_early_volatility_numba(
        timestamps, prices,
        CONFIG['activation_window_seconds'],
        CONFIG['activation_sigma_threshold']
    )

    # -------- Indicators --------
    ss = calculate_super_smoother_numba(prices, CONFIG['super_smoother_period'])
    eama = calculate_eama_numba(ss, CONFIG['eama_er_period'], CONFIG['eama_fast'], CONFIG['eama_slow'])
    eama_slow = calculate_eama_numba(ss, CONFIG['eama_slow_er_period'], CONFIG['eama_slow_fast'], CONFIG['eama_slow_slow'])
    diff_std_std = calculate_diff_std_std(prices, CONFIG['hydra_std_period'])
    warmup = int(CONFIG['std_warmup_seconds'])
    threshold = calculate_hydra_threshold(diff_std_std, warmup)
    hydra = calculate_hydra_ma(eama, eama_slow, diff_std_std, threshold, CONFIG['hydra_window'], warmup)
    kama = calculate_kama_numba(prices, CONFIG['kama_period'], CONFIG['kama_fast'], CONFIG['kama_slow'])

    kama_regime = np.zeros(len(prices), dtype=np.int8)
    if CONFIG['use_kama_regime_filter']:
        kama_regime, _ = calculate_kama_regime_numba(
            timestamps, kama,
            CONFIG['kama_regime_window'],
            CONFIG['kama_regime_count_thresh'],
            CONFIG['kama_regime_lockout'],
            CONFIG['kama_regime_warmup_min']
        )

    # -------- Recreate Strategy Signals --------
    signals = np.zeros(len(prices), dtype=np.float64)
    if not is_active:
        print(f"Day {day_num} INACTIVE — no signals.")
    else:
        last_t=-999999
        state=0
        for i in range(len(prices)):
            if i < CONFIG['strategy_start_tick']: continue

            cross = detect_crossover(eama, hydra, i)
            ct = timestamps[i]

            if state==0:
                if cross==1 and kama_regime[i]==0:
                    signals[i]=1.0; state=1; last_t=ct
                elif cross==-1 and kama_regime[i]==0:
                    signals[i]=-1.0; state=-1; last_t=ct

            elif state==1:
                if cross==-1:
                    signals[i]=-1.0; state=0; last_t=ct

            elif state==-1:
                if cross==1:
                    signals[i]=1.0; state=0; last_t=ct

    # Plot
    df_idx = df.set_index(pd.to_datetime(df['Time_sec'], unit='s', origin='unix'))

    plot_day(day_num, df_idx, prices, eama, hydra, kama,
             signals, kama_regime, is_active, sigma)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    for d in DAYS_TO_PLOT:
        process_day(d)
    print("Done.")
