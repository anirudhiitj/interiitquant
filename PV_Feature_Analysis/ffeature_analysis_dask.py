import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filter import get_pairs_for_day
import math
import sys

warnings.filterwarnings('ignore')

# ==========================================
# SECTION 1: CORE MATH & HELPERS
# ==========================================

def ensure_pandas(df):
    if isinstance(df, dask_cudf.DataFrame):
        return df.compute().to_pandas()
    elif isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df

def rolling_fft_reconstruction(series, window=120, top_k=3):
    n = len(series)
    out = np.full(n, np.nan)
    series = np.array(series, dtype=float)

    for i in range(window, n):
        segment = series[i-window:i]
        fft_vals = np.fft.fft(segment)
        idx = np.argsort(np.abs(fft_vals))[::-1]
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[idx[:top_k]] = fft_vals[idx[:top_k]]
        reconstructed = np.fft.ifft(fft_filtered).real
        out[i] = reconstructed[-1]
    return out

def apply_recursive_filter(series, sc):
    n = len(series)
    out = np.full(n, np.nan)

    start_idx = 0
    if np.isnan(series[0]):
        valid = np.where(~np.isnan(series))[0]
        if len(valid) > 0:
            start_idx = valid[0]
        else:
            return out

    out[start_idx] = series[start_idx]

    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        val = series[i]
        if not np.isnan(val):
            out[i] = out[i-1] + c * (val - out[i-1])
        else:
            out[i] = out[i-1]

    return out

def get_super_smoother_array(prices, period):
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1

    ss = np.zeros(n)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]

    for i in range(2,n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]

    return ss                              


# ==========================================
# SECTION 2: INDICATOR GENERATORS
# ==========================================

def add_basic_super_smoothers(df, price_col="Price"):
    prices = df[price_col].values
    df["EhlersSuperSmoother"] = get_super_smoother_array(prices, 30)
    df["Ehlers_SuperSmoother"] = get_super_smoother_array(prices, 300)
    return df

def add_kama_suite(df, price_col="Price"):
    prices = df[price_col].values
    period, fast, slow = 15, 45, 165
    direction = df[price_col].diff(period).abs()
    volatility = df[price_col].diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0)

    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = (((er * (fast_sc - slow_sc)) + slow_sc)**2).values

    df['Kama'] = apply_recursive_filter(prices, sc)
    df['Kama_Slope'] = df['Kama'].diff().fillna(0).abs()
    return df

def add_kama_regime(df):
    if 'Kama_Slope' not in df.columns:
        raise ValueError("Kama must be calculated before regime.")

    slope=df['Kama_Slope']

    if not pd.api.types.is_timedelta64_dtype(df['Time']):
        try: times=pd.to_timedelta(df['Time'])
        except: return df
    else:
        times=df['Time']

    start_time = times.iloc[0]
    cutoff = start_time + pd.Timedelta(minutes=30)
    mask = times < cutoff
    first = df.loc[mask,"Kama_Slope"].dropna()

    if first.empty:
        threshold=0.0003
    else:
        threshold = np.percentile(first,99)/2

    df['KAMA_Threshold']=threshold

    mag = np.where(slope>threshold,1,0)
    pos_series=pd.Series(mag,index=df.index)
    r_pos = pos_series.rolling(window=120).sum().fillna(0)
    raw_regime = np.where(r_pos>20,1,0).astype(int)

    locked = np.zeros(len(raw_regime),dtype=int)
    hold=0
    for i in range(len(raw_regime)):
        if raw_regime[i]==1:
            hold=60
        if hold>0:
            locked[i]=1
            hold-=1
        else:
            locked[i]=raw_regime[i]

    df['KAMA_Regime']=locked
    return df

def add_AURA_suite(df):
    if "EhlersSuperSmoother" not in df.columns:
        raise ValueError("Super smoother missing.")

    ss=df["EhlersSuperSmoother"]

    period, fast, slow = 15,30,120
    direction = ss.diff(period).abs()
    volatility = ss.diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = ((er*(fast_sc-slow_sc))+slow_sc)**2

    df["AURA"]=apply_recursive_filter(ss.values,sc)
    df["AURA_Slope"]=df["AURA"].diff().fillna(0)
    df["AURA_Slope_MA"]=df["AURA_Slope"].rolling(5).mean()

    period2,fast2,slow2 = 15,60,180
    direction2 = ss.diff(period2).abs()
    volatility2 = ss.diff().abs().rolling(period2).sum()
    er2 = (direction2/volatility2).fillna(0).values
    fast_sc2=2/(fast2+1)
    slow_sc2=2/(slow2+1)
    sc2 = ((er2*(fast_sc2-slow_sc2))+slow_sc2)**2
    df["AURA_Slow"]=apply_recursive_filter(ss.values,sc2)

    period3,fast3,slow3 = 30,60,300
    direction3 = ss.diff(period3).abs()
    volatility3 = ss.diff().abs().rolling(period3).sum()
    er3 = (direction3/volatility3).fillna(0).values
    fast_sc3=2/(fast3+1)
    slow_sc3=2/(slow3+1)
    sc3=((er3*(fast_sc3-slow_sc3))+slow_sc3)**2
    df["AURA_Slowest"]=apply_recursive_filter(ss.values,sc3)

    return df

def add_awesome_oscillator(df, price_col="AURA"):
    """
    Calculates Awesome Oscillator (AO).
    Standard AO = SMA(Price, 5) - SMA(Price, 34).
    """
    sma_5 = df[price_col].rolling(window=5*60).mean()
    sma_34 = df[price_col].rolling(window=34*30).mean()
    df['Awesome_Oscillator'] = sma_5 - sma_34
    return df

def add_rvi(df, price_col="AURA", period=600):
    """
    Calculates Relative Vigor Index (RVI).
    """
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        dt_index = pd.to_datetime("2020-01-01") + df['Time']
    else:
        dt_index = pd.to_datetime(df['Time'], unit='s', origin='unix')
        
    temp = pd.Series(df[price_col].values, index=dt_index)
    ohlc = temp.resample('60s').ohlc()
    ohlc['close'] = ohlc['close'].ffill()
    ohlc['open'] = ohlc['open'].fillna(ohlc['close'])
    ohlc['high'] = ohlc['high'].fillna(ohlc['close'])
    ohlc['low'] = ohlc['low'].fillna(ohlc['close'])
    
    # RVI Calculation
    co = ohlc['close'] - ohlc['open']
    hl = ohlc['high'] - ohlc['low']
    
    # Triangular weighting (approx SWMA)
    num = (co + 2*co.shift(1) + 2*co.shift(2) + co.shift(3)) / 6
    den = (hl + 2*hl.shift(1) + 2*hl.shift(2) + hl.shift(3)) / 6
    
    rvi_calc = num.rolling(period).mean() / den.rolling(period).mean()
    rvi_signal = (rvi_calc + 2*rvi_calc.shift(1) + 2*rvi_calc.shift(2) + rvi_calc.shift(3)) / 6
    
    # Map back
    rvi_series = rvi_calc.reindex(dt_index, method='ffill')
    sig_series = rvi_signal.reindex(dt_index, method='ffill')
    
    df['RVI'] = rvi_series.values
    df['RVI_Signal'] = sig_series.values
    
    return df

def add_rsi_on_AURA(df, source_col="AURA", period=14):
    if source_col not in df.columns:
        return df
    delta = df[source_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["AURA_RSI"] = rsi.fillna(50) 
    return df

def add_adx_on_AURA(df, source_col="AURA", period=14):
    if source_col not in df.columns:
        return df
    delta = df[source_col].diff()
    plus_dm = np.where(delta > 0, delta, 0.0)
    minus_dm = np.where(delta < 0, abs(delta), 0.0)
    tr = abs(delta)
    series_pdm = pd.Series(plus_dm, index=df.index)
    series_mdm = pd.Series(minus_dm, index=df.index)
    series_tr = pd.Series(tr, index=df.index)
    smooth_pdm = series_pdm.ewm(alpha=1/period, adjust=False).mean()
    smooth_mdm = series_mdm.ewm(alpha=1/period, adjust=False).mean()
    smooth_tr = series_tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (smooth_pdm / smooth_tr.replace(0, np.nan)).fillna(0)
    minus_di = 100 * (smooth_mdm / smooth_tr.replace(0, np.nan)).fillna(0)
    denom = plus_di + minus_di
    dx = 100 * (abs(plus_di - minus_di) / denom.replace(0, np.nan)).fillna(0)
    df['AURA_ADX'] = dx.ewm(alpha=1/period, adjust=False).mean()
    return df

def add_kAURA_suite(df):
    if "AURA" not in df.columns:
        raise ValueError("AURA missing.")

    AURA=df["AURA"]
    period,fast,slow = 15,30,120
    direction=AURA.diff(period).abs()
    volatility=AURA.diff().abs().rolling(period).sum()
    er=(direction/volatility).fillna(0).values
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2
    df["KAURA"]=apply_recursive_filter(AURA.values,sc)
    df["KAURA_Slope"]=df["KAURA"].diff().fillna(0).abs()
    return df

def add_ftama_suite(df, price_col="Price"):
    fft_vals = rolling_fft_reconstruction(df[price_col], window=120, top_k=3)
    df["FFT"]=fft_vals
    period,fast,slow=15,30,120

    fft_s = pd.Series(fft_vals)
    direction=fft_s.diff(period).abs()
    volatility=fft_s.diff().abs().rolling(period).sum()
    er=(direction/volatility).fillna(0).values
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2

    df["FTAMA"]=apply_recursive_filter(fft_vals,sc)
    return df

def add_ehler_slope_regime(df, source_col="EhlersSuperSmoother_Slow", window=300):
    if source_col not in df.columns:
        return df

    slope=df[source_col].diff().fillna(0)
    pos=np.where(slope>0,slope,0)
    neg=np.where(slope<0,abs(slope),0)
    pos_s=pd.Series(pos,index=df.index)
    neg_s=pd.Series(neg,index=df.index)
    r_pos=pos_s.rolling(window).sum().fillna(0)
    r_neg=neg_s.rolling(window).sum().fillna(0)
    num=np.maximum(r_pos,r_neg)
    den=np.minimum(r_pos,r_neg)+1e-9
    df["SS_Slope_Ratio"]=num/den
    return df

def add_hybrid_ham(df, price_col="Price"):
    if "EhlersSuperSmoother" not in df.columns:
        raise ValueError("Missing smoother.")

    prices=df[price_col].values
    n=len(prices)
    fft_slow=rolling_fft_reconstruction(prices,120,3)
    er_period=15
    fast,slow=30,120

    direction=df[price_col].diff(er_period).abs().values
    volatility=df[price_col].diff().abs().rolling(er_period).sum().values

    er=np.zeros(n)
    mask=volatility!=0
    er[mask]=direction[mask]/volatility[mask]

    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=(er*(fast_sc-slow_sc)+slow_sc)**2
    sc=np.nan_to_num(sc,0.0)

    fast_ma=df["EhlersSuperSmoother"].values
    slow_ma=fft_slow
    ham_vals=sc*fast_ma+(1-sc)*slow_ma
    ham_vals=pd.Series(ham_vals).bfill().ffill().values

    df["HAM"]=ham_vals
    thr=0.02
    smooth=np.zeros(n)
    smooth[0]=ham_vals[0]

    for i in range(1,n):
        if abs(ham_vals[i]-smooth[i-1])>thr:
            smooth[i]=ham_vals[i]
        else:
            smooth[i]=smooth[i-1]

    df["HAM_Smooth"]=smooth
    df["HAM_Slope"]=np.append([0],np.diff(ham_vals))
    df["HAM_MA"]=pd.Series(ham_vals).rolling(30).mean()
    df["HAM_EMA"]=pd.Series(ham_vals).ewm(span=30).mean()
    return df

def add_hamaz(df, price_col="Price"):
    if "AURA" not in df.columns:
        raise ValueError("AURA missing.")

    z_per=30
    lag=(z_per-1)//2
    z_dat=2*df[price_col]-df[price_col].shift(lag)
    zlema=z_dat.ewm(span=z_per,adjust=False).mean()

    AURA=df["AURA"]
    er_period=15
    fast,slow=30,120
    direction=df[price_col].diff(er_period).abs()
    volatility=df[price_col].diff().abs().rolling(er_period).sum()
    er=(direction/volatility).fillna(0)
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2

    df["HAMAZ"]=sc*zlema+(1-sc)*AURA
    return df

def add_jma(df, price_col="Price"):
    length=1200
    prices=df[price_col].values
    vol=np.abs(np.diff(prices,prepend=prices[0]))
    vol_smooth=pd.Series(vol).rolling(length,min_periods=1).mean().fillna(0).values
    vol_mean=pd.Series(vol_smooth).rolling(length,min_periods=1).mean().replace(0,1).values
    vol_norm=np.clip(vol_smooth/vol_mean,0.1,10)
    alpha=2/(length+1)
    alpha_adapt=np.clip(alpha*(1+0.5*(vol_norm-1)),0.001,0.999)

    out=np.empty(len(prices))
    out[0]=prices[0]
    for i in range(1,len(prices)):
        out[i]=alpha_adapt[i]*prices[i]+(1-alpha_adapt[i])*out[i-1]

    df["JMA"]=out
    return df

def add_hama(df):
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        dt_index=pd.to_datetime("2020-01-01")+df['Time']
    else:
        dt_index=pd.to_datetime(df['Time'],unit='s',origin='unix')

    price_series=pd.Series(df["Price"].values,index=dt_index)
    ohlc=price_series.resample('60s').ohlc()
    ohlc['close']=ohlc['close'].ffill()
    ohlc['open']=ohlc['open'].fillna(ohlc['close'])
    ohlc['high']=ohlc['high'].fillna(ohlc['close'])
    ohlc['low']=ohlc['low'].fillna(ohlc['close'])

    open_arr=ohlc['open'].values
    high_arr=ohlc['high'].values
    low_arr=ohlc['low'].values
    close_arr=ohlc['close'].values
    n=len(ohlc)
    ha_open=np.zeros(n)
    ha_close=np.zeros(n)
    ha_open[0]=(open_arr[0]+close_arr[0])/2
    ha_close[0]=(open_arr[0]+high_arr[0]+low_arr[0]+close_arr[0])/4

    for i in range(1,n):
        ha_close[i]=(open_arr[i]+high_arr[i]+low_arr[i]+close_arr[i])/4
        ha_open[i]=(ha_open[i-1]+ha_close[i-1])/2

    ha_close_series=pd.Series(ha_close,index=ohlc.index)
    hama_calc=ha_close_series.rolling(50,min_periods=1).mean()
    hama_shift=hama_calc.shift(1).fillna(ha_close[0])
    mapped=hama_shift.reindex(dt_index,method='ffill')
    df["HAMA"]=mapped.values
    df["HAMA"]=df["HAMA"].fillna(df["Price"].iloc[0])
    return df

def _calculate_vhf_series(series, period):
    roll_max=series.rolling(period).max()
    roll_min=series.rolling(period).min()
    num=(roll_max-roll_min).abs()
    diffs=series.diff().abs()
    denom=diffs.rolling(period).sum()
    return np.divide(num,denom,out=np.zeros_like(num),where=denom!=0)

def add_price_30s_vhf(df, price_col="Price", vhf_period=20):
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        dt_index=pd.to_datetime("2020-01-01")+df['Time']
    else:
        dt_index=pd.to_datetime(df['Time'],unit='s',origin='unix')

    temp=pd.Series(df[price_col].values,index=dt_index)
    res=temp.resample('30s').ohlc()['close']
    vhf=_calculate_vhf_series(res,vhf_period)
    df['Price_VHF_30s']=vhf.reindex(dt_index,method='ffill').values
    return df

def vhf_30s_regime(df, vhf_col="Price_VHF_30s", threshold=0.3):
    df["VHF_30s_Regime"]=np.where(df[vhf_col]>threshold,1,0)
    return df

def add_AURA_30s_vhf(df, AURA_col="AURA", vhf_period=30):
    if AURA_col not in df.columns:
        return df

    if pd.api.types.is_timedelta64_dtype(df['Time']):
        dt_index=pd.to_datetime("2020-01-01")+df['Time']
    else:
        dt_index=pd.to_datetime(df['Time'],unit='s',origin='unix')

    temp=pd.Series(df[AURA_col].values,index=dt_index)
    res=temp.resample('30s').last()
    vhf=_calculate_vhf_series(res,vhf_period)
    df['AURA_VHF_30s']=vhf.reindex(dt_index,method='ffill').values
    return df

def add_HYDRA_MA(df, window=300):
    required=["Diff_STD_STD","STD_Threshold","AURA","AURA_Slow"]
    miss=[c for c in required if c not in df.columns]
    if miss:
        return df

    raw=np.where(df["Diff_STD_STD"]>df["STD_Threshold"],1,0)
    cnt=pd.Series(raw).rolling(window).sum().fillna(0)
    w=cnt/window
    df["HYDRA_MA"]=w*df["AURA"]+(1-w)*df["AURA_Slow"]

    if pd.api.types.is_timedelta64_dtype(df["Time"]):
        start=df["Time"].iloc[0]
        cutoff=start+pd.Timedelta(minutes=60)
        mask=df["Time"]<cutoff
        df.loc[mask,"HYDRA_MA"]=df.loc[mask,"AURA_Slow"]
    
    df['Cross_Diff'] = df['AURA'] - df['HYDRA_MA']
    df['Cross_Diff'] = df['Cross_Diff'].fillna(0)
    df['Cross_DIff_MA'] = df['Cross_Diff'].rolling(3600).mean()
    
    return df

def add_supersmoother_regime(df, price_col="Price", ss_col="EhlersSuperSmoother_Slow", window=300, threshold=8):
    if ss_col not in df.columns:
        return df

    position=np.where(df[price_col]>df[ss_col],1,0)
    cross=pd.Series(position).diff().abs().fillna(0)
    cnt=cross.rolling(window).sum().fillna(0)
    df["SS_Cross_Count"]=cnt
    df["SS_Regime"]=np.where(cnt>threshold,0,1)
    return df

def add_PAMA_suite(df, price_col="Price"):
    sma=df[price_col].rolling(30).mean()
    period=15
    fast,slow=30,120
    direction=sma.diff(period).abs()
    volatility=sma.diff().abs().rolling(period).sum()
    er=(direction/volatility).fillna(0).values
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2

    df["PAMA"]=apply_recursive_filter(sma.values,sc)
    df["PAMA_Slope"]=df["PAMA"].diff().fillna(0)
    df["PAMA_Slope_MA"]=df["PAMA_Slope"].rolling(5).mean()
    return df

def add_zlama_suite(df, price_col="Price"):
    z_per=30
    lag=(z_per-1)//2
    z_dat=2*df[price_col]-df[price_col].shift(lag)
    zsource=z_dat.ewm(span=z_per,adjust=False).mean()

    period=15
    fast,slow=30,120
    direction=zsource.diff(period).abs()
    volatility=zsource.diff().abs().rolling(period).sum()
    er=(direction/volatility).fillna(0).values
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2

    df["ZLAMA"]=apply_recursive_filter(zsource.values,sc)
    df["ZLAMA_Slope"]=df["ZLAMA"].diff().fillna(0)
    df["ZLAMA_Slope_MA"]=df["ZLAMA_Slope"].rolling(5).mean()
    return df

def add_ekf(df):
    ekf=ExtendedKalmanFilter(dim_x=1,dim_z=1)
    ekf.x=np.array([[df["Price"].iloc[0]]])
    ekf.F=np.array([[1]])
    ekf.Q=np.array([[0.07]])
    ekf.R=np.array([[0.2]])

    def h(x): return np.log(x)
    def H_j(x): return np.array([[1/x[0][0]]])

    vals=[]
    clean=df["Price"].dropna()
    for p in clean.values:
        ekf.predict()
        ekf.update(np.array([[np.log(p)]]),HJacobian=H_j,Hx=h)
        vals.append(ekf.x.item())

    df["EKFTrend"]=pd.Series(vals,index=clean.index).ffill().shift(1)
    return df

def add_ucm(df):
    n=len(df)
    alpha,beta=0.005,0.0001
    mu=np.zeros(n)
    slope=np.zeros(n)
    filt=np.zeros(n)

    fvi=df['Price'].first_valid_index()
    if fvi is not None:
        mu[fvi]=df['Price'].loc[fvi]
        filt[fvi]=mu[fvi]
        for t in range(fvi+1,n):
            px=df["Price"].iloc[t]
            if np.isnan(px):
                mu[t],slope[t],filt[t]=mu[t-1],slope[t-1],filt[t-1]
                continue
            pred=mu[t-1]+slope[t-1]
            mu[t]=pred+alpha*(px-pred)
            slope[t]=slope[t-1]+beta*(px-pred)
            filt[t]=mu[t]+slope[t]

    df["UCM"]=filt
    return df

def add_ekama_suite(df):
    if "EKFTrend" not in df.columns:
        raise ValueError("EKFTrend missing.")

    series=df["EKFTrend"]
    period,fast,slow=15,30,120
    direction=series.diff(period).abs()
    volatility=series.diff().abs().rolling(period).sum()
    er=(direction/volatility).fillna(0).values
    fast_sc=2/(fast+1)
    slow_sc=2/(slow+1)
    sc=((er*(fast_sc-slow_sc))+slow_sc)**2
    df["EKAMA"]=apply_recursive_filter(series.values,sc)
    df["EKAMA_Slope"]=df["EKAMA"].diff().fillna(0)
    df["EKAMA_Slope_MA"]=df["EKAMA_Slope"].rolling(5).mean()
    return df

def add_AURA_ekf(df, source_col="AURA"):
    if source_col not in df.columns:
        return df

    ekf=ExtendedKalmanFilter(dim_x=1,dim_z=1)
    clean=df[source_col].dropna()
    if clean.empty:
        df["AURA_EKF"]=np.nan
        return df

    ekf.x=np.array([[clean.iloc[0]]])
    ekf.F=np.array([[1]])
    ekf.Q=np.array([[0.07]])
    ekf.R=np.array([[0.2]])

    def h(x): return np.log(x)
    def H_j(x): return np.array([[1/x[0][0]]])

    vals=[]
    for v in clean.values:
        ekf.predict()
        ekf.update(np.array([[np.log(v)]]),HJacobian=H_j,Hx=h)
        vals.append(ekf.x.item())

    series=pd.Series(vals,index=clean.index)
    df["AURA_EKF"]=series.reindex(df.index).ffill().shift(1)
    return df

def add_standard_indicators(df):
    df['ATR']=df['AURA'].diff().abs().ewm(60).mean()
    df['Slow_MA']=df['Price'].rolling(1200).mean()
    df['Slow_MA_Slope']=df['Slow_MA'].diff()
    df['Fast_MA']=df['Price'].rolling(180).mean()
    df['STD']=df['Price'].rolling(20).std()
    df['EMA_Fast']=df['Price'].ewm(span=25).mean()
    df['EMA_Slow']=df['Price'].ewm(span=1200).mean()
    df['Diff_STD']=df['Price'].diff().rolling(120).std()
    df['Diff_STD_STD']=df['Diff_STD'].rolling(120).std()

    z_per=1500
    lag=(z_per-1)//2
    z_dat=2*df['Price']-df['Price'].shift(lag)
    df['ZLEMA']=z_dat.ewm(z_per).mean()
    return df

def add_std_threshold(df):
    if "Diff_STD_STD" not in df.columns:
        return df

    if not pd.api.types.is_timedelta64_dtype(df['Time']):
        try: times=pd.to_timedelta(df['Time'])
        except: return df
    else:
        times=df["Time"]

    start=times.iloc[0]
    cutoff=start+pd.Timedelta(minutes=60)
    first=df.loc[times<cutoff,"Diff_STD_STD"].dropna()

    if first.empty:
        df["STD_Threshold"]=np.nan
    else:
        df["STD_Threshold"]=np.percentile(first,99)/3.0

    return df

# ---------- Fisher Transform (Ehlers style) ----------
def add_fisher_transform(df, source_col="AURA", length=600):
    """
    Ehlers Fisher Transform applied on a normalized source.
    Produces sharp spikes for sudden changes in the source.
    """
    if source_col not in df.columns:
        return df

    x = df[source_col].astype(float).copy()
    # rolling min/max with a reasonable min_periods to start producing values
    maxv = x.rolling(length, min_periods=1).max()
    minv = x.rolling(length, min_periods=1).min()

    rng = (maxv - minv).replace(0, np.nan)
    # normalize to -1..1, clip to avoid infinite log
    norm = ((x - minv) / rng * 2 - 1).clip(-0.999, 0.999).fillna(0)

    fisher = np.zeros(len(norm))
    trigger = np.zeros(len(norm))

    for i in range(1, len(norm)):
        v = norm.iloc[i]
        # handle nan safely
        if np.isnan(v):
            v = 0.0
        # classic fisher transform
        f = 0.5 * np.log((1 + v) / (1 - v))
        fisher[i] = 0.33 * f + 0.67 * fisher[i - 1]
        trigger[i] = fisher[i - 1]

    df["Fisher"] = fisher
    df["Fisher_Signal"] = trigger
    return df

# ==========================================
# SECTION 3: PLOTTING (SINGLE PLOT)
# ==========================================

def plot_market_analysis(data_df, selected_features=None):

    exclude=['Time','Price','Day','PlotIndex','KAMA_Mode','KAMA_Switch',
             'HA_Open','HA_High','HA_Low','HA_Close','Price_VHF_30s','AURA_VHF_30s',
             'RVI_Signal','Buy_Signal','Sell_Signal'] 

    cols=[c for c in data_df.columns if c not in exclude]
    if selected_features:
        cols=[c for c in cols if c in selected_features]

    fig=make_subplots(
        rows=1, cols=1, shared_xaxes=True,
        specs=[[{"secondary_y":True}]]
    )

    for day, df_day in data_df.groupby("Day"):
        
        # 1. Base Line (Gray/Black)
        fig.add_trace(
            go.Scatter(
                x=df_day["PlotIndex"], y=df_day["Price"],
                mode='lines', line=dict(width=1, color='black'),
                opacity=1.0,
                showlegend=False, hoverinfo='skip'
            ),
            row=1, col=1, secondary_y=False
        )

        # 2. Buy Signals (Green Triangles)
        if "Buy_Signal" in df_day.columns:
            mask_buy = df_day["Buy_Signal"].notna()
            if mask_buy.any():
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask_buy, "PlotIndex"],
                        y=df_day.loc[mask_buy, "Buy_Signal"],
                        mode='markers',
                        name='Buy (JMA Cross Up)',
                        marker=dict(symbol='triangle-up', color='green', size=20),
                        showlegend=True
                    ),
                    row=1, col=1, secondary_y=False  # Changed to False (Primary Axis)
                )

        # 3. Sell Signals (Red Triangles)
        if "Sell_Signal" in df_day.columns:
            mask_sell = df_day["Sell_Signal"].notna()
            if mask_sell.any():
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask_sell, "PlotIndex"],
                        y=df_day.loc[mask_sell, "Sell_Signal"],
                        mode='markers',
                        name='Sell (JMA Cross Down)',
                        marker=dict(symbol='triangle-down', color='red', size=20),
                        showlegend=True
                    ),
                    row=1, col=1, secondary_y=False # Changed to False (Primary Axis)
                )

        colors=['#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b',
                '#e377c2','#7f7f7f','#bcbd22','#17becf']

        for i,col in enumerate(cols):
            mask=df_day[col].notna()
            if mask.sum()==0: continue
            
            # --- CUSTOM STYLING FOR AURA (Bold & Bright Cyan) ---
            if col == "AURA":
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask,"PlotIndex"],
                        y=df_day.loc[mask,col],
                        mode='lines', name=col, legendgroup=col,
                        visible='legendonly',
                        line=dict(color='#00E5FF', width=3.5) # Bright Cyan, Wide
                    ),
                    row=1,col=1,secondary_y=True
                )

            # --- CUSTOM STYLING FOR HYDRA_MA (Bold & Bright Magenta) ---
            elif col == "HYDRA_MA":
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask,"PlotIndex"],
                        y=df_day.loc[mask,col],
                        mode='lines', name=col, legendgroup=col,
                        visible='legendonly',
                        line=dict(color='#FF00FF', width=3.5) # Bright Magenta, Wide
                    ),
                    row=1,col=1,secondary_y=True
                )

            # --- AWESOME OSCILLATOR HANDLING ---
            elif col == "Awesome_Oscillator":
                ao_vals = df_day.loc[mask, col]
                prev = ao_vals.shift(1).fillna(0)
                ao_colors = np.where(ao_vals >= prev, 'green', 'red')
                
                fig.add_trace(
                    go.Bar(
                        x=df_day.loc[mask,"PlotIndex"],
                        y=ao_vals,
                        name=col,
                        marker_color=ao_colors,
                        opacity=0.3,
                        visible='legendonly'
                    ),
                    row=1, col=1, secondary_y=True
                )
            
            # --- RVI HANDLING ---
            elif col == "RVI":
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask,"PlotIndex"],
                        y=df_day.loc[mask,col],
                        mode='lines',name="RVI",
                        line=dict(color='green', width=1.5),
                        visible='legendonly'
                    ),
                    row=1,col=1,secondary_y=True
                )
                if "RVI_Signal" in df_day.columns:
                     fig.add_trace(
                        go.Scatter(
                            x=df_day.loc[mask,"PlotIndex"],
                            y=df_day.loc[mask,"RVI_Signal"],
                            mode='lines',name="RVI Signal",
                            line=dict(color='red', width=1.5),
                            visible='legendonly'
                        ),
                        row=1,col=1,secondary_y=True
                    )

            # --- Fisher Transform Plot ---
            elif col == "Fisher":
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask, "PlotIndex"],
                        y=df_day.loc[mask, col],
                        mode="lines",
                        name="Fisher",
                        line=dict(color="purple", width=1.5),
                        visible="legendonly"
                    ),
                    row=1, col=1, secondary_y=True
                )
                if "Fisher_Signal" in df_day.columns:
                    sig_mask = df_day["Fisher_Signal"].notna()
                    fig.add_trace(
                        go.Scatter(
                            x=df_day.loc[sig_mask, "PlotIndex"],
                            y=df_day.loc[sig_mask, "Fisher_Signal"],
                            mode="lines",
                            name="Fisher Signal",
                            line=dict(color="orange", width=1.2, dash="dot"),
                            visible="legendonly"
                        ),
                        row=1, col=1, secondary_y=True
                    )

            # --- STANDARD LINES (Everything else) ---
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_day.loc[mask,"PlotIndex"],
                        y=df_day.loc[mask,col],
                        mode='lines',name=col,legendgroup=col,
                        visible='legendonly',
                        line=dict(color=colors[i%len(colors)],width=1.2)
                    ),
                    row=1,col=1,secondary_y=True
                )

    tick_vals=[ data_df[data_df["Day"]==d]["PlotIndex"].iloc[0]
                for d in range(int(data_df["Day"].max())+1)
                if not data_df[data_df["Day"]==d].empty ]
    tick_txt=[f"Day {d}" for d in range(len(tick_vals))]

    fig.update_layout(
        title="Market Analysis (Colored Price Trends)",
        hovermode="x unified",
        height=900, width=1700,
        plot_bgcolor="white",
        showlegend=True
    )

    fig.update_xaxes(tickvals=tick_vals,ticktext=tick_txt)
    fig.update_yaxes(gridcolor="rgba(220,220,220,0.4)",row=1,col=1)

    fig.show()


# ==========================================
# MAIN ANALYSIS
# ==========================================

def analyze_feature(data_df, price_pairs=None, selected_features=None):
    data_df=ensure_pandas(data_df)
    df=data_df[['Time','Price']].copy()
    df["Time"]=pd.to_timedelta(df["Time"])
    df["Day"]=(df["Time"].diff()<pd.Timedelta(0)).cumsum()
    df=df.dropna(subset=["Price"]).reset_index(drop=True)
    df["PlotIndex"]=range(len(df))

    df=add_basic_super_smoothers(df)
    df=add_ehler_slope_regime(df)
    df=add_kama_suite(df)

    df=add_AURA_suite(df)
    df = add_fisher_transform(df, source_col="AURA", length=600)

    # New Oscillators
    df=add_awesome_oscillator(df, price_col="Price")
    df=add_rvi(df, price_col="Price", period=600)
    
    df=add_rsi_on_AURA(df, source_col="AURA", period=900)
    df=add_adx_on_AURA(df, source_col="AURA", period=900)
    
    df=add_hybrid_ham(df)
    df=add_zlama_suite(df)
    df=add_AURA_ekf(df)
    df=add_hamaz(df)
    df=add_kAURA_suite(df)
    df=add_kama_regime(df)

    df=add_ftama_suite(df)
    df=add_jma(df)
    df=add_hama(df)
    df=add_PAMA_suite(df)
    df=add_ekf(df)
    df=add_ucm(df)
    df=add_ekama_suite(df)
    df=add_vhf(df)
    df=add_supersmoother_regime(df)
    df=add_standard_indicators(df)

    df=add_std_threshold(df)
    df=add_HYDRA_MA(df)

    # ==========================================
    # CROSSOVER LOGIC (JMA vs AURA_Slow)
    # ==========================================
    if "JMA" in df.columns and "AURA_Slow" in df.columns:
        jma = df['JMA']
        aura_s = df['AURA_Slow']
        
        # Buy: JMA crosses ABOVE AURA_Slow (Current JMA > AURA_Slow, Prev JMA <= Prev AURA_Slow)
        buy_cond = (jma > aura_s) & (jma.shift(1) <= aura_s.shift(1))
        
        # Sell: JMA crosses BELOW AURA_Slow (Current JMA < AURA_Slow, Prev JMA >= Prev AURA_Slow)
        sell_cond = (jma < aura_s) & (jma.shift(1) >= aura_s.shift(1))
        
        # Assign values for plotting (Use the Value of Price at the cross point)
        df['Buy_Signal'] = np.where(buy_cond, df['Price'], np.nan)
        df['Sell_Signal'] = np.where(sell_cond, df['Price'], np.nan)

    plot_market_analysis(df,selected_features)


def add_vhf(df, price_col="Price"):
    p=30
    series=df[price_col]
    max_v=series.rolling(p).max()
    min_v=series.rolling(p).min()
    num=(max_v-min_v).abs()
    den=series.diff().abs().rolling(p).sum()
    vhf=np.divide(num,den,out=np.zeros_like(num),where=den!=0)
    df["VHF"]=vhf
    return df


# ==========================================
# ENTRY POINT
# ==========================================

if __name__=="__main__":
    sys.path.append('.')
    days=[36]  # Specify days to process

    for d in days:
        print(f"Processing Day {d}...")
        success,pairs,df=get_pairs_for_day(d)
        if not success: continue

        feats=['AURA_Slow', 'AURA']

        analyze_feature(cudf.from_pandas(df),price_pairs=pairs,selected_features=feats)