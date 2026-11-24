# features.py — 15 項特徵的定義（管線 03 與 notebooks 共用同一份，避免兩套公式）
import numpy as np, pandas as pd                             # 載入：數值運算、表格

def rmean(x, n): return x.rolling(n, min_periods=n).mean()   # 定義：n 日移動平均（窗內有空值就給空值）
def rsum(x, n):  return x.rolling(n, min_periods=n).sum()    # 定義：n 日移動加總
def rstd(x, n):  return x.rolling(n, min_periods=n).std()    # 定義：n 日移動標準差

def make_features(g, L):                                     # 定義：對單一股票、回看期 L 計算 15 項特徵
    a, b, c = max(2, round(L / 4)), round(L / 2), L          # 三個波動窗：20 日版為 5/10/20
    k = round(0.7 * L)                                       # RSI/ATR 窗：20 日版為 14
    C, O, Hh, Lw = g.aclose, g.aopen, g.ahigh, g.alow        # 取出還原後收、開、高、低
    r = np.log(C / C.shift(1))                               # 每日收盤對數報酬
    f = pd.DataFrame(index=g.index)                          # 準備空表放特徵
    for n in (a, b, c): f[f"hv{n}"] = np.sqrt(252 / n * rsum(r ** 2, n))  # 歷史波動（年化，與標籤同定義）
    f["ret1"] = r                                            # 1 日報酬
    f[f"ret{a}"] = np.log(C / C.shift(a))                    # 短期報酬
    f[f"ret{c}"] = np.log(C / C.shift(c))                    # 回看期報酬
    f["hl"] = (Hh - Lw) / C                                  # 當日高低價差比例
    f["oc"] = (C - O) / O                                    # 當日開收盤價差比例
    f[f"ma{a}_dist"] = C / rmean(C, a) - 1                   # 與短均線距離
    f[f"ma{c}_dist"] = C / rmean(C, c) - 1                   # 與長均線距離
    f["vol_ratio"] = g.avolume / rmean(g.avolume, c)         # 成交量比
    f["vol_cv"] = rstd(g.avolume, c) / rmean(g.avolume, c)   # 量的變動（標準差 ÷ 平均）
    f["amt_ratio"] = g.amount / rmean(g.amount, c)           # 成交金額比
    d = C.diff()                                             # 每日價格變化
    up, dn = rmean(d.clip(lower=0), k), rmean((-d).clip(lower=0), k)  # 固定窗平均漲幅與跌幅
    f["rsi"] = np.where(dn > 0, 100 - 100 / (1 + up / dn.replace(0, np.nan)), 100.0)  # RSI（無跌幅時為 100）
    f.loc[up.isna() | dn.isna(), "rsi"] = np.nan             # 窗內有空值就設空值
    tr = pd.concat([Hh - Lw, (Hh - C.shift()).abs(), (Lw - C.shift()).abs()], axis=1).max(axis=1, skipna=False)  # 真實波幅
    f["atr_p"] = rmean(tr, k) / C                            # ATR ÷ 股價
    return f                                                 # 回傳 15 項特徵
