# features_v2.py — 候選特徵（v2 的 27 項 + v3 新增大盤 HMM，共 28 項）：依波動預測文獻與第一輪檢查結果重新設計
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from features import rmean, rsum, rstd                       # 載入：移動平均、加總、標準差（窗內有空值就給空值）

def rvar(x, n): return x.rolling(n, min_periods=n).var()     # 定義：n 日移動變異數

def stock_features(g, rm):                                   # 定義：單一股票的特徵（g＝依日期排序的單檔資料，rm＝0050 日報酬）
    C, O, Hh, Lw = g.aclose, g.aopen, g.ahigh, g.alow        # 還原後收、開、高、低
    r = np.log(C / C.shift(1))                               # 收盤到收盤報酬
    ro = np.log(O / O.shift(1))                              # 開盤到開盤報酬（與標籤同定義）
    on = np.log(O / C.shift(1))                              # 隔夜報酬（前收到今開）
    oc = np.log(C / O)                                       # 日內報酬（今開到今收）
    f = pd.DataFrame(index=g.index)                          # 準備空表
    # --- A. 波動水準（HAR 結構：日／週／月／季） ---
    f["rv1"] = r.abs() * np.sqrt(252)                        # 當日絕對報酬年化（日成分）
    for n in (5, 20, 60): f[f"rv{n}"] = np.sqrt(252 / n * rsum(r ** 2, n))  # 5、20、60 日實現波動
    f["oo20"] = np.sqrt(252 / 20 * rsum(ro ** 2, 20))        # 開盤到開盤 20 日波動（與標籤同一把尺）
    # --- B. 區間型估計量（用高低價，比收盤價更有效率） ---
    f["hl"] = (Hh - Lw) / C                                  # 當日高低價差比
    f["park20"] = np.sqrt(252 / (4 * np.log(2)) * rmean(np.log(Hh / Lw) ** 2, 20))  # Parkinson 20 日波動
    rs = np.log(Hh / C) * np.log(Hh / O) + np.log(Lw / C) * np.log(Lw / O)  # Rogers–Satchell 單日項
    k = 0.34 / (1.34 + 21 / 19)                              # Yang–Zhang 權重（n=20）
    f["yz20"] = np.sqrt(252 * (rvar(on, 20) + k * rvar(oc, 20) + (1 - k) * rmean(rs, 20)).clip(lower=0))  # Yang–Zhang 20 日波動
    tr = pd.concat([Hh - Lw, (Hh - C.shift()).abs(), (Lw - C.shift()).abs()], axis=1).max(axis=1, skipna=False)  # 真實波幅
    f["atr_p"] = rmean(tr, 14) / C                           # ATR14 ÷ 股價
    # --- C. 隔夜跳空（標籤包含隔夜，收盤波動看不到） ---
    f["on20"] = np.sqrt(252 / 20 * rsum(on ** 2, 20))        # 隔夜 20 日波動
    # --- D. 不對稱與尾部 ---
    f["semi_down20"] = np.sqrt(252 / 20 * rsum(r.clip(upper=0) ** 2, 20))  # 下方半波動
    f["ret20"] = np.log(C / C.shift(20))                     # 20 日報酬（跌越多之後越波動：槓桿效應）
    f["maxabs20"] = r.abs().rolling(20, min_periods=20).max()  # 20 日內最大單日變動（跳動）
    # --- E. 波動的動態 ---
    f["term5_60"] = f.rv5 / f.rv60                           # 短長期波動比（期限結構：>1 表示最近變熱）
    f["volvol20"] = rstd(r.abs(), 20) / rmean(r.abs(), 20)   # 波動的波動
    # --- F. 市場與系統性風險 ---
    cov = (r * rm).rolling(60, min_periods=60).mean() - rmean(r, 60) * rmean(rm, 60)  # 與 0050 的 60 日共變異
    var_m = rvar(rm, 60) * 59 / 60                           # 0050 的 60 日變異（母體）
    f["beta60"] = cov / var_m                                # 60 日 beta
    f["idio60"] = np.sqrt(252 * (rvar(r, 60) * 59 / 60 - f.beta60 ** 2 * var_m).clip(lower=0))  # 個股特有波動
    # --- G. 量與流動性 ---
    vol = g.avolume                                          # 還原成交量
    f["vol_cv"] = rstd(vol, 20) / rmean(vol, 20)             # 成交量變動（Volume_Volatility）
    f["amt_ratio"] = g.amount / rmean(g.amount, 20)          # 成交金額比（Turnover）
    f["log_adv20"] = np.log(rmean(g.amount, 20))             # 20 日平均成交金額（取 log；規模／流動性）
    f["amihud20"] = np.log(rmean(r.abs() / g.amount * 1e8, 20))  # Amihud 非流動性（每億元成交造成的變動，取 log）
    # --- H. 趨勢 ---
    f["ma20_dist"] = C / rmean(C, 20) - 1                    # 與 20 日均線距離（MA_Gap_20）
    return f                                                 # 回傳

def calendar_features(cal, H=5):                             # 定義：行事曆特徵（交易所行事曆事先公告，決策時已知）
    out = []                                                 # 收集
    for p, d in enumerate(cal):                              # 逐一交易日
        w = cal[p + 1: p + H + 2]                            # 標籤窗口 t+1…t+6
        if len(w) < H + 1: out.append((d, np.nan, np.nan)); continue  # 窗口不完整
        rev = sum(x.day <= 10 for x in w)                    # 窗口內落在每月 1–10 日的天數（上市公司 10 日前公布月營收）
        gap = int(pd.Series(w.insert(0, d)).diff().dt.days.max())  # 窗口內最長休市間隔（農曆年、連假）
        out.append((d, rev, gap))                            # 記錄
    return pd.DataFrame(out, columns=["date", "rev_days", "holiday_gap"])  # 回傳

# 第二版候選特徵與家族、是否為「每日同值」（市場層級）
CANDIDATES = {
    "rv1": ("A 波動水準", False), "rv5": ("A 波動水準", False), "rv20": ("A 波動水準", False), "rv60": ("A 波動水準", False), "oo20": ("A 波動水準", False),
    "hl": ("B 區間估計", False), "park20": ("B 區間估計", False), "yz20": ("B 區間估計", False), "atr_p": ("B 區間估計", False),
    "on20": ("C 隔夜", False), "semi_down20": ("D 不對稱尾部", False), "ret20": ("D 不對稱尾部", False), "maxabs20": ("D 不對稱尾部", False),
    "term5_60": ("E 波動動態", False), "volvol20": ("E 波動動態", False), "garch_fc5": ("E 波動動態", False),
    "beta60": ("F 市場", False), "idio60": ("F 市場", False), "rv20_rank": ("F 市場", False), "mkt_rv20": ("F 市場", True),
    "vol_cv": ("G 量與流動性", False), "amt_ratio": ("G 量與流動性", False), "log_adv20": ("G 量與流動性", False), "amihud20": ("G 量與流動性", False),
    "ma20_dist": ("H 趨勢", False), "rev_days": ("I 行事曆", True), "holiday_gap": ("I 行事曆", True),
    "hmm_stress": ("F 市場", True)}                          # 28 項（v3 新增：大盤 HMM 壓力狀態機率，範本 04）
