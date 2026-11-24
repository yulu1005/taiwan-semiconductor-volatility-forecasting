# garch.py — 每檔 GARCH(1,1) 走勢前進估計（管線 03b 與 notebooks/04 共用）
import warnings, numpy as np, pandas as pd                   # 載入：警告控制、數值運算、表格
from arch import arch_model                                  # 載入：GARCH 估計
from common import CFG                                       # 載入：設定
warnings.filterwarnings("ignore")                            # 關掉收斂提示
G = CFG["model_based"]["garch"]                              # GARCH 設定
BURN, EVERY = G["burnin_sessions"], G["refit_every"]         # 暖身天數、重估間隔
H = CFG["schedule"]["label_horizon_days"]                    # 標籤期間
HOLDOUT_START = pd.Timestamp(CFG["protocol"]["holdout"]["test"][0])  # 封存期起點

def fit(r):                                                  # 定義：用訓練報酬（%）估參數
    res = arch_model(r, mean="Constant", vol="GARCH", p=G["p"], q=G["q"], dist=G["dist"]).fit(disp="off", show_warning=False)  # 最大概似估計
    p = res.params                                           # 參數
    return {"mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
            "backcast": float(res.model.volatility.backcast(np.asarray(res.resid)))}  # 起始變異數只用訓練殘差

def recurse(r, c):                                           # 定義：固定參數跑變異數遞迴，回傳每天收盤後的一步與 5 日平均預測（年化）
    n = len(r); s2 = np.empty(n + 1); s2[0] = c["backcast"]  # s2[i]：第 i 天的條件變異數
    e = r - c["mu"]                                          # 意外
    for i in range(n): s2[i + 1] = c["omega"] + c["alpha"] * e[i] ** 2 + c["beta"] * s2[i]  # 只用到第 i 天
    one = s2[1:]                                             # 第 i 天收盤後對第 i+1 天的預測
    pers = c["alpha"] + c["beta"]                            # 持續性
    if pers >= 0.999: fc5 = one                              # 近乎單根：多步預測等於一步
    else:                                                    # 一般情況
        lr = c["omega"] / (1 - pers); h = np.arange(2, H + 2)  # 長期變異數、標籤期間 t+2…t+6
        fc5 = (lr + pers ** (h - 1)[None, :] * (one[:, None] - lr)).mean(axis=1)  # 期間平均變異數
    return np.sqrt(one * 252) / 100, np.sqrt(fc5 * 252) / 100  # 年化

def walk(r, dates):                                          # 定義：單一股票走勢前進（每 EVERY 天重估，封存期凍結）
    n = len(r); out1, out5 = np.full(n, np.nan), np.full(n, np.nan); fits = []  # 輸出與紀錄
    freeze = int((pd.DatetimeIndex(dates) < HOLDOUT_START).sum())  # 封存期前的報酬數
    c = None                                                 # 目前參數
    for b in range(BURN, n, EVERY):                          # 每個區塊起點
        if b <= freeze or c is None:                         # 封存期前才重估
            try: c = fit(r[:min(b, freeze)]); fits.append({"fit_end": dates[min(b, freeze) - 1], **c})  # 只用起點前資料
            except Exception: continue                       # 不收斂則該區塊留空
        e = min(b + EVERY, n); v1, v5 = recurse(r[:e], c)    # 遞迴到區塊終點
        out1[b:e], out5[b:e] = v1[b:e], v5[b:e]              # 只寫入本區塊
    return out1, out5, fits                                  # 回傳

def garch_panel(panel, sids):                                # 定義：對多檔股票計算 GARCH 特徵
    rows, fits = [], []                                      # 收集
    for sid, g in panel[panel.sid.isin(sids) & panel.valid].sort_values(["sid", "date"]).groupby("sid"):  # 逐檔（有效交易日）
        g = g.dropna(subset=["aclose"]); r = 100 * np.log(g.aclose / g.aclose.shift(1)).values[1:]; d = g.date.values[1:]  # 報酬（%）
        v1, v5, f = walk(r, pd.DatetimeIndex(d))             # 走勢前進
        rows.append(pd.DataFrame({"date": d, "sid": sid, "garch_vol1": v1, "garch_fc5": v5})); fits += [{"sid": sid, **x} for x in f]  # 存
    return pd.concat(rows), pd.DataFrame(fits)               # 回傳特徵與參數紀錄
