# hmm_regime.py — 大盤（0050）兩狀態隱馬可夫模型：重估時程、濾波機率（只用當天以前）、封存期凍結（範本 04 C.1）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from hmmlearn.hmm import GaussianHMM                         # 載入：高斯 HMM
from scipy.stats import norm                                 # 載入：常態分配（算發射機率）
from common import CFG                                       # 載入：設定

M = CFG["v3"]["model_based"]["hmm"]                          # HMM 設定
HOLDOUT_START = pd.Timestamp(CFG["protocol"]["holdout"]["test"][0])  # 封存期起點

def market_obs(panel):                                       # 定義：0050 的觀測值（日報酬、20 日實現波動，百分比）
    b = panel[panel.sid == M["market"]].set_index("date").sort_index()  # 取出 0050
    r = 100 * np.log(b.aclose / b.aclose.shift(1))          # 日報酬（%）
    rv = np.sqrt(252 / 20 * (r ** 2).rolling(20, min_periods=20).sum())  # 20 日實現波動（%，年化）
    X = pd.DataFrame({"log_return": r, "rv20": rv}).dropna()  # 兩欄觀測
    return X                                                 # 回傳（索引為日期）

def fit(X):                                                  # 定義：多個起點擬合，取對數概似最高者（固定種子）
    best, ll = None, -np.inf                                 # 目前最佳
    for seed in range(M["n_restarts"]):                      # 多個起點
        m = GaussianHMM(n_components=M["n_states"], covariance_type="diag", n_iter=200, random_state=seed, tol=1e-4)  # 模型
        try: m.fit(X); s = m.score(X)                        # 擬合並計分
        except Exception: continue                           # 失敗就換下一個起點
        if s > ll: best, ll = m, s                           # 保留最好的
    return best                                              # 回傳

def filtered_stress(m, X):                                   # 定義：前向濾波——第 t 天的機率只用第 t 天以前的觀測（不能用 predict_proba，那是平滑機率）
    stress = int(np.argmax(m.means_[:, 1]))                  # 平均波動較高的狀態＝壓力狀態（固定順序）
    sd = np.sqrt(np.stack([np.diag(c) for c in m.covars_]))  # 各狀態各欄的標準差
    logp = np.stack([norm.logpdf(X, m.means_[k], sd[k]).sum(axis=1) for k in range(m.n_components)], axis=1)  # 發射對數機率
    a = np.log(m.startprob_ + 1e-300) + logp[0]              # 第一天
    out = np.empty(len(X)); A = np.log(m.transmat_ + 1e-300)  # 輸出與轉移矩陣（對數）
    for t in range(len(X)):                                  # 逐日前向
        if t > 0: a = logp[t] + np.logaddexp.reduce(a[:, None] + A, axis=0)  # 前一天的濾波分布 × 轉移 × 今天的發射
        a = a - np.logaddexp.reduce(a)                       # 正規化（避免溢位）
        out[t] = np.exp(a[stress])                           # 今天在壓力狀態的機率
    return out                                               # 回傳

def walk(X):                                                 # 定義：重估時程（暖身 → 每 refit_every 天用至今資料重估；封存期凍結）
    v = X.values; d = X.index; n = len(v)                    # 觀測、日期、長度
    out = np.full(n, np.nan); fits = []                      # 輸出與紀錄
    freeze = int((d < HOLDOUT_START).sum())                  # 封存期前的天數
    m = None                                                 # 目前模型
    for b in range(M["burnin_sessions"], n, M["refit_every"]):  # 每個區塊起點
        if b <= freeze or m is None:                         # 封存期前才重估
            mm = fit(v[:min(b, freeze)])                     # 只用區塊起點以前的資料
            if mm is not None: m = mm; fits.append({"fit_end": d[min(b, freeze) - 1], "vol_calm": float(m.means_[:, 1].min()), "vol_stress": float(m.means_[:, 1].max())})  # 記錄
        if m is None: continue                               # 還沒有模型
        e = min(b + M["refit_every"], n)                     # 區塊終點
        out[b:e] = filtered_stress(m, v[:e])[b:e]            # 從頭濾波到區塊終點，只寫入本區塊
    return pd.Series(out, index=d, name="hmm_stress"), pd.DataFrame(fits)  # 回傳
