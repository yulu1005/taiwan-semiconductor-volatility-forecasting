# model2.py — Model 2 資產配置（V1 架構）：風險預算 → 共變異數 Σ = DRD → 等風險最佳化 → 組合波動目標
#   每週決策日只用 t 日收盤以前的資料；股票只做多、不借錢；剩下的錢全部留現金（不含黃金）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from scipy.optimize import minimize                          # 載入：有限制的最佳化（SLSQP）

def shrink_cc(X):                                            # 定義：Ledoit–Wolf（2004）常數相關收縮，X＝T 天 × N 檔的報酬
    X = X - X.mean(0)                                        # 每檔減去平均（只保留波動）
    X = X / X.std(0, ddof=0)                                 # 每檔標準化成變異數 1（收縮後就是相關係數矩陣）
    t, n = X.shape                                           # 天數、檔數
    S = X.T @ X / t                                          # 樣本相關矩陣
    var = np.diag(S).copy(); sv = np.sqrt(var)               # 對角線（約為 1）與其平方根
    rbar = ((S / np.outer(sv, sv)).sum() - n) / (n * (n - 1))  # 所有配對的平均相關（收縮目標的常數）
    F = rbar * np.outer(sv, sv); np.fill_diagonal(F, var)    # 收縮目標：每一對都等於平均相關
    Y = X ** 2                                               # 報酬平方
    phiM = Y.T @ Y / t - 2 * (X.T @ X) * S / t + S ** 2      # 每個元素的估計變異（Ledoit–Wolf 的 π）
    phi = phiM.sum()                                         # 加總
    H = X.T @ X / t; hd = np.diag(H)                         # 輔助矩陣
    th = (X ** 3).T @ X / t - hd[:, None] * S - H * var[None, :] + var[:, None] * S  # Ledoit–Wolf 的 θ
    np.fill_diagonal(th, 0)                                  # 對角線不用
    rho = np.diag(phiM).sum() + rbar * ((1 / sv)[:, None] * sv[None, :] * th).sum()  # Ledoit–Wolf 的 ρ
    gamma = ((S - F) ** 2).sum()                             # 樣本矩陣與目標的距離
    lam = float(np.clip((phi - rho) / gamma / t, 0, 1)) if gamma > 0 else 1.0  # 最佳收縮強度 λ（0～1）
    R = (1 - lam) * S + lam * F                              # 收縮後矩陣
    d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)          # 確保對角線剛好是 1
    return R, lam                                            # 回傳相關矩陣與 λ

def risk_budget(sig, alpha):                                 # 定義：風險預算 b ∝ σ̂^(−α)，加總 100%
    b = np.asarray(sig, float) ** (-alpha)                   # α＝0 時每檔相同
    return b / b.sum()                                       # 正規化

def erc(Sig, b, cap):                                        # 定義：在 0 ≤ q ≤ cap、Σq = 1 下，讓每檔風險貢獻比例接近 b
    n = len(b); cap = np.minimum(np.asarray(cap, float), 1.0)  # 檔數、上限
    if cap.sum() < 1 - 1e-9: return cap / cap.sum(), np.nan  # 上限加總不到 100%：內部權重依上限比例分配（曝險另降為上限加總，見 weekly）
    x0 = 1 / np.sqrt(np.diag(Sig)); x0 = x0 / x0.sum()      # 起點：反波動率
    for _ in range(50):                                      # 起點若超過上限，把超出部分分給其他檔
        over = x0 > cap
        if not over.any(): break
        ex = (x0[over] - cap[over]).sum(); x0[over] = cap[over]; free = ~over & (x0 < cap)
        x0[free] += ex * x0[free] / x0[free].sum()
    def f(q):                                                # 目標函數：Σ(風險貢獻比例 − 預算)²
        m = Sig @ q; v = q @ m                               # 邊際風險、組合變異
        return 1e4 * (((q * m) / v - b) ** 2).sum()          # 放大 1 萬倍讓數值穩定
    res = minimize(f, x0, method="SLSQP", bounds=list(zip(np.zeros(n), cap)),
                   constraints=[{"type": "eq", "fun": lambda q: q.sum() - 1}], options={"ftol": 1e-12, "maxiter": 500})  # 求解
    q = np.clip(res.x, 0, cap); q = q / q.sum()              # 修掉數值誤差
    m = Sig @ q; dev = np.abs(q * m / (q @ m) - b).max()     # 最大的「風險預算偏差」
    return q, dev                                            # 回傳內部權重與偏差

def weekly(dec_date, sids, sig, adv20, rets, P2):            # 定義：一個決策日的完整配置
    """sids：當週可交易股票；sig：年化預測波動；adv20：20 日平均成交金額；rets：開盤到開盤日報酬（列＝日期、欄＝股票）"""
    n = len(sids)                                            # 可交易檔數
    if n == 0: return pd.Series(dtype=float), dict(a=0.0, n=0)  # 沒有可交易股票：全部現金
    W = rets.loc[:dec_date, sids].iloc[-P2["corr_window"]:]  # 決策日以前（含）最近 60 個交易日
    W = W.dropna()                                           # 只用每檔都有報酬的日子
    if len(W) >= P2["corr_min_obs"] and n >= 2:              # 資料夠才估相關
        R, lam = shrink_cc(W.values) if P2["shrink"] else (np.corrcoef(W.values.T), 0.0)  # 收縮或樣本相關
    else:
        R, lam = np.eye(n), np.nan                           # 資料不足：視為互不相關（記錄下來）
    D = np.diag(sig); Sig = D @ R @ D                        # Σ = DRD
    cap = np.minimum(P2["name_cap"], P2["adv_cap"] * adv20 / P2["capital"])  # 單檔上限：30% 與「一天買得到」取小
    b = risk_budget(sig, P2["alpha"])                        # 風險預算
    q, dev = erc(Sig, b, cap)                                # 股票內部權重
    sp = float(np.sqrt(q @ Sig @ q))                         # 股票組合的預測年化波動
    amax = min(P2["max_exposure"], float(cap.sum()))         # 曝險上限：100%，但買得到的上限加總不到 100% 時再降低
    a = float(np.clip(P2["target_vol"] / sp, 0, amax)) if P2.get("target_vol") else np.nan  # 股票總曝險
    if P2.get("fixed_exposure") is not None: a = min(P2["fixed_exposure"], amax)  # 比較組：固定曝險
    if P2.get("full_exposure"): a = amax                     # 比較組：股票滿倉（只看最佳化本身）
    w = pd.Series(a * q if not np.isnan(a) else q, index=sids)  # 最終權重（佔總資金）
    return w, dict(a=a, n=n, sigma_p=sp, lam=lam, rb_dev=dev, cap_sum=float(cap.sum()), q_max=float(q.max()))  # 權重與診斷
