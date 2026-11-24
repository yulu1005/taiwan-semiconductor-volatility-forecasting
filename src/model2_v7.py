# model2_v7.py — 依 V7 說明書的 Model 2：風險預算 → Σ̂＝D R_shrink D → 風險預算最佳化（凸問題，超過 30% 改 SLSQP）→ 波動目標
import warnings, numpy as np, pandas as pd                   # 載入：警告、數值、表格
from scipy.optimize import minimize                          # 載入：數值最佳化
from pypfopt.risk_models import CovarianceShrinkage          # 載入：PyPortfolioOpt 收縮估計（V7 5.3 指定）
warnings.filterwarnings("ignore")                            # 關掉套件提示
CAP = 0.30                                                   # V7 6.3：內部權重上限 30%
MIN_N = 4                                                    # V7 4.3：至少 4 檔才執行正常最佳化

def corr_matrix(W, shrink=True):                             # 定義：一段報酬（日期 × 股票）→ 相關矩陣
    """W：決策日以前 L 日的開盤到開盤小數報酬。只保留每檔都有報酬的日子（V7 5.3：日期對齊、不得有缺值）。"""
    X = W.dropna()                                           # 有任一檔缺值（停牌）的日子整列刪除
    if shrink:                                               # V7 定案：常數相關目標 Ledoit–Wolf
        cs = CovarianceShrinkage(X, returns_data=True, frequency=252)  # 建立估計器
        C = np.asarray(cs.ledoit_wolf(shrinkage_target="constant_correlation"), float); lam = float(cs.delta)  # 收縮共變異數與收縮強度 λ
    else:                                                    # 穩健性測試：不收縮
        C = np.asarray(X.cov(), float); lam = 0.0
    d = np.sqrt(np.diag(C)); R = C / np.outer(d, d)          # 轉成相關矩陣（對角線為 1）
    ev = np.linalg.eigvalsh(R).min()                         # V7 數值檢查：最小特徵值
    return R, lam, len(X), ev                                # 回傳相關矩陣、λ、實際使用天數、最小特徵值


def window_correlation(returns, lookback, shrink=True):
    """V7 window estimate and the documented fallback for sparse observations."""
    R, lam, nobs, ev = corr_matrix(returns, shrink)
    failed = nobs < 0.75 * lookback
    if failed and returns.shape[1] > 1:
        sample = returns.corr(min_periods=int(0.5 * lookback)).fillna(0).to_numpy()
        np.fill_diagonal(sample, 1)
        n = len(sample)
        mean = (sample.sum() - n) / (n * (n - 1))
        target = np.full_like(sample, mean)
        np.fill_diagonal(target, 1)
        R, lam = 0.5 * sample + 0.5 * target, 0.5
    return R, lam, nobs, ev, failed

def cap_redistribute(q, cap=CAP):                            # 定義：把超過上限的部分按比例分給其他股票（SLSQP 起點用）
    q = q.copy()
    for _ in range(100):
        over = q > cap + 1e-12
        if not over.any(): break
        ex = (q[over] - cap).sum(); q[over] = cap; free = q < cap - 1e-12
        q[free] += ex * q[free] / q[free].sum()
    return q

def risk_contrib(q, Sig):                                    # 定義：各股票的風險貢獻比例 PRC
    m = Sig @ q; return q * m / (q @ m)

def erc(Sig, b, cap=CAP):                                    # 定義：V7 6.4 兩步求解
    n = len(b)
    y0 = b / np.sqrt(np.diag(Sig))                           # 起點
    f = lambda y: 0.5 * y @ Sig @ y - b @ np.log(y)          # 凸問題目標：½ yᵀΣy − Σ b ln y（Spinu 2013）
    g = lambda y: Sig @ y - b / y                            # 梯度
    res = minimize(f, y0, jac=g, method="L-BFGS-B", bounds=[(1e-12, None)] * n, options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20000})
    q = res.x / res.x.sum(); method = "convex"               # 正規化成內部權重
    if q.max() > cap + 1e-9:                                 # 有股票超過 30%：改解有上限的最小平方問題
        obj = lambda x: 1e4 * ((risk_contrib(x, Sig) - b) ** 2).sum()  # 6.2 的目標（放大讓數值穩定）
        res2 = minimize(obj, cap_redistribute(q), method="SLSQP", bounds=[(0, cap)] * n,
                        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-14, "maxiter": 2000})
        q = np.clip(res2.x, 0, cap); q = q / q.sum(); method = "slsqp"
    rc = risk_contrib(q, Sig)
    return q, rc, method

def allocate(sig, R, alpha, target):                         # 定義：一個決策日的完整配置（sig：年化波動；R：相關矩陣）
    n = len(sig); D = np.diag(sig); Sig = D @ R @ D           # Σ̂ = D R D（年化）
    b = sig ** (-alpha); b = b / b.sum()                     # V7 4.2：風險預算
    if n < MIN_N:                                            # V7 4.3 例外規則：Nₜ < 4
        q = np.full(n, 1 / n); sp = float(np.sqrt(q @ Sig @ q))  # 以等權重內部組合計算 σ̂ₚ
        a = min(1.0, target / sp) if target else np.nan
        w = np.full(n, min(a / n, CAP)) if target else np.full(n, np.nan)  # 每檔 min(aₜ ÷ Nₜ, 30%)，其餘現金
        return dict(q=q, w=w, a=float(np.nansum(w)) if target else np.nan, sigma_p=sp, b=b, rc=risk_contrib(q, Sig), method="exception", exception=True)
    q, rc, method = erc(Sig, b)                              # V7 6.4
    sp = float(np.sqrt(q @ Sig @ q))                         # V7 7.2：純股票組合預測波動
    a = min(1.0, target / sp) if target else np.nan          # 股票總曝險
    return dict(q=q, w=a * q if target else np.full(n, np.nan), a=a, sigma_p=sp, b=b, rc=rc, method=method, exception=False)
