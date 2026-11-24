# 03d_garch_oo.py — V7 2.4.1：第 18 項特徵「開盤到開盤 GARCH(1,1) 5 日波動預測」，寫入 data/processed/features_garch_oo.parquet
#   r = 100 × ln(O_t ÷ O_{t−1})；ε = r − μ；h_{t+1} = ω + a ε² + b h；h_{t+j} = h̄ + (a+b)^{j−1}(h_{t+1} − h̄)
#   GARCH^OO = √(252 × 平均(h_{t+2} … h_{t+6})) ÷ 100
#   規則：每 63 個交易日重估（擴張視窗，自第一個可用交易日起）、至少 250 個有效報酬、常態準最大概似、a+b<1；
#        V7 沒有封存期凍結 → 2025 年後照常重估；估計失敗時沿用上一季參數並記錄
import warnings, numpy as np, pandas as pd                   # 載入：警告控制、數值運算、表格
from common import CFG, PROC, ROOT, VTAB                     # 載入：設定與路徑
from garch import fit, recurse                               # 載入：單次估計與固定參數遞迴（與既有 GARCH 共用同一套數學）
from garch_open_returns import opening_returns, read_actions
warnings.filterwarnings("ignore")                            # 關掉收斂提示
BURN, EVERY = 250, 63                                        # V7：至少 250 個報酬；每 63 個交易日重估

def walk_oo(r, dates):                                       # 定義：單一股票的走勢前進（不凍結）
    n = len(r); out1, out5 = np.full(n, np.nan), np.full(n, np.nan); log = []  # 輸出與紀錄
    c = None                                                 # 目前使用的參數
    for b in range(BURN, n, EVERY):                          # 每個區塊起點（第一次在第 250 個報酬之後）
        try:                                                 # 嘗試用區塊起點以前的全部報酬估計
            c_new = fit(r[:b])                               # 擴張視窗：r[0] … r[b−1]
            ok = c_new["alpha"] >= 0 and c_new["beta"] >= 0 and c_new["alpha"] + c_new["beta"] < 1 and c_new["omega"] > 0  # V7 的參數限制
            if not ok: raise ValueError("參數不符合限制")      # 不符合就當成失敗
            c = c_new; log.append({"fit_end": dates[b - 1], "status": "ok", **c})  # 成功：換新參數
        except Exception as e:                               # 估計失敗
            log.append({"fit_end": dates[b - 1], "status": f"failed: {e}" + ("，沿用上一季參數" if c else "，尚無參數")})  # 記錄
            if c is None: continue                           # 第一次就失敗：這個區塊沒有預測
        e = min(b + EVERY, n); v1, v5 = recurse(r[:e], c)    # 用目前參數遞迴到區塊終點（每天用最新報酬更新 h）
        out1[b:e], out5[b:e] = v1[b:e], v5[b:e]              # 只寫入本區塊
    return out1, out5, log                                   # 回傳

if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")          # 日資料（已還原）
    rows, logs = [], []                                      # 收集
    returns = opening_returns(panel[panel.sid.isin(CFG["universe"]["stocks"])], read_actions(ROOT))
    for sid, g in returns.groupby("sid"):
        r, d = g.r.to_numpy(), g.date.to_numpy()
        v1, v5, lg = walk_oo(r, pd.DatetimeIndex(d))         # 走勢前進
        rows.append(pd.DataFrame({"date": d, "sid": sid, "garch_oo1": v1, "garch_oo5": v5}))  # 一步預測與 5 日（t+2…t+6）平均
        logs += [{"sid": sid, **x} for x in lg]              # 估計紀錄
    out = pd.concat(rows); out.to_parquet(PROC / "features_garch_oo.parquet", index=False)  # 寫出特徵
    L = pd.DataFrame(logs); L.to_csv(VTAB / "garch_oo_fits.csv", index=False)  # 寫出每次估計的參數與狀態
    print(out.groupby("sid").apply(lambda t: t.dropna(subset=["garch_oo5"]).date.min(), include_groups=False).dt.date.to_dict())  # 每檔第一筆有值的日子
    print("估計次數", len(L), "| 失敗", int((L.status != "ok").sum()), "| a+b 中位數", round((L.alpha + L.beta).median(), 3))  # 摘要
