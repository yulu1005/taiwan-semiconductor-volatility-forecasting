# 03_features_labels.py — 建立 15 項特徵（10/20/40 日三種回看版本）與開盤到開盤 5 日實現波動標籤
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import ROOT, CFG, PROC, TAB                      # 載入：路徑與設定

panel = pd.read_parquet(PROC/"panel.parquet")                # 讀清理後面板
panel["avolume"] = panel.volume / panel.adj                  # 還原後成交量（股數依公司行動同步調整）
panel.loc[~panel.valid, ["avolume", "amount"]] = np.nan      # 無效日的量與金額設為空值
cal = pd.DatetimeIndex(sorted(panel.date.unique()))          # 交易日曆
H = CFG["schedule"]["label_horizon_days"]                    # 預測期間 5 日

from features import rmean, rsum, rstd, make_features   # 載入：特徵公式（與 notebooks 共用）

def realized(aO, h):                                         # 定義：t+1 開盤起 h 日的開盤到開盤實現波動
    ro = np.log(aO / aO.shift(1))                            # 開盤對數報酬
    return np.sqrt(252 / h * rsum(ro ** 2, h)).shift(-(h + 1))  # 加總 t+2…t+h+1 的報酬平方並年化

out = []                                                     # 收集每檔結果
for sid, g in panel.groupby("sid"):                          # 逐一股票
    g = g.set_index("date").sort_index()                     # 以日期為索引
    parts = [g[["valid", "open", "close", "amount", "aopen", "aclose", "adj"]]]  # 保留回測需要的原始欄位
    for L in (10, 20, 40):                                   # 三種回看期
        parts.append(make_features(g, L).add_suffix(f"_L{L}"))  # 計算並加上後綴
    lab = pd.DataFrame({"y": realized(g.aopen, H),           # 主標籤：5 日
                        "y10": realized(g.aopen, 10),        # IC 衰減用：10 日
                        "y20": realized(g.aopen, 20)}, index=g.index)  # IC 衰減用：20 日
    pos = cal.get_indexer(g.index)                           # 每天在日曆中的位置
    lab["label_end"] = [cal[p + H + 1] if p + H + 1 < len(cal) else pd.NaT for p in pos]  # 標籤結束日 t+6
    lab["adv20"] = rmean(g.amount, 20)                       # 過去 20 日平均成交金額（容量限制用）
    x = pd.concat(parts + [lab], axis=1); x["sid"] = sid     # 合併並加上代號
    out.append(x.reset_index())                              # 收進清單
df = pd.concat(out, ignore_index=True)                       # 合併所有股票

start = pd.Timestamp(CFG["data"]["study_start"])             # 研究起點
grid = cal[cal >= start][::CFG["schedule"]["decision_every_n_days"]]  # 決策日：起點後每 5 個交易日
df["is_decision"] = df.date.isin(grid)                       # 標記決策日
F20 = [c for c in df.columns if c.endswith("_L20")]          # 20 日版特徵欄位
df["eligible"] = df.valid & df[F20].notna().all(axis=1)      # 合格：t 日有成交且 15 項特徵齊全
df = df[df.date >= start - pd.Timedelta(days=0)]             # 只留研究期間
df.to_parquet(PROC/"samples.parquet", index=False)           # 存檔
pd.Series(grid).to_csv(TAB/"decision_dates.csv", index=False, header=["decision_date"])  # 存決策日

stk = df[~df.sid.isin(["0050", "00635U"])]                   # 只看 15 檔股票
cov = stk[stk.is_decision & stk.eligible].groupby("date").size()  # 每個決策日合格股票數
cov.to_csv(TAB/"coverage_eligible_names.csv", header=["n_eligible"])  # 存覆蓋率
print("features:", F20)                                      # 印出特徵名稱
print("decision dates:", len(grid), grid[0].date(), grid[-1].date())  # 印出決策日數量
print(cov.describe())                                        # 印出覆蓋率摘要
