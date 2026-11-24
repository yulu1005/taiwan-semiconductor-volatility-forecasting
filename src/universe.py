# universe.py — v3 時點正確的可交易股票池與每週決策日（範本 01 B.2；書面提案：週五收盤決策）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC                                 # 載入：設定與路徑

U = CFG["v3"]["universe"]["liquidity_gate"]                  # 流動性門檻設定
STK = CFG["universe"]["stocks"]                              # 15 檔
panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])  # 日資料
panel = panel[panel.sid.isin(STK)].copy()                    # 只看股票
amt = panel.amount.where(panel.valid)                        # 有效日成交金額（無效日空值）
g = amt.groupby(panel.sid)                                   # 依股票分組
panel["adv250"] = g.transform(lambda s: s.rolling(U["window_sessions"], min_periods=U["min_valid_days"]).mean())  # 過去 250 日平均成交金額（含決策日當天，收盤後已知）
panel["n_valid250"] = g.transform(lambda s: s.notna().rolling(U["window_sessions"], min_periods=1).sum())  # 窗內有效日數
cal = pd.DatetimeIndex(sorted(panel.date.unique()))          # 交易日曆
wk = pd.Series(cal, index=cal).groupby(cal.to_period("W-SUN")).max()  # 每週最後一個交易日
wk = wk[wk >= pd.Timestamp(CFG["data"]["study_start"])]      # 研究期間內
smp = pd.read_parquet(PROC / "samples.parquet", columns=["date", "sid", "eligible"])  # 既有合格標記
u = panel[["date", "sid", "adv250", "n_valid250"]].merge(smp, on=["date", "sid"], how="left")  # 合併
u["tradable"] = u.eligible.fillna(False) & (u.adv250 >= U["floor_twd"]) & (u.n_valid250 >= U["min_valid_days"])  # 可交易：合格且流動性達標
u["is_decision_wk"] = u.date.isin(set(wk))                   # 每週決策日
u.drop(columns="eligible").to_parquet(PROC / "universe_v3.parquet", index=False)  # 寫出
d = u[u.is_decision_wk]                                      # 決策日
cnt = d.groupby("date").tradable.sum()                       # 每週可交易檔數
print(f"每週決策日 {d.date.nunique()} 個；可交易檔數 中位數 {cnt.median():.0f}、最少 {cnt.min()}、低於 5 檔的週數 {(cnt < 5).sum()}")  # 印出
print(cnt.groupby(cnt.index.year).agg(["min", "median", "max"]).T.to_string())  # 各年分布
print("各股可交易比例：", d.groupby("sid").tradable.mean().round(2).to_dict())  # 各股
