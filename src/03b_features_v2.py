# 03b_features_v2.py — 計算候選特徵（28 項，含 GARCH、大盤 HMM 與行事曆），寫入 data/processed/features_v2.parquet
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC, TAB                            # 載入：設定與路徑
from features_v2 import stock_features, calendar_features, CANDIDATES  # 載入：第二版特徵公式
from garch import garch_panel                                # 載入：GARCH 走勢前進
from hmm_regime import market_obs, walk as hmm_walk         # 載入：大盤 HMM（濾波機率）

STK, BM = CFG["universe"]["stocks"], CFG["universe"]["passive_benchmark"]  # 股票與 0050
H = CFG["schedule"]["label_horizon_days"]                    # 標籤期間
panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])  # 清理後日資料
panel["avolume"] = (panel.volume / panel.adj).where(panel.valid)  # 還原成交量（無效日空值）
panel["amount"] = panel.amount.where(panel.valid)            # 無效日成交金額空值
b = panel[panel.sid == BM].set_index("date"); rm = np.log(b.aclose / b.aclose.shift(1))  # 0050 日報酬
cal = pd.DatetimeIndex(sorted(panel.date.unique()))          # 交易日曆

X = pd.concat([stock_features(g.set_index("date"), rm.reindex(g.set_index("date").index)).assign(sid=sid).reset_index()
               for sid, g in panel[panel.sid.isin(STK)].groupby("sid")])  # 各檔個股特徵
gf, fits = garch_panel(panel, STK)                           # GARCH（每季重估、封存期凍結）
X = X.merge(gf, on=["date", "sid"], how="left")              # 合併 GARCH
X = X.merge(pd.DataFrame({"date": rm.index, "mkt_rv20": np.sqrt(252 / 20 * (rm ** 2).rolling(20, min_periods=20).sum()).values}), on="date", how="left")  # 市場波動
X = X.merge(calendar_features(cal, H), on="date", how="left")  # 行事曆特徵
hs, hfits = hmm_walk(market_obs(panel))                      # 大盤壓力狀態機率（重估時程、封存凍結）
X = X.merge(hs.rename_axis("date").reset_index(), on="date", how="left")  # 合併（每天所有股票同值）
elig = pd.read_parquet(PROC / "samples.parquet", columns=["date", "sid", "eligible"])  # 合格標記（與 v1 相同的股票池規則）
X = X.merge(elig, on=["date", "sid"], how="left")            # 合併
X["rv20_rank"] = X[X.eligible.fillna(False)].groupby("date").rv20.rank(pct=True)  # 當日合格股票中的百分位
X = X.drop(columns="eligible")                               # 不重複存
X.to_parquet(PROC / "features_v2.parquet", index=False)      # 寫出全部 27 項候選
gf.to_parquet(PROC / "garch_features.parquet", index=False)  # 寫出 GARCH（notebooks 也會讀）
fits.to_csv(TAB / "garch_fits.csv", index=False)             # 寫出每次估計的參數
hfits.to_csv(TAB / "hmm_fits.csv", index=False)              # 寫出 HMM 每次估計
print(X.shape, "| 候選：", len(CANDIDATES), "項")            # 印出
