# 05d_baselines_v3.py — 波動率模型基準（HV20、EWMA、GARCH、HAR）vs 主模型 Lasso：整體、各年度、各股票（驗證期，不交易）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC, VTAB, VER, vproc, nw_mean_ci   # 載入：設定、路徑、HAC 區間
from models import load, split, make_model, FLOOR            # 載入：樣本、切分、模型、下限
assert VER in ("v3", "v4"), "請在 setup.yaml 設 features.active: v3 或 v4"   # v3／v4 共用
DEC = "is_decision_wk"                                       # 每週最後交易日決策
df = load()                                                  # 合格樣本
panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])  # 日資料
r = np.log(panel.aclose / panel.groupby("sid").aclose.shift())  # 每日收盤對數報酬
panel["ewma"] = np.sqrt(252 * (r ** 2).groupby(panel.sid).transform(lambda s: s.ewm(alpha=0.06, adjust=False, min_periods=20).mean()))  # EWMA（λ=0.94，RiskMetrics），年化
df = df.merge(panel[["date", "sid", "ewma"]], on=["date", "sid"], how="left")  # 合併
P = pd.read_parquet(vproc("oos_validation.parquet"))         # 05_validate_v3 的樣本外預測（依版本）
PRIMARY = pd.read_csv(VTAB / "selected_configs.csv", index_col=0).config["primary"]  # 主模型名稱
P = P[(P.fold != "calib2021") & P.config.isin(["hv", PRIMARY])]  # 只取 HV20 與主模型
rows = []                                                    # 收集基準預測
for k, fd in enumerate(CFG["protocol"]["folds"]):            # 三個驗證年度
    tr, te = split(df, fd["train"][1], *fd["valid"], dec_col=DEC)  # 訓練與每週決策日
    har = make_model("ridge", {"alpha": 1e-4}).fit(tr[["rv1", "rv5", "rv20"]], tr.y).predict(te[["rv1", "rv5", "rv20"]])  # HAR：日、週、月波動的線性迴歸
    for name, p in [("ewma", te.ewma.values), ("garch", te.garch_fc5.values), ("har", har)]:  # 三個基準
        p = np.asarray(p, float); p = np.where(np.isnan(p), te.rv20.values, p)  # GARCH 暖身期缺值用 20 日波動代替
        rows.append(te[["date", "sid", "y", "tradable"]].assign(p=np.maximum(p, FLOOR), config=name, fold=f"fold{k+1}"))  # 記錄
B = pd.concat([P[["date", "sid", "y", "tradable", "p", "config", "fold"]]] + rows)  # 全部合併
B = B[B.tradable.fillna(False).astype(bool)]                 # 只評估可交易股票
B["ql"] = (B.y / B.p) ** 2 - np.log((B.y / B.p) ** 2) - 1    # 每筆 QLIKE
def ic(t): return t.groupby("date").apply(lambda g: g.p.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).mean()  # 定義：每週排名相關的平均
out = B.groupby(["config", "fold"]).apply(lambda t: pd.Series({"QLIKE": t.ql.mean(), "MAE": (t.p - t.y).abs().mean(),
                                                               "bias": t.p.mean() - t.y.mean(), "IC": ic(t)}), include_groups=False)  # 各模型各年度指標
out.round(4).to_csv(VTAB / "baselines_by_year.csv")         # 存檔
w = B.pivot_table(index=["date", "sid"], columns="config", values="ql")  # 每筆 QLIKE（寬表）
pc = []                                                      # 成對比較
for b in ["hv", "ewma", "garch", "har"]:                     # 每個基準
    d = (w[PRIMARY] - w[b]).groupby(level=0).mean().dropna()  # 每週平均差異（負＝主模型較好）
    m, lo, hi, _ = nw_mean_ci(d.values); pc.append({"baseline": b, "diff": m, "lo95": lo, "hi95": hi})  # HAC 區間
ps = B.groupby(["config", "sid"]).ql.mean().unstack(0)       # 每檔股票的 QLIKE
for x in pc: x["stocks_primary_better"] = int((ps[PRIMARY] < ps[x["baseline"]]).sum()); x["stocks"] = int(ps[x["baseline"]].notna().sum())  # 主模型較好的股票數
pd.DataFrame(pc).round(4).to_csv(VTAB / "baselines_paired.csv", index=False); ps.round(4).to_csv(VTAB / "baselines_by_stock.csv")  # 存檔
print(out.groupby("config").mean().sort_values("QLIKE").round(4)); print(pd.DataFrame(pc).round(4))  # 印出
