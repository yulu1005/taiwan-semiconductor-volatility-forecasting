# 04_triage.py — 特徵篩選五道關卡（只用第一段訓練期 2016-11～2021-12，不碰驗證與封存資料）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from scipy import stats                                      # 載入：統計檢定
from common import ROOT, CFG, PROC, TAB, nw_mean_ci          # 載入：路徑、設定、HAC 信賴區間

df = pd.read_parquet(PROC/"samples.parquet")                 # 讀樣本
STK = CFG["universe"]["stocks"]                              # 15 檔股票
end = pd.Timestamp(CFG["protocol"]["folds"][0]["train"][1])  # 第一段訓練期終點
d = df[df.sid.isin(STK) & df.is_decision & df.eligible & (df.label_end <= end)].copy()  # 決策日、合格、標籤在訓練期內結束
F = [c for c in df.columns if c.endswith("_L20")]            # 15 項特徵
NAMES = {"hv5": "5日波動", "hv10": "10日波動", "hv20": "20日波動", "ret1": "1日報酬", "ret5": "5日報酬", "ret20": "20日報酬",
         "hl": "高低價差比", "oc": "開收價差比", "ma5_dist": "5日均線距離", "ma20_dist": "20日均線距離", "vol_ratio": "成交量比",
         "vol_cv": "量的變動", "amt_ratio": "成交金額比", "rsi": "RSI14", "atr_p": "ATR14/股價"}  # 中文名稱
ROLE = {k: ("狀態" if k.startswith(("hv", "hl", "atr")) else "訊號") for k in NAMES}  # 角色：風險狀態或訊號

all_rows = df[df.sid.isin(STK) & df.eligible].sort_values(["sid", "date"])  # 所有交易日資料（做安慰劑位移用）
for f in F:                                                  # 逐一特徵
    all_rows[f + "_lag10"] = all_rows.groupby("sid")[f].shift(10)  # 位移 10 個交易日的舊值（時間安慰劑）
d = d.merge(all_rows[["date", "sid"] + [f + "_lag10" for f in F]], on=["date", "sid"], how="left")  # 併回決策日樣本

def ic_series(x, y):                                         # 定義：每個決策日的橫斷面 Spearman IC
    t = pd.DataFrame({"x": x, "y": y, "date": d.date}).dropna()  # 合併並去空值
    return t.groupby("date").apply(lambda g: g.x.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).dropna()  # 至少 5 檔才算

rows = []                                                    # 收集結果
mkt = d.groupby("date").y.mean()                             # 每日平均實現波動（時間序列用）
vol_regime = pd.qcut(mkt.rank(method="first"), 3, labels=["低波動", "中波動", "高波動"])  # 市場波動三分位（訓練期內）
for f in F:                                                  # 逐一特徵
    k = f.replace("_L20", "")                                # 去掉後綴
    ic = ic_series(d[f], d.y)                                # 主 IC 序列
    m, lo, hi, se = nw_mean_ci(ic.values)                    # HAC 平均與 95% 區間
    p = 2 * (1 - stats.norm.cdf(abs(m / se)))                # 雙尾 p 值
    yr = ic.groupby(ic.index.year).mean()                    # 各年度平均 IC（年度 fold）
    same = (np.sign(yr) == np.sign(m)).mean()                # 與整體同號的年度比例
    plc = ic_series(d[f + "_lag10"], d.y).mean()            # 位移 10 日的安慰劑 IC
    ic10, ic20 = ic_series(d[f], d.y10).mean(), ic_series(d[f], d.y20).mean()  # 10 日、20 日標籤的 IC（衰減）
    q = d.assign(q=d.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)))  # 每日分 5 組
    qm = q.groupby("q").y.mean()                             # 各組平均實現波動
    mono = stats.spearmanr(qm.index, qm.values)[0]           # 單調性（±1 表示完全單調）
    fr = d.pivot(index="date", columns="sid", values=f).rank(axis=1)  # 每日特徵排名
    auto = fr.corrwith(fr.shift(1), axis=1).mean()           # 相鄰決策日排名相關（越高越穩定、換手越低）
    ts = np.corrcoef(d.groupby("date")[f].mean().loc[mkt.index], mkt)[0, 1]  # 時間序列：平均特徵 vs 平均實現波動
    reg = ic.groupby(vol_regime.reindex(ic.index)).mean()    # 不同市場波動狀態下的 IC
    rows.append({"feature": k, "中文": NAMES[k], "role": ROLE[k], "IC_mean": m, "IC_lo95": lo, "IC_hi95": hi, "p_value": p,
                 "ICIR": ic.mean() / ic.std(), "share_years_same_sign": same, "IC_placebo_lag10": plc,
                 "IC_h10": ic10, "IC_h20": ic20, "quintile_monotonicity": mono, "rank_autocorr": auto,
                 "TS_corr_mean_level": ts, "IC_lowvol": reg.get("低波動"), "IC_midvol": reg.get("中波動"), "IC_highvol": reg.get("高波動"),
                 "n_dates": len(ic)})                        # 收集所有指標
t = pd.DataFrame(rows)                                       # 轉成表格
o = np.argsort(t.p_value.values); n = len(t)                 # Benjamini–Hochberg：依 p 值排序
bh = np.zeros(n, bool); crit = (np.arange(1, n + 1) / n) * 0.05  # 各名次的門檻
passed = t.p_value.values[o] <= crit                         # 是否低於門檻
if passed.any(): bh[o[: np.max(np.where(passed)[0]) + 1]] = True  # 最大通過名次之前全部通過
t["BH_sig_5pct"] = bh                                        # 多重檢定校正後是否顯著
def decide(r):                                               # 定義：PROCEED / REVISE / STOP 規則（事先寫定）
    if r.BH_sig_5pct and r.share_years_same_sign >= 0.8 and abs(r.quintile_monotonicity) >= 0.9: return "PROCEED"  # 顯著、穩定、單調
    if r.BH_sig_5pct or r.share_years_same_sign >= 0.6: return "REVISE"   # 部分證據
    return "STOP"                                            # 沒有證據
t["decision"] = t.apply(decide, axis=1)                      # 套用規則
t["gate1_placebo_kills"] = (t.IC_placebo_lag10.abs() < 0.8 * t.IC_mean.abs())  # 位移後 IC 明顯下降＝資訊具時效
t.round(4).to_csv(TAB/"triage_table.csv", index=False, encoding="utf-8-sig")  # 存檔

cm = d[F].rank().corr()                                      # 特徵間 Spearman 相關
cm.index = cm.columns = [c.replace("_L20", "") for c in F]   # 簡化名稱
cm.round(3).to_csv(TAB/"feature_correlation.csv", encoding="utf-8-sig")  # 存相關矩陣
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm) for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > 0.8]  # 高度相關配對
pd.DataFrame(pairs, columns=["feat_a", "feat_b", "spearman"]).to_csv(TAB/"feature_high_corr_pairs.csv", index=False)  # 存檔
eff = pd.DataFrame({"nominal_stock_dates": [len(d)], "decision_dates": [d.date.nunique()],
                    "note": ["決策日間隔 5 日、標籤 5 日，時間上不重疊；同日 12–15 檔高度相關，有效樣本約等於決策日數"]})  # 有效樣本說明
eff.to_csv(TAB/"effective_sample.csv", index=False, encoding="utf-8-sig")  # 存檔
print(t[["feature", "IC_mean", "IC_lo95", "IC_hi95", "share_years_same_sign", "IC_placebo_lag10", "quintile_monotonicity", "TS_corr_mean_level", "BH_sig_5pct", "decision"]].round(3).to_string())  # 印出
print(pairs); print(eff)                                     # 印出高相關配對與有效樣本
