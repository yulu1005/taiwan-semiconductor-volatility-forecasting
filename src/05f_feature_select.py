# 05f_feature_select.py — 從「17 項 ∪ 30 項」中挑出 10–30 項（只用 2016-11～2020 篩選），再於驗證期比較一次
# 篩選規則（執行前寫定，與 notebooks/03b 同一套精神，但增量資訊改為必要條件）：
#   R0 完全相同的特徵只留一個（優先保留 17 項的名稱）
#   R1 每週跨股票 IC（HAC）經 Benjamini–Hochberg 5% 校正後顯著
#   R2 2017–2020 每一年的 IC 方向都與整體相同
#   R3 控制 rv20 後的偏 IC（排名殘差）HAC p < 0.05（rv20 為錨點，自動保留）
#   R4 市場層級特徵（同一天所有股票同值）改用時間序列：每週平均標籤 ~ 特徵 + 平均 rv20，特徵係數 HAC p < 0.05
#   R5 訓練期相關 |ρ| > 0.9 只留增量較強者
#   數量：多於 30 取增量最強的 30；少於 10 依序補入通過 R1–R2 但未過 R3 者（增量 t 大者優先）
import numpy as np, pandas as pd, statsmodels.api as sm      # 載入：數值、表格、迴歸（HAC）
from scipy.stats import norm                                 # 載入：常態分配（算 p 值）
from common import PROC, VTAB, VER, nw_mean_ci               # 載入：路徑、HAC 區間
from models import load, feat_cols                           # 載入：樣本、特徵名稱
from features_30 import F30                                  # 載入：30 項名稱
assert VER == "v3"                                           # 只用於 v3
DEC = "is_decision_wk"                                       # 每週決策日
df = load().merge(pd.read_parquet(PROC / "features_30.parquet"), on=["date", "sid"], how="left")  # 合格樣本＋30 項
S17 = feat_cols(20, "v3")                                    # 現行 17 項
tr = df[df[DEC].fillna(False).astype(bool) & df.y.notna() & (df.date <= "2020-12-31") & (df.label_end < pd.Timestamp("2021-01-01"))].copy()  # 篩選期：只到 2020，標籤也在 2020 結束
print("篩選期週數", tr.date.nunique(), "筆數", len(tr))      # 印出

# ---------- R0：完全相同的只留一個 ----------
cand = list(S17)                                             # 先放 17 項
for c in F30:                                                # 再逐一加入 30 項
    dup = [s for s in cand if tr[[c, s]].corr(method="spearman").iloc[0, 1] > 0.999]  # 與已在名單的特徵完全同序
    if dup: print(f"R0 去重：{c} = {dup[0]}"); continue      # 相同就跳過
    cand.append(c)                                           # 不同才加入
mkt = [c for c in cand if tr.groupby("date")[c].nunique().max() <= 1]  # 市場層級（同一天同值）
xs = [c for c in cand if c not in mkt]                       # 個股層級
print("候選", len(cand), "項（個股層級", len(xs), "、市場層級", len(mkt), "）", mkt)  # 印出

# ---------- R1–R3：個股層級特徵 ----------
def rk(s): return s.rank()                                   # 定義：排名
def resid(a, b): A, B = rk(a), rk(b); bb = B - B.mean(); return A - A.mean() - (bb @ (A - A.mean()) / (bb @ bb)) * bb  # 定義：a 的排名對 b 的排名做線性殘差
rows = []                                                    # 收集
for c in xs:                                                 # 逐一特徵
    ic, pic, yrs = [], [], []                                # 每週 IC、偏 IC、年份
    for d, g in tr.groupby("date"):                          # 每週
        g = g[list(dict.fromkeys([c, "y", "rv20"]))].dropna()                     # 去缺值
        if len(g) < 5: continue                              # 少於 5 檔不算
        ic.append(g[c].rank().corr(g.y.rank())); yrs.append(d.year)  # 跨股票排名相關
        pic.append(np.nan if c == "rv20" else np.corrcoef(resid(g[c], g.rv20), resid(g.y, g.rv20))[0, 1])  # 控制 rv20 的偏 IC
    ic, pic = pd.Series(ic, index=yrs), pd.Series(pic, index=yrs)  # 轉成序列
    m, lo, hi, se = nw_mean_ci(ic.values); pm, plo, phi, pse = nw_mean_ci(pic.values) if c != "rv20" else (np.nan,) * 4  # HAC
    by = ic.groupby(level=0).mean().loc[2017:2020]           # 2017–2020 各年平均 IC
    rows.append({"feature": c, "from": "17項" if c in S17 else "30項", "ic": m, "p": 2 * norm.sf(abs(m / se)),
                 "sign_ok": bool((np.sign(by) == np.sign(m)).all()), "inc_ic": pm,
                 "inc_t": pm / pse if c != "rv20" else np.inf, "inc_p": 2 * norm.sf(abs(pm / pse)) if c != "rv20" else 0.0})  # 記錄
T = pd.DataFrame(rows)                                       # 轉表
o = T.p.rank(method="first"); n = len(T); T["bh"] = T.p <= o / n * 0.05  # BH：第 k 小的 p ≤ k/n × 5%
T["bh"] = T.p <= T.loc[T.bh, "p"].max() if T.bh.any() else False  # BH 是「最大的通過者以下全部通過」
T["level"] = "個股"                                          # 標記

# ---------- R4：市場層級特徵（時間序列） ----------
w = tr.groupby("date").agg(y=("y", "mean"), rv20=("rv20", "mean"), **{c: (c, "first") for c in mkt})  # 每週平均
for c in mkt:                                                # 逐一市場特徵
    X = sm.add_constant(w[[c, "rv20"]].dropna()); yv = w.loc[X.index, "y"]  # 自變數
    fit = sm.OLS(yv, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})  # HAC 迴歸
    T.loc[len(T)] = {"feature": c, "from": "17項" if c in S17 else "30項", "ic": np.nan, "p": fit.pvalues[c], "sign_ok": True,
                     "inc_ic": fit.params[c], "inc_t": fit.tvalues[c], "inc_p": fit.pvalues[c], "bh": fit.pvalues[c] < 0.05, "level": "市場"}  # 記錄

# ---------- 套用規則 ----------
T["pass_R1R2"] = T.bh & T.sign_ok                            # 顯著且方向穩定
T["pass_R3"] = T.pass_R1R2 & (T.inc_p < 0.05)                # 還要有增量
keep = T[T.pass_R3].assign(a=lambda d: d.inc_t.abs()).sort_values("a", ascending=False).feature.tolist()  # 依增量強度排序
C = tr[keep].corr(method="spearman").abs()                   # 訓練期相關
sel = []                                                     # 依序挑選
for c in keep:                                               # 強的先選
    if any(C.loc[c, s] > 0.9 for s in sel): T.loc[T.feature == c, "note"] = "與 " + max(sel, key=lambda s: C.loc[c, s]) + " 重複"; continue  # R5
    sel.append(c)                                            # 入選
if len(sel) > 30: sel = sel[:30]                             # 最多 30
if len(sel) < 10:                                            # 少於 10 時補入
    extra = T[T.pass_R1R2 & ~T.feature.isin(sel)].assign(a=lambda d: d.inc_t.abs()).sort_values("a", ascending=False).feature  # 候補
    for c in extra:
        if len(sel) >= 10: break
        if all(abs(tr[[c, s]].corr(method="spearman").iloc[0, 1]) <= 0.9 for s in sel): sel.append(c)  # 不重複才補
T["selected"] = T.feature.isin(sel)                          # 標記
T.sort_values(["selected", "inc_t"], ascending=False).round(4).to_csv(VTAB / "feature_select_screen.csv", index=False)  # 存檔
pd.Series(sel, name="feature").to_csv(VTAB / "feature_select_list.csv", index=False)  # 存入選名單
print(T.sort_values(["selected", "inc_t"], ascending=False)[["feature", "from", "level", "ic", "bh", "sign_ok", "inc_ic", "inc_p", "selected", "note"]].round(3).to_string(index=False))  # 印出
print("入選", len(sel), "項：", sel)                         # 印出

# ---------- 標準版（15–20 項）：與 v2 篩選相同的門檻——顯著（R1）且方向穩定（R2）即可用；不必有增量 ----------
#   排序：先放嚴格版（通過 R3），再依增量 t 值由大到小補入通過 R1–R2 者；|ρ| > 0.9 跳過；上限 20、下限 15
std = list(sel)                                              # 從嚴格版開始
pool = T[T.pass_R1R2 & ~T.feature.isin(std) & T.inc_t.notna()].assign(a=lambda d: d.inc_t.abs()).sort_values("a", ascending=False).feature  # 候補（增量強者優先）
for c in pool:                                               # 逐一補入
    if len(std) >= 20: break                                 # 上限 20
    r = tr[[c] + std].corr(method="spearman").abs().loc[c, std]  # 與已選特徵的相關
    if r.max() > 0.9: print(f"標準版跳過 {c}（與 {r.idxmax()} 相關 {r.max():.2f}）"); continue  # 重複
    std.append(c)                                            # 入選
assert len(std) >= 15, f"只有 {len(std)} 項，未達 15"        # 下限檢查
T["selected_std"] = T.feature.isin(std)                      # 標記
T.sort_values(["selected", "selected_std", "inc_t"], ascending=False).round(4).to_csv(VTAB / "feature_select_screen.csv", index=False)  # 重新存檔
pd.Series(std, name="feature").to_csv(VTAB / "feature_select_list_std.csv", index=False)  # 存標準版名單
print("標準版", len(std), "項：", std)                         # 印出
print(tr[std].corr(method="spearman").abs().where(lambda m: m < 0.999).max().round(2).to_string())  # 每項與其他入選特徵的最大相關
