# 12_correlation_checks.py — V7 要求、但 05／07 沒有輸出報告的相關性檢查
#   A. 特徵之間的相關（V7 2.4）：第 18 項換成開盤到開盤 GARCH 後，是否仍符合入選規則
#        R1 每週跨股票 IC 顯著（HAC；以 40 個候選的 Bonferroni 門檻 0.05/40 檢查，比 BH 更嚴，過了就一定過 BH）
#        R2 2017–2020 每年 IC 方向相同　R5 與其他 17 項的 Spearman |ρ| ≤ 0.9
#        只用篩選期 2016-11～2020 的每週決策日（標籤在 2021 前結束），與當初選特徵相同
#   B. Model 2 相關矩陣的數值檢查（V7 5.3）：每週的 R_shrink 是否對稱、對角線為 1、相關係數在 [−1, 1]、
#        特徵值非負、條件數；並記錄收縮強度 λ 與平均相關 ρ̄
#   C. 權重穩定性（V7 5.3「比較重點」）：20／60／120 日與不收縮，每週內部權重的變動量
import numpy as np, pandas as pd                             # 載入：數值、表格
import matplotlib.pyplot as plt                              # 載入：畫圖
from scipy.stats import norm                                 # 載入：常態分配（p 值）
from common import CFG, PROC, VTAB, VFIG, nw_mean_ci, setup_fonts  # 載入：設定、路徑、HAC、字型
from models import load                                      # 載入：合格樣本
from model2_v7 import corr_matrix, allocate                  # 載入：V7 Model 2 的相關矩陣與配置
setup_fonts(); OUT = VTAB / "correlation"; OUT.mkdir(exist_ok=True)
FE = list(CFG["features"]["v7"])                             # 18 項特徵
df = load()

# ---------- A. 特徵相關與第 18 項的入選規則 ----------
tr = df[df.is_decision_wk.fillna(False).astype(bool) & df.y.notna() & (df.date >= "2016-11-01") & (df.date <= "2020-12-31")
        & (df.label_end < pd.Timestamp("2021-01-01"))].copy()  # 篩選期（與當初選特徵相同）
C = tr[FE].corr(method="spearman")                           # 18 × 18 Spearman 相關
C.round(3).to_csv(OUT / "feature_corr_2016_2020.csv", encoding="utf-8-sig")
pairs = C.where(np.triu(np.ones_like(C, bool), 1)).stack().rename("rho").reset_index()  # 上三角的每一對
pairs["abs"] = pairs.rho.abs(); pairs = pairs.sort_values("abs", ascending=False)
pairs.head(15).round(3).to_csv(OUT / "feature_corr_top_pairs.csv", index=False, encoding="utf-8-sig")
print("相關最高的 10 對：\n", pairs.head(10).round(3).to_string(index=False))

def weekly_ic(c):                                            # 每週跨股票排名 IC
    g = tr.dropna(subset=[c, "y"]).groupby("date")
    s = g.apply(lambda x: x[c].rank().corr(x.y.rank()) if len(x) >= 3 else np.nan, include_groups=False).dropna()
    return s
rows = []
for c in FE:
    ic = weekly_ic(c); m, lo, hi, se = nw_mean_ci(ic.values); p = 2 * (1 - norm.cdf(abs(m / se)))
    yr = ic.groupby(ic.index.year).mean()
    rows.append({"特徵": c, "週數": len(ic), "平均 IC": m, "HAC p": p, "R1 通過（p ≤ 0.05/40）": p <= 0.05 / 40,
                 **{f"IC {y}": yr.get(y, np.nan) for y in (2017, 2018, 2019, 2020)},
                 "R2 通過（四年同號）": bool((np.sign(yr.loc[2017:2020]) == np.sign(m)).all()),
                 "與其他特徵最大 |ρ|": C[c].drop(c).abs().max(), "最相關的特徵": C[c].drop(c).abs().idxmax()})
R = pd.DataFrame(rows); R["R5 通過（|ρ| ≤ 0.9）"] = R["與其他特徵最大 |ρ|"] <= 0.9
R.round(4).to_csv(OUT / "feature_rules_2016_2020.csv", index=False, encoding="utf-8-sig")
print(R[R.特徵 == "garch_oo5"].round(4).T.to_string())
print("|ρ| > 0.9 的特徵對：", int((pairs["abs"] > 0.9).sum()))
v2 = pd.read_parquet(PROC / "features_v2.parquet")         # 舊的收盤 GARCH（若存在）比較
old = [c for c in v2.columns if c.startswith("garch") and "5" in c]
if old:
    x = tr[["date", "sid", "garch_oo5"]].merge(v2[["date", "sid"] + old], on=["date", "sid"])
    print("開盤到開盤 GARCH 與舊 GARCH 的 Spearman 相關：", x[["garch_oo5"] + old].corr(method="spearman").iloc[0, 1:].round(3).to_dict())

fig, ax = plt.subplots(figsize=(8.5, 7.5))                   # 熱圖
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FE)), FE, rotation=90, fontsize=8); ax.set_yticks(range(len(FE)), FE, fontsize=8)
for i in range(len(FE)):
    for j in range(len(FE)):
        if i != j and abs(C.values[i, j]) >= 0.7: ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(C.values[i, j]) > 0.85 else "black")
fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ（2016-11～2020 決策週）")
ax.set_title(f"18 項特徵的相關：最大 |ρ| = {pairs['abs'].max():.3f}（{pairs.iloc[0].level_0}–{pairs.iloc[0].level_1}），沒有超過 0.9", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(VFIG / "v7_fig10_feature_corr.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- B. 每週相關矩陣的數值檢查 ----------
panel = pd.read_parquet(PROC / "panel.parquet")
cal = pd.DatetimeIndex(sorted(panel[(panel.sid == "0050") & panel.valid].date.unique()))
O = panel[panel.valid].pivot(index="date", columns="sid", values="aopen").reindex(cal); RO = np.log(O / O.shift(1))
ml = pd.read_parquet(PROC / "model2_sigma_ml_v7.parquet")    # 07 存下的 Model 1 預測
dec = df[df.is_decision_wk.fillna(False).astype(bool) & (df.date >= "2017-01-01") & (df.date <= "2026-10-31")].merge(ml, on=["date", "sid"], how="left")
dec = dec[dec.tradable.fillna(False).astype(bool) & dec.rv20.notna() & dec.garch_oo5.notna()]  # 與 07 相同的 Uₜ
chk, stab = [], []
for L, shrink in ((20, True), (60, True), (120, True), (20, False)):
    prev = {"A": None, "B": None}
    for t, x in dec.sort_values("sid").groupby("date"):
        sids = list(x.sid); i = cal.get_loc(t); W = RO.iloc[max(0, i - L + 1): i + 1][sids]
        Rm, lam, nobs, _ = corr_matrix(W, shrink); ev = np.linalg.eigvalsh(Rm)
        Rs = np.corrcoef(W.dropna().values.T)                # 不收縮的樣本相關（比較用）
        evs = np.linalg.eigvalsh(Rs) if len(sids) > 1 else np.array([1.0])
        off = Rm[~np.eye(len(sids), dtype=bool)] if len(sids) > 1 else np.array([np.nan])
        chk.append({"L": L, "收縮": shrink, "date": t, "N": len(sids), "有效天數": nobs, "λ": lam, "平均相關 ρ̄": np.nanmean(off),
                    "對稱誤差": float(np.abs(Rm - Rm.T).max()), "對角線誤差": float(np.abs(np.diag(Rm) - 1).max()),
                    "最小相關": np.nanmin(off), "最大相關": np.nanmax(off), "最小特徵值": ev.min(), "條件數": ev.max() / ev.min(),
                    "樣本最小特徵值": evs.min(), "樣本條件數": evs.max() / max(evs.min(), 1e-12)})
        for s, col in (("A", "rv20"), ("B", "ml")):          # C. 權重穩定性（α = 0，內部權重 q）
            if s == "B" and x.ml.isna().any(): continue
            q = pd.Series(allocate(x[col].values.astype(float), Rm, 0.0, None)["q"], index=sids)
            if prev[s] is not None:
                d = q.reindex(q.index.union(prev[s].index), fill_value=0) - prev[s].reindex(q.index.union(prev[s].index), fill_value=0)
                stab.append({"L": L, "收縮": shrink, "策略": s, "date": t, "內部權重週變動（單邊）": 0.5 * d.abs().sum(), "名單變動": set(q.index) != set(prev[s].index)})
            prev[s] = q
K = pd.DataFrame(chk); K.to_csv(OUT / "corr_matrix_checks_weekly.csv", index=False, encoding="utf-8-sig")
K["year"] = pd.DatetimeIndex(K.date).year
summ = K.groupby(["L", "收縮"]).agg(週數=("date", "size"), 對稱誤差最大=("對稱誤差", "max"), 對角線誤差最大=("對角線誤差", "max"),
        最小相關=("最小相關", "min"), 最大相關=("最大相關", "max"), 最小特徵值=("最小特徵值", "min"), 條件數中位數=("條件數", "median"),
        條件數最大=("條件數", "max"), 樣本條件數中位數=("樣本條件數", "median"), 樣本最小特徵值=("樣本最小特徵值", "min"),
        λ中位數=("λ", "median"), ρ̄中位數=("平均相關 ρ̄", "median"), 有效天數最少=("有效天數", "min"))
summ.to_csv(OUT / "corr_matrix_checks_summary.csv", encoding="utf-8-sig"); print("相關矩陣檢查：\n", summ.T.to_string())
fail = K[(K.對稱誤差 > 1e-10) | (K.對角線誤差 > 1e-10) | (K.最小相關 < -1) | (K.最大相關 > 1) | (K.最小特徵值 < -1e-10)]
print("未通過數值檢查的週：", len(fail))
S = pd.DataFrame(stab); S["期間"] = np.where(pd.DatetimeIndex(S.date).year <= 2024, np.where(pd.DatetimeIndex(S.date).year >= 2022, "2022–2024", "2017–2021"), "2025–2026")
st = S[S.期間 != "2017–2021"].groupby(["期間", "策略", "L", "收縮"])["內部權重週變動（單邊）"].agg(["mean", "median"]).round(4)
st.to_csv(OUT / "weight_stability.csv", encoding="utf-8-sig"); print("權重穩定性：\n", st.to_string())

k20 = K[(K.L == 20) & K.收縮].set_index("date"); k60 = K[(K.L == 60) & K.收縮].set_index("date")  # 圖：ρ̄ 與 λ 隨時間
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
ax[0].plot(k60.index, k60["平均相關 ρ̄"], color="#888888", lw=1, label="60 日"); ax[0].plot(k20.index, k20["平均相關 ρ̄"], color="#2a6fdb", lw=1, label="20 日")
ax[0].set_ylabel("平均兩兩相關 ρ̄"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("股票間平均相關（上）與 Ledoit–Wolf 收縮強度 λ（下）", loc="left", fontsize=11)
ax[1].plot(k60.index, k60["λ"], color="#888888", lw=1, label="60 日"); ax[1].plot(k20.index, k20["λ"], color="#2a6fdb", lw=1, label="20 日")
ax[1].set_ylabel("收縮強度 λ"); ax[1].set_ylim(0, 1.02)
for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(color="#eeeeee")
fig.tight_layout(); fig.savefig(VFIG / "v7_fig11_corr_lambda.png", dpi=150, bbox_inches="tight"); plt.close(fig)
