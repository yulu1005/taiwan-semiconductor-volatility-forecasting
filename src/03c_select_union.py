# 03c_select_union.py — 從「現行 17 項 ∪ 組員 30 項」挑出好特徵（篩選只用 2016-11～2020；驗證期只做一次確認）
# 規則在執行前寫定：
#   第一關（2016–2020，每週決策日）：
#     R1 錨點：rv20 保留（HV20 基準）
#     R2 個股層級：BH 校正後顯著（q<0.05）且 2017–2020 每年同方向 ≥ 75%，而且扣掉 rv20 後的增量 IC 顯著（p<0.05）
#        （比 v2 嚴格：v2 是「或」，這裡改成「且」）
#     R3 市場層級（同一天所有股票同值）：控制平均 rv20 後的時間序列偏相關 p<0.05
#     R4 冗餘：|ρ|>0.9 的同一群只留 |IC| 最高者（rv20 所在的群以 rv20 代表）
#   第二關（仍只用 2016–2020）：Lasso 穩定度——留一年、與 20 次週區塊半抽樣，共 25 次；係數非 0 的比例 ≥ 70% 才保留
#   第三關（驗證期 2022–2024，只做一次）：與 17 項、30 項比較；新組「沒有顯著較差」就採用（較精簡、只用訓練期選出）
import sys, numpy as np, pandas as pd                        # 載入：系統、數值、表格
from scipy import stats                                      # 載入：t 分配
from statsmodels.stats.multitest import multipletests        # 載入：BH 校正
from common import ROOT, PROC, VTAB, CFG                     # 載入：路徑與設定
from models import load, make_model, feat_cols               # 載入：樣本與模型
from features_30 import F30                                  # 載入：30 項名稱
sys.path.insert(0, str(ROOT / "notebooks")); from nb_utils import ic_by_date, hac_stats  # 載入：IC 與 HAC 工具
END = pd.Timestamp("2020-12-31"); RNG = np.random.default_rng(0)  # 篩選終點；固定亂數種子
df = load().merge(pd.read_parquet(PROC / "features_30.parquet"), on=["date", "sid"], how="left")  # 合格樣本＋30 項
S17 = feat_cols(20, "v3")                                    # 現行 17 項
S = df[(df.date >= CFG["data"]["study_start"]) & df.y.notna() & (df.label_end <= END)].copy()  # 篩選資料（答案在 2020 年底前揭曉）
D = S[S.is_decision_wk.fillna(False).astype(bool)]           # 每週決策日（標籤不重疊）

# ---------- 0. 候選池：去掉完全相同的特徵（排名相關 ≥ 0.999 視為同一個，保留 17 項的名稱） ----------
pool = list(S17)                                             # 先放 17 項
cm0 = D[S17 + F30].rank().corr()                             # 決策日上的排名相關
dup = {}                                                     # 記錄重複
for f in F30:                                                # 逐一 30 項
    same = [g for g in pool if abs(cm0.loc[f, g]) >= 0.999]  # 與已在池中的特徵完全相同
    if same: dup[f] = same[0]                                # 重複就不加入
    else: pool.append(f)                                     # 新特徵加入
DL = [f for f in pool if D.groupby("date")[f].nunique().max() <= 1]  # 市場層級：同一天所有股票同值
SL = [f for f in pool if f not in DL]                        # 個股層級
print(f"候選池 {len(pool)} 項（去重 {len(dup)} 項：{dup}）；個股層級 {len(SL)}、市場層級 {len(DL)}")  # 印出

# ---------- 1. 第一關：統計篩選 ----------
def resid_rank(t, f):                                        # 定義：每天把特徵排名對 rv20 排名回歸，取殘差
    a, b = t[f].rank(), t.rv20.rank(); v = b.var()           # 兩個排名
    return a - (np.cov(a, b)[0, 1] / v if v > 0 else 0) * b  # 殘差
R = []                                                       # 收集
for f in SL:                                                 # 個股層級
    icd = ic_by_date(D, f, min_n=8); h = hac_stats(icd, 1)   # 每週不重疊 IC
    yr = icd.groupby(icd.index.year).mean(); yr = yr[yr.index >= 2017]  # 各年 IC
    if f == "rv20": inc = {"ic_mean": np.nan, "hac_p": np.nan}  # 錨點不算增量
    else:
        t = D[["date", f, "rv20", "y"]].dropna(); t["res"] = t.groupby("date", group_keys=False).apply(lambda x: resid_rank(x, f), include_groups=False)  # 扣掉 rv20
        inc = hac_stats(ic_by_date(t, "res", min_n=8), 1)    # 增量 IC
    R.append({"feature": f, "ic": h["ic_mean"], "p": h["hac_p"], "sign_years": (np.sign(yr) == np.sign(h["ic_mean"])).mean(),
              "inc_ic": inc["ic_mean"], "inc_p": inc["hac_p"]})  # 記錄
R = pd.DataFrame(R).set_index("feature"); R["bh"] = multipletests(R.p, alpha=0.05, method="fdr_bh")[0]  # BH 校正
for f in DL:                                                 # 市場層級
    m = D.groupby("date")[[f, "y", "rv20"]].mean().dropna()  # 每週平均
    e1, e2 = np.polyfit(m.rv20, m[f], 1), np.polyfit(m.rv20, m.y, 1)  # 控制平均 rv20
    pr = np.corrcoef(m[f] - np.polyval(e1, m.rv20), m.y - np.polyval(e2, m.rv20))[0, 1]  # 偏相關
    R.loc[f, ["ic", "p"]] = [pr, 2 * (1 - stats.t.cdf(abs(pr) * np.sqrt((len(m) - 3) / (1 - pr ** 2)), len(m) - 3))]  # 記錄
R["level"] = np.where(R.index.isin(DL), "市場", "個股")      # 層級
R["pass1"] = (R.index == "rv20") | ((R.level == "個股") & R.bh.fillna(False).astype(bool) & (R.sign_years >= 0.75) & (R.inc_p < 0.05)) | ((R.level == "市場") & (R.p < 0.05))  # R1–R3
cm = D[SL].rank().corr().abs(); red = {}                     # 冗餘
cand = [f for f in R[R.pass1 & (R.level == "個股")].ic.abs().sort_values(ascending=False).index]  # 通過者依 |IC| 排序
kept1 = ["rv20"] if "rv20" in cand else []                   # 錨點先放
for f in cand:                                               # 依序檢查
    if f in kept1: continue                                  # 已放
    hit = [g for g in kept1 if cm.loc[f, g] > 0.9]           # 與已保留者重複
    if hit: red[f] = hit[0]                                  # 記錄被誰取代
    else: kept1.append(f)                                    # 保留
kept1 += [f for f in DL if R.loc[f, "pass1"]]                # 市場層級通過者
R["redundant_with"] = pd.Series(red); R["stage1"] = R.index.isin(kept1)  # 寫入
print("第一關保留", len(kept1), "項：", kept1)                # 印出

# ---------- 2. 第二關：Lasso 穩定度（仍只用 2016–2020） ----------
years = sorted(D.date.dt.year.unique()); weeks = np.array(sorted(D.date.unique()))  # 年份與週
fits = []                                                    # 收集
for y0 in years:                                             # 留一年
    tr = S[S.date.dt.year != y0]; fits.append(make_model("lasso", {"alpha_frac": 0.1}).fit(tr[kept1], tr.y).m.coef_)  # 訓練
blocks = np.array_split(weeks, len(weeks) // 13)             # 約一季一個區塊
for _ in range(20):                                          # 20 次半抽樣
    idx = RNG.choice(len(blocks), len(blocks) // 2, replace=False)  # 抽一半的區塊（每區塊約一季）
    mask = np.zeros(len(S), bool)                            # 準備篩選
    for i in idx: mask |= S.date.between(blocks[i].min(), blocks[i].max()).values  # 區塊期間內的所有交易日
    tr = S[mask]                                             # 抽到的區塊（和留一年一樣用所有交易日）
    fits.append(make_model("lasso", {"alpha_frac": 0.1}).fit(tr[kept1], tr.y).m.coef_)  # 訓練
F = pd.DataFrame(fits, columns=kept1)                        # 25 次係數
stab = (F.abs() > 1e-6).mean(); R.loc[kept1, "stability"] = stab  # 非 0 比例
final = [f for f in kept1 if stab[f] >= 0.70]                # 第二關保留
R["final"] = R.index.isin(final)                             # 寫入
R.round(4).to_csv(VTAB / "union_selection.csv"); pd.Series(final, name="feature").to_csv(VTAB / "union_selected_features.csv", index=False)  # 存檔
print(R.sort_values(["final", "stage1", "ic"], ascending=[False, False, False]).round(3).to_string())  # 印出
print("最終保留", len(final), "項：", final)                   # 印出
