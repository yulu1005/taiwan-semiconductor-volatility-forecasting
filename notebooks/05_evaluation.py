# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 第五組：哪些特徵真的帶有未來波動的資訊？
#
# 03 建了 15 項財務特徵，04 建了 2 項 GARCH 特徵。本 notebook 一次只看一項特徵，回答：它對未來 5 日波動的
# **排名**有沒有資訊？做法跟課程範本相同：
# - 每個交易日把股票依特徵排名、依未來波動排名，算兩個排名的相關（IC），得到一條每日序列。
# - 標準誤用 Newey–West，頻寬＝標籤長度（相鄰兩天的標籤共用 4 個報酬）。
# - 一次檢定 17 項會有運氣成分，用 Benjamini–Hochberg 控制錯誤發現率。
# - 依三個驗證年度分開看方向是否一致。
# - 最後給每項特徵 PROCEED／REVISE／STOP，寫進 triage ledger。
#
# 評估面板是**三個驗證年度的聯集**（2022–2024，和模型被評分的資料相同），並依**標籤結束日**封住封存期。
# 參考：ML4T 第 7 章 7.3–7.4、課程範本 `etfs/05_evaluation`。

# %%
from nb_utils import *                                       # 共用工具
from scipy.stats import spearmanr                            # Spearman 相關
from statsmodels.stats.multitest import multipletests        # BH 多重檢定
style()                                                      # 圖表樣式
MIN_CS, IC_THR, MIN_SIGN, FDR, MIN_COV, MAX_STALE, RED_CUT = 10, 0.01, 0.6, 0.05, 0.70, 0.50, 0.7  # 範本的判斷門檻（事先寫定）
smp = pd.read_parquet(PROC / "samples.parquet")              # 標籤與財務特徵
smp = smp[smp.sid.isin(STK) & smp.eligible].merge(pd.read_parquet(PROC / "garch_features.parquet"), on=["date", "sid"], how="left")  # 加上 GARCH
NM = {"hv5": "Volatility_5", "hv10": "Volatility_10", "hv20": "Volatility_20", "ret1": "Return_1", "ret5": "Return_5", "ret20": "Return_20",
      "hl": "High_Low_Range", "oc": "Close_Open_Range", "ma5_dist": "MA_Gap_5", "ma20_dist": "MA_Gap_20", "vol_ratio": "Volume_Ratio",
      "vol_cv": "Volume_Volatility", "amt_ratio": "Turnover", "rsi": "RSI_14", "atr_p": "ATR_14", "garch_fc5": "GARCH_5d", "garch_vol1": "GARCH_1d"}  # 文件名稱
FAM = {"Volatility_5": "波動", "Volatility_10": "波動", "Volatility_20": "波動", "High_Low_Range": "區間", "ATR_14": "區間", "Close_Open_Range": "區間",
       "Return_1": "報酬／趨勢", "Return_5": "報酬／趨勢", "Return_20": "報酬／趨勢", "MA_Gap_5": "報酬／趨勢", "MA_Gap_20": "報酬／趨勢", "RSI_14": "報酬／趨勢",
       "Volume_Ratio": "量", "Volume_Volatility": "量", "Turnover": "量", "GARCH_5d": "模型型（GARCH）", "GARCH_1d": "模型型（GARCH）"}  # 特徵家族
smp = smp.rename(columns={c: NM[c.replace("_L20", "")] for c in smp.columns if c.replace("_L20", "") in NM and (c.endswith("_L20") or c.startswith("garch"))})  # 換成文件名稱
FEATS = list(FAM)                                            # 候選特徵

# %% [markdown]
# ## A. 評估面板：驗證年度聯集，依標籤結束日封住封存期

# %%
inval = np.zeros(len(smp), bool)                             # 是否在任一驗證窗內
for fd in FOLDS: inval |= ((smp.date >= fd["valid"][0]) & (smp.date <= fd["valid"][1])).values  # 聯集
before = int((inval & smp.y.notna()).sum())                  # 封之前的列數
ev = smp[inval & smp.y.notna() & (smp.label_end < HOLDOUT_START)].copy()  # 封住：標籤必須在封存期前結束
n_dates = ev.date.nunique()                                  # 日期數
print(f"依標籤結束日封住封存期：{before:,} → {len(ev):,} 列（最後觀察日 {ev.date.max().date()}）；"
      f"{ev.sid.nunique()} 檔 × {n_dates} 個交易日；候選特徵 {len(FEATS)} 項")  # 印出
print(f"有效樣本：{n_dates} 個每日 IC，重疊調整後約 {n_dates // H} 個不重疊區塊")  # 印出

# %% [markdown]
# ## B. 欄位能不能用：覆蓋率與陳舊度
# 覆蓋率＝有值的比例（門檻 70%）；陳舊度＝跟同一檔前一天數值完全相同的比例（門檻 50%）。
# 另外先找出「同一天所有股票都一樣」的欄位——它們沒有橫斷面排名可言（本案沒有這類欄位）。

# %%
evs = ev.sort_values(["sid", "date"])                        # 依股票與日期排序
gate = pd.DataFrame({"coverage": [ev[f].notna().mean() for f in FEATS],
                     "staleness": [(evs[f] == evs.groupby("sid")[f].shift(1)).mean() for f in FEATS],
                     "cs_std": [ev.groupby("date")[f].std().mean() for f in FEATS]}, index=FEATS)  # 三項指標
gate["date_level"] = gate.cs_std < 1e-10                     # 每天所有股票同值
gate["pass"] = (gate.coverage >= MIN_COV) & (gate.staleness <= MAX_STALE) & ~gate.date_level  # 是否通過
display(gate.round(3))                                       # 顯示
print(f"{int(gate['pass'].sum())}/{len(FEATS)} 項通過覆蓋率與陳舊度檢查")  # 印出

# %% [markdown]
# ## C. 每日 IC 與 Newey–West 修正

# %%
IC = {f: ic_by_date(ev, f, min_n=MIN_CS) for f in FEATS if gate.loc[f, "pass"]}  # 每日 IC 序列
S = pd.DataFrame({f: hac_stats(s, H) for f, s in IC.items()}).T  # HAC 統計
S["fdr_p"] = multipletests(S.hac_p, alpha=FDR, method="fdr_bh")[1]; S["fdr_sig"] = S.fdr_p < FDR  # BH 校正
n_naive, n_hac, n_fdr = int((S.naive_t.abs() > 1.96).sum()), int((S.hac_t.abs() > 1.96).sum()), int(S.fdr_sig.sum())  # 三種計數
print(f"搜尋集合 {len(S)} 項：天真 |t|>1.96 有 {n_naive} 項；Newey–West 後 {n_hac} 項；BH q<{FDR} 後 {n_fdr} 項")  # 印出

# %% [markdown]
# ## D. 三個驗證年度是否同方向？
# 一致性是對照**自己整體的方向**，不是對照正號：每一年都負的特徵和每一年都正的一樣穩定。

# %%
fold_ic = pd.DataFrame({f: [s[(s.index >= fd["valid"][0]) & (s.index <= fd["valid"][1])].mean() for fd in FOLDS] for f, s in IC.items()},
                       index=[fd["valid"][0][:4] for fd in FOLDS]).T  # 每年平均 IC
S["sign_consistency"] = [(np.sign(fold_ic.loc[f]) == np.sign(S.loc[f, "ic_mean"])).mean() for f in S.index]  # 同號比例
S["worst_fold_ic"] = fold_ic.min(axis=1); S = S.join(fold_ic.add_prefix("ic_"))  # 最差年度與各年
fig, ax = plt.subplots(figsize=(9, 5.5))                     # 每年 IC 點圖
order = S.ic_mean.abs().sort_values().index                  # 依 |IC| 排序
for i, f in enumerate(order):                                # 逐一特徵
    ax.scatter(fold_ic.loc[f], [i] * 3, color=GREY, s=22, zorder=2); ax.scatter(np.median(fold_ic.loc[f]), i, color=YELLOW, marker="D", s=50, zorder=3)  # 各年點與中位數
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.axvline(0, color=GREY, lw=.8); ax.set_xlabel("該年平均每日 IC")  # 座標
ax.set_title(f"{int((S.sign_consistency == 1).sum())}/{len(S)} 項特徵三年同方向；波動、區間、GARCH 三年都遠高於 0")  # 標題
save(fig, "05_fold_ic.png")                                  # 存圖

# %% [markdown]
# ## 每日 IC 序列本身（|IC| 最大的三項）
# 平均值背後的序列：淺灰是每日 IC，藍線是 6 個月移動平均，黃線是平均與 Newey–West 區間。

# %%
top3 = S.ic_mean.abs().sort_values(ascending=False).index[:3]  # 前三名
fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)  # 三列
for ax, f in zip(axs, top3):                                 # 逐一
    s = IC[f]; ax.plot(s.index, s, color="#cfcfca", lw=.6); ax.plot(s.index, s.rolling(126, min_periods=126).mean(), color=BLUE, lw=1.6)  # 每日與移動平均
    for v, ls in [(S.loc[f, "ic_mean"], "-"), (S.loc[f, "ic_lo"], ":"), (S.loc[f, "ic_hi"], ":")]: ax.axhline(v, color=YELLOW, ls=ls, lw=1)  # 平均與區間
    ax.axhline(0, color=GREY, lw=.8); ax.set_title(f, fontsize=10); ax.set_ylabel("IC")  # 標記
fig.suptitle("每日 IC 擺動遠大於平均值的不確定性，但 6 個月平均始終在 0 以上", x=.01, ha="left", fontsize=12)  # 總標題
save(fig, "05_top3_ic_series.png")                           # 存圖

# %% [markdown]
# ## E. 搜尋的代價：天真 t vs Newey–West t

# %%
fig, ax = plt.subplots(figsize=(6, 5))                       # 開圖
ax.scatter(S.naive_t, S.hac_t, c=[BLUE if v else GREY for v in S.fdr_sig], s=40)  # 散布點
m = max(S.naive_t.abs().max(), 1) * 1.1; ax.plot([-m, m], [-m, m], ls="--", color=GREY, lw=.8)  # 對角線
for f in S.index: ax.annotate(f, (S.loc[f, "naive_t"], S.loc[f, "hac_t"]), fontsize=6.5, alpha=.7)  # 標名稱
ax.set_xlabel("天真 t（把每天當獨立樣本）"); ax.set_ylabel("Newey–West t")  # 座標
ax.set_title(f"重疊標籤讓 t 值縮小約 {np.median(S.naive_t.abs() / S.hac_t.abs()):.1f} 倍；藍點通過 BH")  # 標題
save(fig, "05_naive_vs_hac.png")                             # 存圖

# %% [markdown]
# ## E2. 穩健性：頻寬夠不夠？
# 範本把頻寬設為標籤長度（5 日），前提是相鄰 IC 的相依只來自標籤重疊。但波動本身有持續性，特徵的排序也很穩定，
# 所以 IC 序列在 5 日之後可能仍然相關。下面先看 IC 序列自己的自我相關，再用兩種更保守的做法重算：
# 頻寬 20 日、以及只用**不重疊的決策日**（也就是策略真正交易的頻率）。

# %%
dec_dates = set(pd.read_csv(TAB / "decision_dates.csv", parse_dates=["decision_date"]).decision_date)  # 決策日
for f, s in IC.items():                                      # 逐一特徵
    b, d = hac_stats(s, 20), hac_stats(s[s.index.isin(dec_dates)], 1)  # 頻寬 20；決策日（不重疊）
    S.loc[f, ["acf_lag10", "t_h20", "p_h20", "ic_dec", "t_dec", "p_dec"]] = [s.autocorr(10), b["hac_t"], b["hac_p"], d["ic_mean"], d["hac_t"], d["hac_p"]]  # 寫入
S["bh_h20"] = multipletests(S.p_h20, alpha=FDR, method="fdr_bh")[0]; S["bh_dec"] = multipletests(S.p_dec, alpha=FDR, method="fdr_bh")[0]  # 兩種 BH
S["evidence"] = np.where(S.bh_h20 & S.bh_dec & (S.ic_mean.abs() >= 0.1), "強", np.where(S.bh_h20 & S.bh_dec, "中", np.where(S.bh_h20, "弱（只在每日重疊樣本顯著）", "無")))  # 證據強度
display(S[["ic_mean", "hac_t", "acf_lag10", "t_h20", "ic_dec", "t_dec", "bh_h20", "bh_dec", "evidence"]].sort_values("ic_mean", ascending=False).round(3))  # 顯示
print(f"IC 序列落後 10 日的自我相關中位數 {S.acf_lag10.median():.2f}：頻寬 5 日偏短。頻寬 20 日仍通過 BH：{int(S.bh_h20.sum())} 項；"
      f"只用決策日仍通過：{int(S.bh_dec.sum())} 項")             # 印出

# %% [markdown]
# ## F. 關係的形狀：五分位
# 每天依特徵把股票分 5 組（每組約 3 檔），看各組未來波動的平均與中位數，先在當天平均、再跨日平均。
# 單調性＝組別順序與組平均的 Spearman（±1 表示從頭到尾單調；接近 0 表示中間轉彎，線性模型會把它平均掉）。

# %%
def profile(f):                                              # 定義：五分位剖面
    t = ev[["date", f, "y"]].dropna()                        # 有值的列
    t = t[t.groupby("date")[f].transform("size") >= MIN_CS]  # 至少 10 檔的日期
    t["q"] = t.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))  # 當天分組
    m = t.groupby(["date", "q"]).y.mean().groupby(level=1).mean(); md = t.groupby(["date", "q"]).y.median().groupby(level=1).mean()  # 平均與中位數
    return m, md, spearmanr(m.index, m.values)[0]            # 回傳
PROF = {f: profile(f) for f in S.index}                      # 全部特徵
S["monotonicity"] = [PROF[f][2] for f in S.index]            # 單調性
show = S.ic_mean.abs().sort_values(ascending=False).index[:6]  # 前六名
fig, axs = plt.subplots(2, 3, figsize=(11, 5.5), sharey=True)  # 2×3
for ax, f in zip(axs.flat, show):                            # 逐一
    m, md, mono = PROF[f]; ax.bar([f"Q{i+1}" for i in range(5)], m.values, color=BLUE); ax.scatter([f"Q{i+1}" for i in range(5)], md.values, color=YELLOW, marker="D", zorder=3)  # 長條與中位數
    ax.set_title(f"{f}（單調 {mono:+.1f}）", fontsize=9)       # 標題
axs[0, 0].set_ylabel("未來 5 日波動（平均）"); fig.suptitle("最強的幾項從低分組到高分組一路單調上升；平均高於中位數代表右尾", x=.01, ha="left", fontsize=12)  # 標籤
save(fig, "05_quintiles.png")                                # 存圖

# %% [markdown]
# ## G. 哪些是同一份證據算兩次？
# |Spearman ρ| > 0.7 的配對：同一個量的不同窗長，或同一段價格路徑的兩種寫法。不需要刪掉，但一個家族有很多成員通過，
# 不代表有很多份獨立證據。

# %%
cm = ev[list(S.index)].rank().corr()                         # 特徵間 Spearman 相關
pairs = sorted([(a, b, cm.loc[a, b]) for i, a in enumerate(cm) for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > RED_CUT], key=lambda x: -abs(x[2]))  # 高相關配對
display(pd.DataFrame(pairs, columns=["特徵 A", "特徵 B", "ρ"]).round(3))  # 顯示
fam = pd.DataFrame({"family": [FAM[f] for f in S.index], "ic": S.ic_mean.values}).groupby("family").ic.agg(["size", lambda x: x.abs().mean(), "mean"])  # 家族摘要
fam.columns = ["特徵數", "平均 |IC|", "平均 IC（保留正負號）"]; display(fam.sort_values("平均 |IC|", ascending=False).round(3))  # 顯示

# %% [markdown]
# ## H. 每項特徵的決定
# | 決定 | 條件 | 路線 |
# |---|---|---|
# | **PROCEED** | 通過 BH（q < 0.05） | 確認 |
# | **PROCEED** | 三年同方向比例 ≥ 60% 且 \|IC\| ≥ 0.01 | 探索 |
# | **STOP** | 未通過覆蓋率或陳舊度 | — |
# | **REVISE** | 其他 | — |
#
# 依範本：這個篩選是**關於特徵的證據**，不是刪特徵的濾網；模型仍可使用全部特徵。

# %%
def decide(f):                                               # 定義：決策規則
    if not gate.loc[f, "pass"]: return "STOP", "correctness_fail"  # 覆蓋率或陳舊度不合格
    r = S.loc[f]                                             # 這項的統計
    if r.fdr_sig: return "PROCEED", "fdr_significant"        # 確認路線
    if r.sign_consistency >= MIN_SIGN and abs(r.ic_mean) >= IC_THR: return "PROCEED", "stable_and_above_threshold"  # 探索路線
    return "REVISE", "not_significant_standalone"            # 其他
L = S.copy(); L[["decision", "note"]] = [decide(f) for f in L.index]  # 套用
L.insert(0, "family", [FAM[f] for f in L.index]); L = L.join(gate[["coverage", "staleness"]])  # 加上家族與閘門
old = pd.read_csv(TAB / "triage_table.csv").assign(feature=lambda d: d.feature.map(NM)).set_index("feature")[["IC_mean", "decision"]]  # 先前（訓練期）的篩選
L = L.join(old.rename(columns={"IC_mean": "ic_train_2016_21", "decision": "decision_train"}))  # 對照
cols = ["family", "ic_mean", "ic_lo", "ic_hi", "hac_t", "naive_t", "fdr_p", "fdr_sig", "sign_consistency", "worst_fold_ic", "monotonicity",
        "t_h20", "bh_h20", "ic_dec", "t_dec", "bh_dec", "evidence", "coverage", "staleness", "decision", "note", "ic_train_2016_21", "decision_train"]  # 欄位順序
L = L[cols].sort_values("ic_mean", key=abs, ascending=False)  # 排序
L.round(4).to_csv(NB_OUT / "triage_ledger.csv", encoding="utf-8-sig")  # 寫出 ledger
pd.concat({f: s.rename("ic") for f, s in IC.items()}).rename_axis(["feature", "date"]).reset_index().to_parquet(NB_OUT / "ic_timeseries.parquet", index=False)  # 寫出 IC 序列
display(L[["family", "ic_mean", "ic_lo", "ic_hi", "fdr_sig", "sign_consistency", "monotonicity", "evidence", "decision", "decision_train"]].round(3))  # 顯示
fig, ax = plt.subplots(figsize=(9, 3))                       # 漏斗圖
stages = [("候選特徵", len(FEATS)), ("有橫斷面差異", int((~gate.date_level).sum())), ("通過覆蓋率／陳舊度", int(gate["pass"].sum())),
          ("通過 BH", n_fdr), ("PROCEED", int((L.decision == "PROCEED").sum()))]  # 各階段數量
ax.barh([s for s, _ in stages][::-1], [n for _, n in stages][::-1], color=[BLUE] + [GREY] * 4)  # 長條
for i, (_, n) in enumerate(stages[::-1]): ax.text(n + .2, i, str(n), va="center")  # 數字
ax.set_title("篩選漏斗"); save(fig, "05_funnel.png")         # 存圖

# %% [markdown] tags=["results"]
# ## 結論怎麼讀
# - **PROCEED（fdr_significant）**：在驗證年度上、扣掉重疊與多重檢定後仍顯著（範本規則）。
# - **evidence 欄**：再加兩道更保守的檢查（頻寬 20 日、只用不重疊決策日）。「強」＝兩道都過且 |IC| ≥ 0.1；
#   「弱」＝只在每日重疊樣本上顯著，策略實際交易的頻率上看不到。PROCEED 但證據「弱」的特徵，報告裡不能當成有效證據。
# - **REVISE**：單獨看沒有足夠證據；可能在跟其他特徵組合時有用，或需要換定義。
# - 最後一欄是先前只用訓練期（2016–2021）做的篩選，兩段期間的結論是否一致，本身就是穩定度的證據。
#
# **已知限制**：一次一項看不出組合效果（06/07 才回答）；只對 5 日標籤評估；驗證年度同時也被用來挑模型參數，
# 已經被讀過很多次；15 檔的橫斷面很窄（每天 12–15 檔），五分位每組只有約 3 檔。
