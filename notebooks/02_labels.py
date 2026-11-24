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
# # 第五組：標籤工程（Label Engineering）
#
# 每個模型都在預測標籤；標籤錯了不會有任何警告，卻會改變之後所有結果。本 notebook 寫下標籤公式、
# 用斷言檢查它、看它的分布與重疊程度，並在建特徵之前先量好「不做任何特徵工程」的基準線。
#
# 參考：ML4T 第 7 章 7.2、課程範本 `etfs/02_labels`。

# %%
from nb_utils import *                                       # 載入共用工具
import math                                                  # 數學工具
style()                                                      # 圖表樣式
panel = pd.read_parquet(PROC / "panel.parquet")              # 清理後日資料
panel = panel[panel.sid.isin(STK)].sort_values(["sid", "date"])  # 只看 15 檔股票
smp = pd.read_parquet(PROC / "samples.parquet")              # 管線已產生的標籤與特徵
smp = smp[smp.sid.isin(STK)]                                 # 只看股票

# %% [markdown]
# ## C. 標籤公式
#
# $$Y_{i,t}=\sqrt{\frac{252}{5}\sum_{k=1}^{5}\Big[\ln\frac{O_{i,t+1+k}}{O_{i,t+k}}\Big]^2}$$
#
# $O$ 是**還原後開盤價**，$t$ 是週五（決策日）收盤。持倉在 $t+1$ 開盤成交、$t+6$ 開盤調整，所以標籤量的
# 正是持有期間的波動，跟回測成交價完全對齊（範本用收盤到收盤，並在限制中承認它和隔日開盤成交不一致）。
# 下面在每檔**自己的**交易日序列上重算一次，和管線產生的標籤逐筆比對。

# %%
def label(g):                                                # 定義：單一股票的開盤到開盤 5 日波動
    ro = np.log(g.aopen / g.aopen.shift(1))                  # 開盤對數報酬（遇停牌空值就保持空值）
    return np.sqrt(252 / H * (ro ** 2).rolling(H, min_periods=H).sum()).shift(-(H + 1))  # t+2…t+6 的報酬平方和
re = pd.concat([label(g.set_index("date")).rename("y_check").reset_index().assign(sid=sid) for sid, g in panel.groupby("sid")])  # 逐檔重算（每檔獨立，不會跨股票）
chk = smp[["date", "sid", "y"]].merge(re, on=["date", "sid"], how="left")  # 與管線標籤合併
diff = (chk.y - chk.y_check).abs().max()                     # 最大差異
assert diff < 1e-10 and (chk.y.isna() == chk.y_check.isna()).all(), "重算的標籤與管線不一致"  # 斷言：完全一致
print(f"重算標籤與管線標籤一致：{chk.y.notna().sum():,} 筆有值，最大差異 {diff:.1e}")  # 印出

# %% [markdown]
# ## D. 窗口有效性：四個不會自己報錯的錯誤
# 1. 窗口不完整（最後 6 個交易日）一定是空值，不能有數字。
# 2. 窗口內只要有一天停牌或零成交（無效開盤價），標籤就是空值，不能跨過缺口。
# 3. 標籤不能跨到另一檔股票（逐檔計算 + 筆數核對）。
# 4. 標籤是非負的浮點數。

# %%
g = panel.merge(smp[["date", "sid", "y"]], on=["date", "sid"], how="left")  # 合併有效旗標與標籤
g["pos_from_end"] = g.groupby("sid").cumcount(ascending=False)  # 距離該檔最後一天幾個交易日
g["bad_open"] = g.aopen.isna().astype(int)                   # 無效開盤價
g["bad_in_window"] = g.groupby("sid").bad_open.transform(lambda b: b[::-1].rolling(H + 1, min_periods=1).sum()[::-1].shift(-1))  # t+1…t+6 有幾天無效
assert g[g.pos_from_end <= H].y.isna().all(), "尾端不完整窗口出現數值"  # 斷言 1
assert g[(g.bad_in_window > 0) & g.y.notna()].empty, "標籤跨過了停牌缺口"  # 斷言 2
cal = pd.DatetimeIndex(sorted(panel.date.unique()))          # 全市場交易日曆
holiday = int(pd.Series(cal[1:] - cal[:-1]).dt.days.max())   # 最長的全市場休市間隔（農曆年，最多 13 天）
tol = math.ceil((H + 1) * 7 / 5) + holiday                   # 容許天數：6 個交易日約 8.4 天 + 最長休市（由日曆推出，不是調出來的）
g["span"] = (g.groupby("sid").date.shift(-(H + 1)) - g.date).dt.days  # 標籤窗口橫跨的日曆天數
assert g[g.y.notna()].span.max() <= tol, "標籤窗口橫跨過長的缺口"  # 斷言 2b
assert len(chk) == len(smp), "標籤列數與樣本不符（可能跨股票）"  # 斷言 3
assert (smp.y.dropna() >= 0).all() and smp.y.dtype == float, "標籤不是非負浮點數"  # 斷言 4
print(f"四項全部通過：尾端空值 {int((g.pos_from_end <= H).sum())} 筆；因窗口內停牌而空值 {int(((g.bad_in_window > 0) & g.valid).sum())} 筆；"
      f"窗口最長 {int(g[g.y.notna()].span.max())} 天（容許 {tol} 天）")  # 印出

# %% [markdown]
# ## E. 分布與每年的尺度
# 以下只用**開發期**：依標籤「結束日」切（結束日在封存期之前），不是依觀察日。

# %%
dev = smp[smp.y.notna() & (smp.label_end < HOLDOUT_START)]   # 開發期標籤
print(f"開發期標籤 {len(dev):,} 筆，最後觀察日 {dev.date.max().date()}（結束日 {dev.label_end.max().date()}）")  # 印出
st = pd.Series({"平均": dev.y.mean(), "中位數": dev.y.median(), "標準差": dev.y.std(), "偏態": dev.y.skew(),
                "超額峰態": dev.y.kurt(), "剛好為 0 的比例": (dev.y == 0).mean(), "log(y) 偏態": np.log(dev.y[dev.y > 0]).skew()})  # 摘要統計
display(st.round(3).to_frame("開發期"))                       # 顯示
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))              # 兩張圖
axs[0].hist(dev.y.clip(upper=2), bins=80, color=BLUE); axs[0].set_xlabel("年化波動（>2 截在 2）"); axs[0].set_title("原始標籤：右偏、長尾")  # 原始分布
axs[1].hist(np.log(dev.y[dev.y > 0]), bins=80, color=AQUA); axs[1].set_xlabel("log(年化波動)"); axs[1].set_title("取對數後接近對稱")  # 對數分布
save(fig, "02_label_distribution.png")                       # 存圖
yr = dev.groupby([dev.date.dt.year, "date"]).y.agg(["mean", "std"]).groupby(level=0).mean()  # 每年平均水準與橫斷面離散度
fig, ax = plt.subplots(figsize=(10, 3.4))                    # 開圖
ax.bar(yr.index - .2, yr["mean"], .38, color=BLUE, label="當日平均波動（水準）"); ax.bar(yr.index + .2, yr["std"], .38, color=YELLOW, label="當日跨股票標準差（離散度）")  # 長條
ax.legend(frameon=False, fontsize=8); ax.set_title(f"水準與離散度都隨年份變動：離散度最高 {yr['std'].max():.2f}（{yr['std'].idxmax()}），中位年 {yr['std'].median():.2f}")  # 依數字產生標題
save(fig, "02_label_by_year.png")                            # 存圖

# %% [markdown]
# ## F. 重疊與有效樣本
# 模型用**每日**樣本訓練：相鄰兩天的標籤共用 5 個報酬中的 4 個，所以列數高估了資訊量。
# 下面在每檔內部算標籤的自我相關（扣掉各檔平均），預期在落後 5 天左右降到接近 0；有效樣本約為 N/5。
# 評估與回測只用每 5 日一次的決策日，時間上不重疊。

# %%
dd = dev.sort_values(["sid", "date"]).copy()                 # 依股票與日期排序
dd["yc"] = dd.y - dd.groupby("sid").y.transform("mean")      # 扣掉各檔平均
ac = [dd.yc.corr(dd.groupby("sid").yc.shift(k)) for k in range(1, 11)]  # 落後 1–10 日的合併自我相關
fig, ax = plt.subplots(figsize=(9, 3.2))                     # 開圖
ax.bar(range(1, 11), ac, color=BLUE, width=.6); ax.axvline(H, color=RED, ls=":", label=f"落後 {H} 日＝標籤長度"); ax.axhline(0, color=GREY, lw=.8)  # 長條與參考線
ax.legend(frameon=False); ax.set_xlabel("落後交易日"); ax.set_title(f"重疊在標籤長度內幾乎線性衰減；落後 {H} 日之後只剩波動本身的持續性")  # 標題
save(fig, "02_overlap_acf.png")                              # 存圖
n_dec = dev[dev.is_decision].shape[0]                        # 決策日樣本數
print(f"每日樣本 N={len(dev):,}，重疊調整後約 N/{H}={len(dev)//H:,}；決策日樣本 {n_dec:,}（{dev[dev.is_decision].date.nunique()} 個決策日）。"
      f"落後 1 日相關 {ac[0]:.2f}，落後 {H} 日 {ac[H-1]:.2f}，落後 {H+1} 日 {ac[H]:.2f}")  # 印出

# %% [markdown]
# ## G. 基準線：不做特徵工程的歷史波動 HV20
# 研究假設點名的最簡單訊號是「過去 20 日波動」。先在開發期量它的排名相關（IC），之後每個特徵都要跟這個
# 事先固定的數字比。IC 序列逐日計算，標準誤用 Newey–West，頻寬設為標籤長度（範本做法），並列出天真 t 值。

# %%
ic_daily = ic_by_date(dev, "hv20_L20")                       # 每日 IC（至少 10 檔）
ic_dec = ic_by_date(dev[dev.is_decision], "hv20_L20")        # 只用決策日
a, b = hac_stats(ic_daily, H), hac_stats(ic_dec, 1)          # HAC（每日序列用 5 期；決策日不重疊用 1 期）
mae = (dev[dev.is_decision].hv20_L20 - dev[dev.is_decision].y).abs().mean()  # HV20 直接當預測的 MAE
display(pd.DataFrame([a, b], index=["每日樣本", "決策日樣本"]).round(4))  # 顯示
print(f"基準線：HV20 對未來 5 日波動的平均 IC {a['ic_mean']:.3f}（HAC t {a['hac_t']:.1f}，天真 t {a['naive_t']:.1f}）；"
      f"決策日上直接當預測的 MAE {mae:.3f}")                  # 印出

# %% [markdown]
# ## H. 標籤覆蓋率：缺的都有原因嗎？
# 參考集合是「價格面板中有效的（股票, 交易日）」。每個缺標籤的位置都必須能被機制解釋：
# 研究起點前的暖身期、尾端 6 日、或窗口內有停牌。斷言要求「未解釋」為 0。

# %%
cov = g[g.valid].copy()                                      # 有效交易日
cov["reason"] = np.select([cov.y.notna(), cov.date < pd.Timestamp(CFG["data"]["study_start"]), cov.pos_from_end <= H, cov.bad_in_window > 0],
                          ["有標籤", "研究起點前（特徵暖身期，刻意不建樣本）", "尾端窗口不完整", "窗口內有停牌/零成交"], "未解釋")  # 分類原因
tab = cov.reason.value_counts(); display(tab.to_frame("筆數"))  # 顯示
assert tab.get("未解釋", 0) == 0, "有無法解釋的缺標籤"          # 斷言：沒有未解釋的缺口
inside = cov[cov.date >= pd.Timestamp(CFG["data"]["study_start"])]  # 研究期間內
print(f"研究期間內覆蓋率 {inside.y.notna().mean():.2%}；缺的 {int(inside.y.isna().sum()):,} 筆全部可由機制解釋")  # 印出

# %% [markdown] tags=["results"]
# ## 標籤紀錄
# | 項目 | 內容 |
# |---|---|
# | 錨點 | t+1 還原開盤價（週五收盤決策、下一交易日開盤成交） |
# | 期間 | 5 個開盤到開盤報酬（t+1→t+6） |
# | 何時揭曉 | t+6 開盤 |
# | 相鄰樣本重疊 | 4 個報酬（每日樣本）；決策日樣本不重疊 |
# | 尺度 | 年化波動，右偏；取 log 後接近對稱 |
#
# **已知限制**：股票池是今天的名單；停牌窗口直接丟掉，可能低估極端事件；標籤用平方報酬不扣平均，
# 5 個報酬的估計本身噪音很大（這是預測誤差的下限來源）。
