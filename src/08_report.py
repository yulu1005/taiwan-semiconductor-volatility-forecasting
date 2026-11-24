# 08_report.py — 總表、六張必附圖、風險分析、預測區間、Deflated Sharpe、Excel 彙整（讀取前面步驟的結果）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
import matplotlib; matplotlib.use("Agg")                     # 不開視窗
import matplotlib.pyplot as plt                              # 載入：畫圖
import matplotlib.dates as mdates                            # 載入：日期座標
from scipy.stats import norm                                 # 載入：常態分配
import shap                                                  # 載入：SHAP
from common import ROOT, CFG, PROC, TAB, FIG, setup_fonts, nw_mean_ci  # 載入：共用工具
from models import load, split, feat_cols, make_model, metrics  # 載入：模型工具
from strategies import parse, thresholds, make_target        # 載入：策略工具
from bt import Market, run, perf                             # 載入：回測引擎

setup_fonts()                                                # 中文字型
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": .6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.dpi": 100, "savefig.dpi": 200, "axes.titlesize": 12, "font.size": 10})  # 低調的格線與座標
COL = {"LGBM_dyn_EW": "#2a78d6", "HV_dyn_EW": "#eb6834", "EW100_股票等權": "#1baf7a", "Passive_0050": "#eda100"}  # 固定顏色：依策略，不依排名
LAB = {"LGBM_dyn_EW": "主策略：LightGBM 動態配置（等權）", "HV_dyn_EW": "非 ML 基準：歷史波動動態配置",
       "EW100_股票等權": "15 檔股票等權（100% 股票）", "Passive_0050": "被動基準：0050"}  # 圖例名稱
VS, HS = pd.Timestamp("2022-01-01"), pd.Timestamp(CFG["protocol"]["holdout"]["test"][0])  # 驗證起點、封存起點
df = load(); mk = Market()                                   # 樣本與市場資料
SEL = pd.read_csv(TAB/"selected_configs.csv", index_col=0).config  # 選定設定
EV = pd.read_parquet(PROC/"equity_val_full.parquet"); EH = pd.read_parquet(PROC/"equity_hold_full.parquet")  # 驗證與封存每日狀態
PV = pd.read_csv(TAB/"backtest_validation_perf.csv", index_col=0); PH = pd.read_csv(TAB/"backtest_holdout_perf.csv", index_col=0)  # 績效表
def eq(E, k): return E.xs(k, level=0)                        # 定義：取出某策略的每日狀態
def save(fig, name): fig.tight_layout(); fig.savefig(FIG/name, bbox_inches="tight"); plt.close(fig)  # 定義：存圖

# ---------- 預測資料（驗證：各 fold 樣本外；封存：最終模型） ----------
oos = pd.read_parquet(PROC/"oos_validation.parquet")         # 驗證期樣本外預測（含 2021 校準 fold）
hold = pd.concat([pd.read_parquet(PROC/f"bt_pred_hold_{m}.parquet").assign(model=m) for m in ("hv", "ridge", "lgbm")])  # 封存預測
hold = hold[hold.y.notna() & (hold.label_end <= pd.Timestamp(CFG["protocol"]["holdout"]["test"][1]))]  # 只留完整標籤
allp = pd.concat([oos[oos.model.isin(["hv", "ridge", "lgbm"])][["date", "sid", "y", "p", "label_end", "model"]], hold])  # 合併
allp["seg"] = np.where(allp.date >= HS, "holdout", np.where(allp.date >= VS, "validation", "calib"))  # 區段

# ---------- 1. 預測區間（conformal，名目 90%，只用先前已完成的樣本外誤差） ----------
rows, ivs = [], []                                           # 收集
for m, g in allp.groupby("model"):                           # 逐一模型
    g = g.sort_values("date"); g["s"] = (g.y - g.p).abs() / g.p  # 標準化誤差（相對誤差）
    for d, gd in g[g.seg != "calib"].groupby("date"):        # 逐一決策日
        cal = g[(g.label_end < d) & (g.date >= d - pd.Timedelta(days=365))].s.values  # 過去一年、標籤已結束的誤差
        n = len(cal); q = np.quantile(cal, min(1, np.ceil((n + 1) * .9) / n))  # 有限樣本修正的 90% 分位數
        lo, hi = (gd.p * (1 - q)).clip(lower=0), gd.p * (1 + q)  # 區間上下界
        ivs.append(pd.DataFrame({"model": m, "date": d, "sid": gd.sid, "y": gd.y, "p": gd.p, "lo": lo, "hi": hi}))  # 記錄
IV = pd.concat(ivs); IV["cover"] = (IV.y >= IV.lo) & (IV.y <= IV.hi); IV["width"] = IV.hi - IV.lo  # 是否涵蓋與寬度
cov = IV.groupby(["model", IV.date.dt.year]).agg(coverage=("cover", "mean"), avg_width=("width", "mean"), n=("cover", "size")).reset_index()  # 分年度
cov["flag_below_80pct"] = cov.coverage < 0.8                 # 低於 80% 要標示
cov.round(4).to_csv(TAB/"conformal_coverage.csv", index=False)  # 存檔

# ---------- 2. 每年 IC（HAC 區間）與滾動 IC ----------
def icser(g): return g.groupby("date").apply(lambda x: x.p.rank().corr(x.y.rank()) if len(x) >= 5 else np.nan, include_groups=False).dropna()  # 每日 IC
ICS = {m: icser(allp[(allp.model == m) & (allp.seg != "calib")]) for m in ("hv", "ridge", "lgbm")}  # 各模型 IC 序列
yic = pd.DataFrame([{"model": m, "year": y, **dict(zip(["IC", "lo", "hi", "se"], nw_mean_ci(s[s.index.year == y].values)))}
                    for m, s in ICS.items() for y in sorted(set(s.index.year))])  # 每年平均 IC 與區間
yic.round(4).to_csv(TAB/"ic_by_year_hac.csv", index=False)   # 存檔
fig, ax = plt.subplots(figsize=(9, 4))                       # 圖：每年 IC 與 HAC 區間
mc = {"hv": "#eb6834", "ridge": "#1baf7a", "lgbm": "#2a78d6"}; ml = {"hv": "歷史波動 HV20", "ridge": "Ridge", "lgbm": "LightGBM"}  # 顏色與名稱
for i, m in enumerate(("hv", "ridge", "lgbm")):              # 逐一模型
    t = yic[yic.model == m]; x = np.arange(len(t)) + (i - 1) * 0.25  # 位置
    ax.bar(x, t.IC, 0.22, color=mc[m], label=ml[m]); ax.errorbar(x, t.IC, yerr=[t.IC - t.lo, t.hi - t.IC], fmt="none", ecolor="#52514e", lw=1, capsize=2)  # 長條與區間
ax.set_xticks(range(len(t))); ax.set_xticklabels([f"{y}{'（封存）' if y >= 2025 else '（驗證）'}" for y in t.year])  # 年度標籤
ax.axhline(0, color="#52514e", lw=.8); ax.set_ylabel("平均每日排名相關 IC"); ax.legend(frameon=False, ncol=3, loc="upper left")  # 零線與圖例
ax.set_title("預測能分辨高低風險股票嗎？每年 IC 與 Newey–West 95% 區間（全部高於 0）"); save(fig, "extra_ic_by_year_hac.png")  # 標題與存檔

fig, ax = plt.subplots(figsize=(10, 3.8))                    # 圖 2：滾動 3 個月 IC
for m in ("hv", "lgbm"):                                     # 兩個模型
    s = ICS[m].rolling(13, min_periods=8).mean(); ax.plot(s.index, s, color=mc[m], lw=2, label=ml[m])  # 13 個決策日 ≈ 3 個月
ax.axhline(0, color="#52514e", lw=1); ax.axvline(HS, color="#0b0b0b", ls="--", lw=1); ax.text(HS, ax.get_ylim()[1] * .95, " 封存測試開始", va="top")  # 零線與分界
ax.set_ylabel("滾動 3 個月平均 IC"); ax.legend(frameon=False, loc="lower left"); ax.set_title("圖2　排名預測力是否穩定？滾動 3 個月 IC 始終高於 0；LightGBM 的領先在封存後期消失")  # 標籤
save(fig, "fig2_rolling_ic.png")                             # 存檔

# ---------- 3. 權益曲線（淨成本，驗證接封存） ----------
def chain(k):                                                # 定義：驗證期與封存期淨值串接（封存從驗證期末值開始）
    v = eq(EV, k).value; h = eq(EH, k).value                 # 兩段淨值
    return pd.concat([v / v.iloc[0], h / h.iloc[0] * (v.iloc[-1] / v.iloc[0])])  # 標準化並串接
fig, ax = plt.subplots(figsize=(10, 4.6))                    # 圖 1
for k in COL: s = chain(k); ax.plot(s.index, s, color=COL[k], lw=2.2 if k == "LGBM_dyn_EW" else 1.6, label=LAB[k])  # 四條線
ax.axvline(HS, color="#0b0b0b", ls="--", lw=1); ax.text(HS, ax.get_ylim()[1] * .97, " 驗證｜封存測試", va="top")  # 分界線
ax.set_yscale("log"); ax.yaxis.set_major_locator(plt.FixedLocator([0.6, 0.8, 1, 1.5, 2, 3])); ax.yaxis.set_minor_locator(plt.NullLocator()); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax.set_ylabel("淨值（起點 = 1，對數座標）"); ax.legend(frameon=False, loc="upper left", fontsize=9)  # 座標與圖例
ax.set_title("圖1　扣成本後，波動配置是否勝過基準？（2022–2024 驗證、2025–2026/10 封存）"); save(fig, "fig1_equity_curves.png")  # 存檔

# ---------- 4. 毛 vs 淨 Sharpe 與兩平成本 ----------
be = []                                                      # 收集兩平成本
for seg, f in (("驗證", "cost_sweep_validation.csv"), ("封存", "cost_sweep_holdout.csv")):  # 兩段
    sw = pd.read_csv(TAB/f)                                  # 成本掃描
    for k, g in sw.groupby("strategy"):                      # 逐一策略
        gr, nt = g[g.cost_mult == "gross"].iloc[0], g[g.cost_mult == "1.0"].iloc[0] if (g.cost_mult == "1.0").any() else g[g.cost_mult == "1"].iloc[0]  # 毛與淨
        be.append({"segment": seg, "strategy": k, "gross_Sharpe": gr.Sharpe, "net_Sharpe": nt.Sharpe, "gross_CAGR": gr.CAGR, "net_CAGR": nt.CAGR,
                   "turnover_2way": gr.turnover_2way, "breakeven_bp_per_leg": gr.CAGR / gr.turnover_2way * 1e4,
                   "assumed_bp_per_leg": 19.25 + 15})        # 假設：手續費 14.25 + 滑價 5 + 稅 30/2
BE = pd.DataFrame(be); BE.round(4).to_csv(TAB/"breakeven_both.csv", index=False)  # 存檔
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)  # 圖 3：兩段各一張
for ax, seg in zip(axs, ("驗證", "封存")):                    # 逐一區段
    t = BE[BE.segment == seg].set_index("strategy").loc[["LGBM_dyn_EW", "HV_dyn_EW", "EW100_股票等權"]]  # 三策略
    x = np.arange(3)                                         # 位置
    ax.bar(x - .18, t.gross_Sharpe, .34, color="#9ec5f0", label="毛（無成本）"); ax.bar(x + .18, t.net_Sharpe, .34, color="#2a78d6", label="淨（含稅費滑價）")  # 毛與淨
    for i, (_, r) in enumerate(t.iterrows()):                # 標註兩平成本
        txt = f"兩平 {r.breakeven_bp_per_leg:.0f}bp/邊" if r.breakeven_bp_per_leg > 0 else "毛報酬已為負"  # 文字
        ax.text(i, max(r.gross_Sharpe, r.net_Sharpe, 0), txt + "\n", va="bottom", ha="center", fontsize=8.5, color="#52514e")  # 放在長條上方
    lo_, hi_ = min(t.gross_Sharpe.min(), t.net_Sharpe.min(), 0), max(t.gross_Sharpe.max(), t.net_Sharpe.max(), 0)  # 資料範圍
    ax.set_ylim(lo_ - .1 * (hi_ - lo_ + .1), hi_ + .25 * (hi_ - lo_ + .1))  # 上方留空間放標註
    ax.set_xticks(x); ax.set_xticklabels(["LightGBM 動態", "HV 動態", "股票等權"]); ax.axhline(0, color="#52514e", lw=.8)  # 座標
    ax.set_title(f"{seg}期"); ax.set_ylabel("年化 Sharpe")   # 標題
axs[0].legend(frameon=False, loc="lower left"); fig.suptitle("圖3　成本吃掉多少？毛 vs 淨 Sharpe（假設成本約 34bp/邊，含稅平均）", fontsize=12)  # 總標題
save(fig, "fig3_gross_vs_net_sharpe.png")                    # 存檔

fig, ax = plt.subplots(figsize=(8, 4))                       # 補充圖：成本掃描
for seg, f, ls in (("驗證", "cost_sweep_validation.csv", "-"), ("封存", "cost_sweep_holdout.csv", "--")):  # 兩段
    sw = pd.read_csv(TAB/f); sw = sw[sw.cost_mult != "gross"].assign(bp=lambda x: x.per_leg_bp_ex_tax.astype(float))  # 去掉毛報酬列
    for k in ("LGBM_dyn_EW", "HV_dyn_EW", "EW100_股票等權"):  # 三策略
        t = sw[sw.strategy == k]; ax.plot(t.bp, t.Sharpe, ls, color=COL[k], marker="o", ms=4, lw=1.8, label=f"{LAB[k][:12]}（{seg}）")  # 線
ax.axvline(19.25, color="#52514e", lw=.8, ls=":"); ax.text(19.25, ax.get_ylim()[0], " 假設成本", va="bottom", fontsize=8)  # 假設成本位置
ax.set_xlabel("手續費＋滑價（bp/邊，不含證交稅）"); ax.set_ylabel("淨 Sharpe"); ax.legend(frameon=False, fontsize=7.5, ncol=2)  # 標籤
ax.set_title("成本提高到 4 倍時 Sharpe 如何變化？"); save(fig, "extra_cost_sweep.png")  # 存檔

# ---------- 5. 市場狀態切片（0050 波動 × 趨勢，用 t−1 資訊、訓練期門檻） ----------
b = mk.aclose["0050"]; r50 = np.log(b / b.shift(1))          # 0050 還原價與報酬
hv = np.sqrt(252 / 20 * (r50 ** 2).rolling(20).sum())        # 0050 20 日波動
t1, t2 = hv.loc["2016-11-01":"2021-12-31"].quantile([1 / 3, 2 / 3])  # 門檻：只用 2016-11～2021 資料
volr = pd.cut(hv, [-np.inf, t1, t2, np.inf], labels=["低波動", "中波動", "高波動"]).shift(1)  # 前一日的波動狀態
trend = np.where(b > b.rolling(60).mean(), "上升趨勢", "下降趨勢"); trend = pd.Series(trend, index=b.index).shift(1)  # 前一日趨勢
def rets(k): s = pd.concat([eq(EV, k).value.pct_change().dropna(), eq(EH, k).value.pct_change().dropna()]); return s  # 定義：兩段日報酬
reg = []                                                     # 收集
for k in ("LGBM_dyn_EW", "EW100_股票等權", "Passive_0050"):  # 三策略
    r = rets(k); g = pd.DataFrame({"r": r, "vol": volr.reindex(r.index), "trend": trend.reindex(r.index)}).dropna()  # 合併狀態
    for (v, tr), x in g.groupby(["vol", "trend"], observed=True):  # 逐一狀態
        c = (1 + x.r).cumprod(); reg.append({"strategy": k, "vol": v, "trend": tr, "days": len(x), "ann_ret": x.r.mean() * 252,
                                             "Sharpe": x.r.mean() / x.r.std() * np.sqrt(252), "MDD_concat": (c / c.cummax() - 1).min()})  # 指標
REG = pd.DataFrame(reg); REG.round(4).to_csv(TAB/"regime_slice.csv", index=False)  # 存檔
fig, axs = plt.subplots(1, 2, figsize=(12, 4))               # 圖 4
cells = [(v, t) for t in ("上升趨勢", "下降趨勢") for v in ("低波動", "中波動", "高波動")]  # 六個狀態
for ax, met, yl in zip(axs, ("Sharpe", "ann_ret"), ("年化 Sharpe", "年化報酬（日平均×252）")):  # 兩個指標各一張
    for i, k in enumerate(("LGBM_dyn_EW", "EW100_股票等權", "Passive_0050")):  # 三策略
        t = REG[REG.strategy == k].set_index(["vol", "trend"]).reindex(cells)  # 依狀態排序
        ax.bar(np.arange(6) + (i - 1) * .26, t[met], .24, color=COL[k], label=LAB[k])  # 長條
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{v}\n{t}\n({int(REG[(REG.strategy=='LGBM_dyn_EW')&(REG.vol==v)&(REG.trend==t)].days.sum())}天)" for v, t in cells], fontsize=8)  # 標籤含天數
    ax.axhline(0, color="#52514e", lw=.8); ax.set_ylabel(yl)  # 零線
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.06)); fig.suptitle("圖4　策略在哪種市場狀態失效？0050 波動（訓練期三分位）× 60 日均線趨勢，皆用前一日資訊", fontsize=12)  # 標題
save(fig, "fig4_regime_slice.png")                           # 存檔

# ---------- 6. SHAP（部署模型＝以 2016-11～2024 訓練的最終 LightGBM，解釋封存期樣本） ----------
cols = feat_cols(20); H = CFG["protocol"]["holdout"]         # 特徵與封存設定
tr, _ = split(df, H["train"][1], *H["test"])                 # 最終訓練集
kind, prm = parse(SEL["lgbm"]); m = make_model(kind, prm).fit(tr[cols], tr.y)  # 重新訓練（固定種子，與 07 相同）
te = df[df.is_decision & (df.date >= pd.Timestamp(H["test"][0])) & (df.date <= pd.Timestamp(H["test"][1]))]  # 封存期決策日樣本
sv = shap.TreeExplainer(m).shap_values(te[cols])             # SHAP 值
NM = {"hv5": "5日波動", "hv10": "10日波動", "hv20": "20日波動", "ret1": "1日報酬", "ret5": "5日報酬", "ret20": "20日報酬", "hl": "高低價差比",
      "oc": "開收價差比", "ma5_dist": "5日均線距離", "ma20_dist": "20日均線距離", "vol_ratio": "成交量比", "vol_cv": "量的變動",
      "amt_ratio": "成交金額比", "rsi": "RSI14", "atr_p": "ATR14/股價"}  # 中文名稱
X = te[cols].copy(); X.columns = [NM[c.replace("_L20", "")] for c in cols]  # 換成中文欄名
plt.figure(); shap.summary_plot(sv, X, show=False, max_display=15, plot_size=(9, 6))  # 蜂群圖
plt.title("圖5　模型靠什麼預測高波動？最終 LightGBM 的 SHAP（封存期樣本）", fontsize=12); plt.savefig(FIG/"fig5_shap_beeswarm.png", bbox_inches="tight", dpi=200); plt.close()  # 存檔
pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0),
              "sign_corr": [np.corrcoef(X.iloc[:, i], sv[:, i])[0, 1] for i in range(len(cols))]}).sort_values("mean_abs_shap", ascending=False).round(4).to_csv(
    TAB/"shap_final_holdout.csv", index=False)               # 存 SHAP 摘要

# ---------- 7. 回撤圖與停損階梯 ----------
s = chain("LGBM_dyn_EW"); dd = s / s.cummax() - 1            # 主策略回撤
s2 = chain("EW100_股票等權"); dd2 = s2 / s2.cummax() - 1     # 等權回撤
fig, ax = plt.subplots(figsize=(10, 4))                      # 圖 6
ax.fill_between(dd.index, dd, 0, color="#2a78d6", alpha=.35, lw=0); ax.plot(dd.index, dd, color="#2a78d6", lw=1.5, label=LAB["LGBM_dyn_EW"])  # 主策略
ax.plot(dd2.index, dd2, color="#1baf7a", lw=1.2, label=LAB["EW100_股票等權"])  # 等權
for lv, txt in zip((.05, .10, .15, .20), ("觀察", "減半", "停止加碼", "全部出清")):  # 停損階梯
    ax.axhline(-lv, color="#e34948", lw=.9, ls="--"); ax.text(dd.index[0], -lv + .004, f"−{int(lv*100)}% {txt}", va="bottom", fontsize=8, color="#52514e", bbox=dict(facecolor="white", edgecolor="none", pad=1))  # 門檻線
ax.axvline(HS, color="#0b0b0b", ls=":", lw=1)                # 封存分界
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.legend(frameon=False, loc="lower right")  # 百分比與圖例
ax.set_title("圖6　停損階梯何時會觸發？自高點回撤與 5/10/15/20% 門檻"); save(fig, "fig6_drawdown_killswitch.png")  # 存檔

# ---------- 8. 簡報第 10 頁用：風險等級與資產比例 ----------
TL = pd.concat([pd.read_csv(TAB/"tier_log_val_primary.csv"), pd.read_csv(TAB/"tier_log_hold_primary.csv")]); TL["date"] = pd.to_datetime(TL.date)  # 風險等級紀錄
real = allp[(allp.model == "lgbm") & (allp.seg != "calib")].groupby("date").y.mean()   # 每日平均實際波動
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})  # 上下兩張
a = axs[0]; a.plot(TL.date, TL.avg_pred, color="#2a78d6", lw=1.8, label="平均預測波動（LightGBM）"); a.plot(real.index, real, color="#8a8984", lw=1, label="平均實際波動（事後）")  # 預測與實際
a.step(TL.date, TL.lo, where="post", color="#1baf7a", lw=1, ls="--", label="低門檻（訓練期 P33）"); a.step(TL.date, TL.hi, where="post", color="#e34948", lw=1, ls="--", label="高門檻（訓練期 P67）")  # 門檻
a.axvline(HS, color="#0b0b0b", ls=":", lw=1); a.set_ylabel("年化波動率"); a.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")  # 標籤
a.set_title("預測風險落在哪一段？決定股票 70／50／30%")       # 標題
e = pd.concat([eq(EV, "LGBM_dyn_EW"), eq(EH, "LGBM_dyn_EW")]); cash = 1 - e.stock_w - e.gold_w  # 每日權重
axs[1].stackplot(e.index, e.stock_w, e.gold_w, cash, colors=["#2a78d6", "#eda100", "#d9d8d2"], labels=["股票", "黃金 00635U", "現金"], edgecolor="white", lw=.3)  # 堆疊面積
axs[1].axvline(HS, color="#0b0b0b", ls=":", lw=1); axs[1].set_ylim(0, 1); axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))  # 座標
axs[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); axs[1].set_ylabel("實際持有比例")  # 圖例
save(fig, "slide10_tier_and_weights.png")                    # 存檔
TL.tier.groupby(TL.date >= HS).value_counts().unstack().rename(index={False: "驗證", True: "封存"}).to_csv(TAB/"tier_counts.csv")  # 風險等級次數

# ---------- 9. 風險：VaR/CVaR、Beta、壓力測試 ----------
rk = []                                                      # 收集
for seg, E in (("驗證", EV), ("封存", EH)):                   # 兩段
    r0 = eq(E, "Passive_0050").value.pct_change()            # 0050 日報酬
    for k in ("LGBM_dyn_EW", "HV_dyn_EW", "Fixed_50-20-30", "EW100_股票等權", "Passive_0050"):  # 策略
        r = eq(E, k).value.pct_change().dropna(); v = np.percentile(r, 5)  # 日報酬與 5% 分位
        beta = r.rolling(60).cov(r0) / r0.rolling(60).var()  # 60 日滾動 beta
        rv = r[volr.reindex(r.index) == "高波動"]             # 高波動狀態下的報酬
        rk.append({"segment": seg, "strategy": k, "VaR95_daily": -v, "CVaR95_daily": -r[r <= v].mean(),
                   "VaR95_highvol": -np.percentile(rv, 5) if len(rv) > 20 else np.nan, "beta60_mean": beta.mean(), "beta60_max": beta.max()})  # 指標
pd.DataFrame(rk).round(4).to_csv(TAB/"risk_var_cvar_beta.csv", index=False)  # 存檔
st = []                                                      # 壓力測試
dec = pd.read_csv(TAB/"decision_dates.csv", parse_dates=["decision_date"]).decision_date  # 決策日
pv = pd.read_parquet(PROC/"bt_pred_val_lgbm.parquet"); ph = pd.read_parquet(PROC/"bt_pred_hold_lgbm.parquet")  # 預測
thv = pd.read_csv(TAB/"tier_thresholds_validation.csv").query("model=='lgbm'").set_index("year")  # 驗證門檻
fv = lambda d: tuple(thv.loc[d.year, ["thr_lo_p33", "thr_hi_p67"]]); th = thresholds(df, H["train"][1], H["test"][0])  # 門檻函數
end_h = mk.dates[mk.dates <= pd.Timestamp(H["test"][1])][-1]  # 封存終點
stress_eq = {"驗證": run(mk, list(dec[(dec >= VS) & (dec <= "2024-12-31")]), make_target(pv, fv), end=pd.Timestamp("2024-12-31"), slip_mult=3)[0],
             "封存": run(mk, list(dec[(dec >= HS) & (dec < end_h)]), make_target(ph, lambda d: th), end=end_h, slip_mult=3)[0]}  # 滑價 3 倍
for name, (a0, a1), seg in (("2022 升息與科技股修正", ("2022-01-03", "2022-10-25"), "驗證"), ("2025/4 關稅衝擊", ("2025-03-26", "2025-04-30"), "封存")):  # 兩段歷史事件
    E = EV if seg == "驗證" else EH                           # 對應區段
    for k, v in [(k, eq(E, k).value) for k in ("LGBM_dyn_EW", "HV_dyn_EW", "EW100_股票等權", "Passive_0050")] + [("LGBM_dyn_EW（滑價×3）", stress_eq[seg].value)]:  # 策略
        x = v.loc[a0:a1]; st.append({"episode": name, "strategy": k, "return": x.iloc[-1] / x.iloc[0] - 1, "MDD": (x / x.cummax() - 1).min()})  # 期間報酬與回撤
pd.DataFrame(st).round(4).to_csv(TAB/"stress_tests.csv", index=False)  # 存檔

# ---------- 10. 統計推論：Sharpe 區間、Deflated Sharpe、最短紀錄長度 ----------
rl = pd.read_csv(ROOT/"run_log.csv")                         # 研究帳本
pre = rl[rl.logged_at < rl[rl.trial_family == "HOLDOUT"].logged_at.min()]  # 封存測試執行之前記錄的試驗（之後的 v2 試驗不算進 v1 的 DSR）
N = pre.drop_duplicates(["trial_family", "family", "config", "fset", "lookback"]).shape[0]  # 試驗次數（每個設定算一次，不重複算 fold；封存期紀錄不算）
srs = PV.Sharpe / np.sqrt(252)                               # 驗證期各策略與變體的日 Sharpe
def dsr(r, N, var_sr):                                       # 定義：Deflated Sharpe Ratio（Bailey & López de Prado）
    n = len(r); sr = r.mean() / r.std(); g3, g4 = r.skew(), r.kurt() + 3  # 日 Sharpe、偏態、峰態
    e = 0.5772156649                                         # Euler–Mascheroni 常數
    sr0 = np.sqrt(var_sr) * ((1 - e) * norm.ppf(1 - 1 / N) + e * norm.ppf(1 - 1 / (N * np.e)))  # 純運氣下的預期最大 Sharpe
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)  # 標準化
    psr0 = norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))  # PSR（相對於 0）
    return norm.cdf(z), psr0, sr0 * np.sqrt(252)             # 回傳 DSR、PSR、年化門檻
def mintrl(sr): return ((1.645 + 0.842) / sr) ** 2 if sr > 0 else np.inf  # 最短紀錄長度（年），α=5%、檢定力 80%

# ---------- 11. 總表（headline table） ----------
sig = {}                                                     # 訊號層
for seg in ("validation", "holdout"):                        # 兩段
    s = ICS["lgbm"][(ICS["lgbm"].index >= (HS if seg == "holdout" else VS)) & (ICS["lgbm"].index < (pd.Timestamp("2100-01-01") if seg == "holdout" else HS))]  # IC 序列
    m_, lo, hi, _ = nw_mean_ci(s.values); yrs = s.groupby(s.index.year).mean()  # 平均、區間、各年
    sig[seg] = f"{m_:.3f} [{lo:.3f}, {hi:.3f}]；ICIR {s.mean()/s.std():.2f}；正 IC 年度 {int((yrs>0).sum())}/{len(yrs)}"  # 文字
def sw_row(f, k, mult): t = pd.read_csv(TAB/f); return t[(t.strategy == k) & (t.cost_mult.astype(str) == mult)].iloc[0]  # 定義：取成本掃描某列
def fmt_sr(r): return f"{r.Sharpe:.2f} [{r.Sharpe_lo:.2f}, {r.Sharpe_hi:.2f}]"  # Sharpe 與區間
HT = []                                                      # 總表列
for seg, f, P, E in (("驗證 2022–2024", "cost_sweep_validation.csv", PV, EV), ("封存 2025–2026/10", "cost_sweep_holdout.csv", PH, EH)):  # 兩段
    g = sw_row(f, "LGBM_dyn_EW", "gross"); n1 = P.loc["LGBM_dyn_EW"]; ov = P.loc["LGBM_dyn_EW_overlay15"]  # 毛、淨、覆蓋
    bsel = BE[(BE.segment == seg[:2]) & (BE.strategy == "LGBM_dyn_EW")].iloc[0]  # 兩平
    r = eq(E, "LGBM_dyn_EW").value.pct_change().dropna(); D, PS, sr0 = dsr(r, N, srs.var())  # 推論
    HT.append({"區段": seg, "訊號：IC [HAC 95%]": sig["validation" if seg.startswith("驗證") else "holdout"],
               "投組：毛 Sharpe [95%]": fmt_sr(g), "投組：MDD": f"{g.MDD:.1%}", "投組：年化雙邊換手": f"{g.turnover_2way:.2f}",
               "成本：淨 Sharpe": fmt_sr(n1), "成本：淨年化報酬": f"{n1.CAGR:.1%}", "成本：兩平成本 bp/邊": f"{bsel.breakeven_bp_per_leg:.0f}" if bsel.breakeven_bp_per_leg > 0 else "毛報酬為負",
               "風險覆蓋：淨 Sharpe／MDD（回撤15%停止加碼）": f"{ov.Sharpe:.2f}／{ov.MDD:.1%}",
               f"推論：DSR（試驗數 N={N}）": f"{D:.2f}（PSR>0：{PS:.2f}）", "推論：最短紀錄長度（年）": "∞（Sharpe≤0）" if n1.Sharpe <= 0 else f"{mintrl(n1.Sharpe):.1f}",
               "基準：股票等權 淨 Sharpe／MDD": f"{P.loc['EW100_股票等權'].Sharpe:.2f}／{P.loc['EW100_股票等權'].MDD:.1%}",
               "基準：0050 淨 Sharpe／MDD": f"{P.loc['Passive_0050'].Sharpe:.2f}／{P.loc['Passive_0050'].MDD:.1%}",
               "非ML規則：HV 動態 淨 Sharpe／MDD": f"{P.loc['HV_dyn_EW'].Sharpe:.2f}／{P.loc['HV_dyn_EW'].MDD:.1%}"})  # 一列
HTd = pd.DataFrame(HT).set_index("區段").T; HTd.to_csv(TAB/"headline_table.csv", encoding="utf-8-sig")  # 存總表（轉置成直式）
print(HTd.to_string()); print("N trials", N)                 # 印出

# ---------- 12. 文件用的表：PIT 檢查、風險控制矩陣 ----------
pd.DataFrame([
    ["數值", "價格與量：t 日 13:30 收盤後已知；特徵只用 ≤ t 的資料；回溯還原因子只影響比例型特徵的尺度，不引入未來資訊", "PASS"],
    ["數值", "標籤 t+1 開盤 → t+6 開盤，與回測成交價一致；訓練剔除標籤未在驗證起點前結束的樣本（purge 6 日）", "PASS"],
    ["數值", "標準化、截尾、風險門檻只用當次訓練資料估計", "PASS"],
    ["股票池", "15 檔為今日現存名單，未納入期間下市或被剔除的半導體股（事後選股）", "FAIL（已揭露：可能高估報酬、低估風險）"],
    ["股票池", "上市前（興櫃）資料剔除：5222、6770、8028；上市後需 21 個有效價格才合格", "PASS"],
    ["股票池", "2371 大同（電機機械）、2455 全新（通信網路）證交所產業別非半導體", "揭露（名單定義問題）"],
    ["代號", "15 檔期間內無代號重複使用；0050 分割 2025-06-18 手動加入因子", "PASS"],
    ["代號", "公司行動：除權息 + 減資參考價還原，殘餘 >10.5% 跳動 2 筆皆有合理解釋", "PASS"]],
    columns=["層次", "檢查內容", "結果"]).to_csv(TAB/"pit_checklist.csv", index=False, encoding="utf-8-sig")  # 存 PIT 檢查表
pd.DataFrame([
    ["市場", "組合對 0050 的 60 日 beta、股票總比例", "依預測波動 70/50/30%；黃金 20%", "每次決策"],
    ["因子／產業集中", "全部為半導體鏈，記憶體股權重", "與 0050 比較；揭露產業集中", "每月"],
    ["槓桿", "現金是否為負", "只做多、不借款、先賣後買", "每次成交"],
    ["個股集中", "最大單檔權重、HHI", "單檔 ≤ 股票部位 20%（≤ 全組合 14%）", "每次決策"],
    ["流動性", "訂單 ÷ 20 日平均成交金額、未成交金額", "訂單 ≤ 5% ADV；停牌/鎖漲跌停不成交", "每次成交"],
    ["模型", "滾動 MAE 相對 HV20、IC、預測區間涵蓋率", "連續兩季 MAE 劣於 HV20 → 改用 HV20 規則", "每季"],
    ["操作", "資料缺漏、預測異常、權重違規", "異常即暫停新增交易", "每次決策前"]],
    columns=["風險", "代理指標", "控制方式", "檢查頻率"]).to_csv(TAB/"risk_control_matrix.csv", index=False, encoding="utf-8-sig")  # 存風險控制矩陣
