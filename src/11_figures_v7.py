# 11_figures_v7.py — 把 V7 的結果畫成圖，存到 outputs/v7/figures
#   配色：策略 B（用 Model 1）＝藍色重點；策略 A（HV20）＝深灰；0050 參考＝淺灰虛線
import json, numpy as np, pandas as pd                       # 載入：JSON、數值、表格
import matplotlib.pyplot as plt, matplotlib.dates as mdates  # 載入：畫圖、日期刻度
from common import PROC, VTAB, VFIG, setup_fonts             # 載入：路徑與中文字型
setup_fonts()                                                # 讓圖可以顯示中文
M2 = VTAB / "model2"                                         # Model 2 結果資料夾
BLUE, GREY, LIGHT = "#2a6fdb", "#555555", "#b0b0b0"          # 三種顏色
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "figure.dpi": 150})  # 簡潔外觀

def save(fig, name):                                         # 定義：存檔並關閉
    for ax in fig.axes:                                      # 日期軸：每半年一個刻度，避免擠在一起
        if isinstance(ax.xaxis.get_major_formatter(), mdates.AutoDateFormatter) or ax.xaxis.get_major_formatter().__class__.__name__ == "ConciseDateFormatter":
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(); fig.savefig(VFIG / name, bbox_inches="tight"); plt.close(fig); print("已存", name)

# ---------- 圖 1：Model 1 各年度 QLIKE（越低越好） ----------
t = pd.read_csv(VTAB / "model1_table_2_8.csv").set_index("config")
ph = pd.read_csv(VTAB / "model1_posthoc_2025_2026.csv").set_index("config")
names = {"lasso_alpha_frac0.1": "Lasso（主模型）", "hv": "HV20", "ewma": "EWMA", "garch": "GARCH（開盤到開盤）", "har": "HAR"}
Q = pd.DataFrame({"2022": t.fold1, "2023": t.fold2, "2024": t.fold3, "2025–26（事後）": ph.QLIKE}).loc[list(names)]
fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.16
for i, (k, lab) in enumerate(names.items()):                 # 每個模型一組顏色：主模型藍色，其他灰階
    col = BLUE if i == 0 else plt.cm.Greys(0.35 + 0.13 * i)
    ax.bar(np.arange(4) + (i - 2) * w, Q.loc[k], w, label=lab, color=col)
ax.set_xticks(range(4), Q.columns); ax.set_ylabel("QLIKE（越低越好）")
ax.set_title("Model 1：Lasso 在 2022–2024 每年都最低，2025–26 輸給 GARCH 與 EWMA", loc="left", fontsize=11)
ax.legend(ncol=5, fontsize=8, loc="upper left", frameon=False); save(fig, "v7_fig1_model1_qlike.png")

# ---------- 圖 2：預測區間實際涵蓋率 ----------
c = pd.read_csv(VTAB / "model1_conformal.csv"); c["期間"] = ["2022", "2023", "2024", "2025–26"]
fig, ax = plt.subplots(figsize=(6, 3.2)); ax.bar(c["期間"], c.coverage, color=GREY, width=0.5)
ax.axhline(0.9, color=BLUE, ls="--", lw=1); ax.text(0.5, 0.903, "目標 90%", color=BLUE, ha="center", fontsize=9)
for i, v in enumerate(c.coverage): ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
ax.set_ylim(0.8, 0.96); ax.set_ylabel("實際涵蓋率"); ax.set_title("90% 預測區間的實際涵蓋率（以 2021 殘差校準）", loc="left", fontsize=11)
save(fig, "v7_fig2_conformal_coverage.png")

# ---------- 圖 3：相關性窗口比較（V7 5.5） ----------
wc = pd.read_csv(M2 / "window_comparison.csv")
wc["設定"] = wc.apply(lambda r: f"{int(r.L)} 日" + ("" if r.shrink else "（不收縮）"), axis=1)
wc = wc.sort_values("三年平均")
fig, ax = plt.subplots(figsize=(7, 3.4))
cols = [BLUE if s == "20 日" else (GREY if s == "60 日" else LIGHT) for s in wc["設定"]]
ax.barh(wc["設定"], wc["三年平均"], color=cols); ax.invert_yaxis()
for i, v in enumerate(wc["三年平均"]): ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("策略 B 組合波動預測的 QLIKE，2022–2024 平均（越低越好）")
ax.set_title("相關性窗口：20 日比 60 日低 3.2%，超過 1% 平手範圍 → 選 20 日", loc="left", fontsize=11); ax.set_xlim(0.3, 0.4)
save(fig, "v7_fig3_window.png")

# ---------- 圖 4：α 選擇（V7 4.4） ----------
al = pd.read_csv(M2 / "alpha_selection.csv")
fig, ax = plt.subplots(figsize=(7, 3.4))
for j, y in enumerate(["Sharpe 2022", "Sharpe 2023", "Sharpe 2024"]):
    ax.plot(al.alpha, al[y], marker="o", color=plt.cm.Greys(0.4 + 0.2 * j), label=y.replace("Sharpe ", ""))
ax.plot(al.alpha, al["三年平均 Sharpe"], marker="o", color=BLUE, lw=2.2, label="三年平均")
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(al.alpha); ax.set_xlabel("α"); ax.set_ylabel("扣成本 Sharpe")
ax.set_title("α 越大 Sharpe 越低 → 選 α = 0（策略 D，2022–2024）", loc="left", fontsize=11); ax.legend(fontsize=8, frameon=False, ncol=4)
save(fig, "v7_fig4_alpha.png")

# ---------- 圖 5：淨值曲線（兩個期間，各自從 1 開始） ----------
nav = pd.read_csv(M2 / "nav_curves.csv", index_col=0, parse_dates=True)
panel = pd.read_parquet(PROC / "panel.parquet"); c50 = panel[(panel.sid == "0050") & panel.valid].set_index("date").aclose
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, per in zip(axes, ["2022–2024 驗證期", "2025–2026 事後評估"]):
    a = nav[f"{per}|A"].dropna(); b = nav[f"{per}|B"].dropna(); r = c50.reindex(a.index).ffill()
    ax.plot(r / r.iloc[0], color=LIGHT, ls="--", lw=1.2, label="0050（參考）")
    ax.plot(a / a.iloc[0], color=GREY, lw=1.4, label="A：HV20")
    ax.plot(b / b.iloc[0], color=BLUE, lw=1.8, label="B：Model 1")
    for s, col in (("A", GREY), ("B", BLUE)):
        x = nav[f"{per}|{s}"].dropna(); ax.text(x.index[-1], x.iloc[-1] / x.iloc[0], f" {x.iloc[-1] / x.iloc[0]:.2f}", color=col, fontsize=9, va="center")
    ax.axhline(1, color="black", lw=0.5); ax.set_title(per, loc="left", fontsize=11); ax.set_ylabel("淨值（期初 = 1）")
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("淨值：B 在兩段期間都略高於 A；2022–2024 兩者都落後 0050", x=0.01, ha="left", fontsize=12)
save(fig, "v7_fig5_nav.png")

# ---------- 圖 6：回撤 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, per in zip(axes, ["2022–2024 驗證期", "2025–2026 事後評估"]):
    for s, col in (("A", GREY), ("B", BLUE)):
        x = nav[f"{per}|{s}"].dropna(); dd = x / x.cummax() - 1
        ax.plot(dd, color=col, lw=1.3, label=f"{s}（最大 {dd.min():.1%}）")
    ax.set_title(per, loc="left", fontsize=11); ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.set_ylabel("回撤")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
save(fig, "v7_fig6_drawdown.png")

# ---------- 圖 7：每週股票總曝險 a ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, yr, per in zip(axes, ["2022", "2025"], ["2022–2024 驗證期", "2025–2026 事後評估"]):
    for s, col in (("A", GREY), ("B", BLUE)):
        w = pd.read_csv(M2 / f"weekly_{s}_{yr}.csv", parse_dates=["date"]); ax.step(w.date, w.a, color=col, lw=1.1, where="post", label=f"{s}（平均 {w.a.mean():.0%}）")
    ax.set_ylim(0, 1.05); ax.set_title(per, loc="left", fontsize=11); ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.set_ylabel("股票曝險 a")
fig.suptitle("股票曝險：預測波動高於目標 20.9% 時自動降低，其餘放現金", x=0.01, ha="left", fontsize=12)
save(fig, "v7_fig7_exposure.png")

# ---------- 圖 8：目標波動率的來源（策略 A 2017–2020 的 σ̂ₚ 分布） ----------
ta = pd.read_csv(M2 / "target_strategyA_2017_2020.csv"); fr = json.load(open(M2 / "frozen_rules.json"))
fig, ax = plt.subplots(figsize=(7, 3.4)); ax.hist(ta[~ta.exception].sigma_p, bins=30, color=LIGHT)
for v, lab, col in ((fr["sigma_target"], "中位數 = 目標", BLUE), (fr["sens_targets"]["p25"], "P25", GREY), (fr["sens_targets"]["p75"], "P75", GREY)):
    ax.axvline(v, color=col, lw=1.5 if col == BLUE else 1, ls="-" if col == BLUE else ":"); ax.text(v, ax.get_ylim()[1] * 0.92, f" {lab}\n {v:.1%}", color=col, fontsize=8)
ax.set_xlabel("策略 A 每週預測的組合波動（年化）"); ax.set_ylabel("週數")
ax.set_title(f"目標波動率 = 2017–2020 中位數 {fr['sigma_target']:.1%}（195 個正常週）", loc="left", fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); save(fig, "v7_fig8_target.png")

# ---------- 圖 9：敏感度（Sharpe） ----------
s = pd.read_csv(M2 / "sensitivity.csv"); base = pd.read_csv(M2 / "backtest_A_to_D.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
order = ["主設定"] + list(dict.fromkeys(s["設定"]))
for ax, per in zip(axes, ["2022–2024 驗證期", "2025–2026 事後評估"]):
    for k, (st, col) in enumerate((("A", GREY), ("B", BLUE))):
        v = [base[(base.期間 == per) & (base.策略 == st)].Sharpe.iloc[0]] + [s[(s.設定 == o) & (s.期間 == per) & (s.策略 == st)].Sharpe.iloc[0] for o in order[1:]]
        ax.scatter(v, np.arange(len(order)) + (k - 0.5) * 0.25, color=col, s=28, label=st, zorder=3)
    ax.set_yticks(range(len(order)), order); ax.set_title(per, loc="left", fontsize=11); ax.set_xlabel("扣成本 Sharpe")
axes[0].set_ylim(len(order) - 0.5, -0.5); axes[0].legend(frameon=False, fontsize=8)  # 主設定放最上面（兩圖共用 y 軸，只設定一次）
sp = s.pivot_table(index=["設定", "期間"], columns="策略", values="Sharpe"); nb = int((sp.B > sp.A).sum())  # 計算 B 勝過 A 的設定數
fig.suptitle(f"敏感度：{len(sp)} 個「設定 × 期間」中，B（藍）有 {nb} 個高於 A（灰）", x=0.01, ha="left", fontsize=12)
save(fig, "v7_fig9_sensitivity.png")
