# 06b_compare_versions.py — 驗證期（2022–2024）v1 vs v2 特徵組比較：預測指標、回測績效、風險等級
import numpy as np, pandas as pd                             # 載入：數值運算、表格
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # 畫圖（不開視窗）
from common import ROOT, TAB, VTAB, VFIG, VER, setup_fonts   # 載入：v1 表格（TAB）與目前版本表格（VTAB）
setup_fonts()                                                # 中文字型
assert VER != "v1", "請在 setup.yaml 設 features.active: v2 後執行"  # 只在 v2 時有意義

pool = pd.read_csv(VTAB / "validation_pooled.csv", index_col=0)  # 同一次執行中 v1（重算）與 v2 的合併指標
pc = pd.read_csv(VTAB / "paired_comparison.csv")             # 成對比較（含 v1 → v2）
pred = pd.DataFrame({                                        # 預測指標比較表
    "v1（15 項）": pool.loc[["ridge_v1", "lgbm_v1"], ["MAE", "RMSE", "IC", "TS_corr_level"]].set_axis(["Ridge", "LightGBM"]).stack(),
    "v2（17 項）": pool.loc[["ridge", "lgbm"], ["MAE", "RMSE", "IC", "TS_corr_level"]].set_axis(["Ridge", "LightGBM"]).stack()}).unstack()  # 兩版並列
pair = pc[pc["from"].isin(["ridge_v1", "lgbm_v1"])][["from", "to", "dMAE", "dMAE_lo", "dMAE_hi", "dIC", "dIC_lo", "dIC_hi"]]  # v1 → v2 差異與 HAC 區間
b1 = pd.read_csv(TAB / "backtest_validation_perf.csv", index_col=0)  # v1 回測（原始結果；選定參數未變，故仍有效）
b2 = pd.read_csv(VTAB / "backtest_validation_perf.csv", index_col=0)  # v2 回測
S = ["RIDGE_dyn_EW", "RIDGE_dyn_IV", "LGBM_dyn_EW", "LGBM_dyn_IV", "HV_dyn_EW", "EW100_股票等權", "Passive_0050"]  # 比較的策略
bt = pd.concat({"v1": b1.loc[S, ["Sharpe", "CAGR", "MDD", "turnover_2way", "avg_stock_w"]],
                "v2": b2.loc[S, ["Sharpe", "CAGR", "MDD", "turnover_2way", "avg_stock_w"]]}, axis=1)  # 兩版並列
t1 = pd.read_csv(TAB / "tier_log_val_primary.csv").tier.value_counts(); t2 = pd.read_csv(VTAB / "tier_log_val_primary.csv").tier.value_counts()  # 主策略風險等級次數
tiers = pd.DataFrame({"v1": t1, "v2": t2}).reindex(["低", "中", "高"])  # 並列
with pd.ExcelWriter(VTAB / "v1_vs_v2_validation.xlsx") as w:  # 寫成一個 Excel
    pred.round(4).to_excel(w, sheet_name="預測指標"); pair.round(4).to_excel(w, sheet_name="v1到v2差異", index=False)  # 預測
    bt.round(4).to_excel(w, sheet_name="回測績效"); tiers.to_excel(w, sheet_name="主策略風險等級")  # 回測與等級
print(pred.round(4)); print(pair.round(4)); print(bt.round(3)); print(tiers)  # 印出

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))             # 圖：三個面向
x = np.arange(2); w = .36                                    # 位置
for ax, met, lab in zip(axs[:2], ["MAE", "TS_corr_level"], ["MAE（越低越好）", "每日平均水準相關（越高越好）"]):  # 預測兩項
    ax.bar(x - w/2, pred[("v1（15 項）", met)], w, color="#9ec5f0", label="v1"); ax.bar(x + w/2, pred[("v2（17 項）", met)], w, color="#2a78d6", label="v2")  # 長條
    ax.set_xticks(x); ax.set_xticklabels(["Ridge", "LightGBM"]); ax.set_title(lab); lo = min(pred.xs(met, axis=1, level=1).min()) * .95; ax.set_ylim(lo, None)  # 座標
s2 = ["RIDGE_dyn_EW", "LGBM_dyn_EW"]; axs[2].bar(x - w/2, b1.loc[s2, "Sharpe"], w, color="#9ec5f0", label="v1"); axs[2].bar(x + w/2, b2.loc[s2, "Sharpe"], w, color="#2a78d6", label="v2")  # 回測 Sharpe
axs[2].axhline(0, color="#8a8984", lw=.8); axs[2].set_xticks(x); axs[2].set_xticklabels(["Ridge 動態", "LightGBM 動態"]); axs[2].set_title("回測淨 Sharpe"); axs[2].legend(frameon=False)  # 座標
fig.suptitle("驗證期 2022–2024：v2 特徵讓預測更準、整體水準判斷更好，但回測 Sharpe 沒有改善", x=.01, ha="left", fontsize=12)  # 總標題
fig.tight_layout(); fig.savefig(VFIG / "v1_vs_v2_validation.png", dpi=200, bbox_inches="tight")  # 存圖
