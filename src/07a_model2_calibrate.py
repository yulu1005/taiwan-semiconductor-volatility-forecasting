# 07a_model2_calibrate.py — Model 2 目標波動校準：只用 2021 校準年（與訓練期 HV 參考），不碰 2022–2024 驗證期
#   規則（執行前寫定）：目標波動 = 0.6 × 2021 每週「股票組合預測波動」的中位數，取到整數 %，讓股票曝險平常約 60%
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC, VTAB                           # 載入：設定與路徑
from models import load                                      # 載入：樣本
from model2 import weekly                                    # 載入：每週配置
P2 = dict(corr_window=60, corr_min_obs=40, shrink=True, alpha=0.0, name_cap=0.30, adv_cap=0.05,
          capital=CFG["execution"]["capital_twd"], max_exposure=1.0, target_vol=None)  # 校準時先不設目標
panel = pd.read_parquet(PROC / "panel.parquet")              # 日資料
op = panel.pivot(index="date", columns="sid", values="aopen").where(panel.pivot(index="date", columns="sid", values="valid").fillna(False).astype(bool))  # 還原開盤價（無效日空值）
rets = np.log(op / op.shift(1))                              # 開盤到開盤日報酬（與標籤同口徑）
df = load()                                                  # 合格樣本（含 rv20、adv20、tradable、每週決策日）
pred = pd.read_parquet(PROC / "oos_validation_v4.parquet")   # Model 1 樣本外預測
prim = pd.read_csv(VTAB / "selected_configs.csv", index_col=0).config["primary"]  # 主模型（Lasso）
ml = pred[(pred.fold == "calib2021") & (pred.config == prim)].merge(df[["date", "sid", "adv20", "tradable"]].rename(columns={"tradable": "tr2"}), on=["date", "sid"])  # 2021 預測
ml = ml[ml.tr2.fillna(False).astype(bool)]                   # 只用可交易股票
hv = df[df.is_decision_wk.fillna(False).astype(bool) & df.tradable.fillna(False).astype(bool) & df.date.between("2017-01-01", "2021-12-31")]  # 訓練期每週決策日
def run(tab, col):                                           # 定義：逐週算配置診斷
    out = []                                                 # 收集
    for d, g in tab.groupby("date"):                         # 每個決策日
        g = g.dropna(subset=[col, "adv20"])                  # 去缺值
        w, info = weekly(d, g.sid.tolist(), g[col].values, g.adv20.values, rets, P2)  # 配置
        out.append(dict(date=d, **info))                     # 記錄
    return pd.DataFrame(out)                                 # 回傳
A = run(ml, "p"); B = run(hv, "rv20")                        # 2021 用 ML 預測；2017–2021 用 20 日歷史波動當參考
for name, T in [("2021 ML 預測", A), ("2017–2021 HV20（參考）", B)]:  # 印出兩組
    print(name, "週數", len(T), "可交易檔數", T.n.min(), "–", T.n.max(), "| 上限加總 < 1 的週數", int((T.cap_sum < 1).sum()))
    print("  股票組合預測波動 百分位 10/25/50/75/90：", np.round(T.sigma_p.quantile([.1, .25, .5, .75, .9]).values * 100, 1))
    print("  收縮強度 λ 中位數", round(T.lam.median(), 2), "| 風險預算最大偏差 中位數", round(T.rb_dev.median(), 4), "| 最大內部權重 中位數", round(T.q_max.median(), 3))
tv = round(0.6 * A.sigma_p.median(), 2)                      # 事先寫定的規則：0.6 × 2021 中位數，取到 1%
print("目標波動 =", tv)                                       # 印出
for name, T in [("2021", A), ("2017–2021 HV", B)]:           # 套用目標後的曝險分布
    a = np.minimum(1.0, tv / T.sigma_p)                      # 股票曝險
    print(name, "曝險 百分位 10/25/50/75/90：", np.round(a.quantile([.1, .25, .5, .75, .9]).values * 100, 0), "| 滿倉週比例", round((a >= 1).mean(), 3))
A.assign(src="ml2021").to_csv(VTAB / "model2_calib_2021.csv", index=False); B.assign(src="hv2017_2021").to_csv(VTAB / "model2_calib_hv.csv", index=False)  # 存檔
