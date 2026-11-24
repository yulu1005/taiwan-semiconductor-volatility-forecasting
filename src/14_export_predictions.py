# 14_export_predictions.py — 把每週每檔的「預測波動率」整理成一張好讀的表（Excel 也能直接開的 CSV）
#   欄位：決策日、股票、是否可交易、Model 1（Lasso）預測、90% 預測區間、HV20、GARCH、EWMA、實際波動（標籤）
#   預測來源：07_model2_v7.py 存下的逐年重訓預測（2021 起；訓練到前一年底，2025–26 用訓練到 2024 年底的模型）
#   實際波動：未來 5 個交易日開盤到開盤的年化實現波動；研究期末最後幾週還沒有實際值（空白）
import numpy as np, pandas as pd                             # 載入：數值、表格
from common import PROC, VTAB                                # 載入：路徑
from models import load                                      # 載入：合格樣本
q = float(pd.read_csv(VTAB / "model1_meta.csv", index_col=0).loc["qhat", "value"])  # conformal 90% 區間寬度（對數尺度）
ml = pd.read_parquet(PROC / "model2_sigma_ml_v7.parquet")    # Model 1 每週預測
df = load()
panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])
r = np.log(panel.aclose / panel.groupby("sid").aclose.shift())  # EWMA 用收盤報酬（與 05 相同）
panel["ewma"] = np.sqrt(252 * (r ** 2).groupby(panel.sid).transform(lambda s: s.ewm(alpha=0.06, adjust=False, min_periods=20).mean()))
x = ml.merge(df[["date", "sid", "tradable", "rv20", "garch_oo5", "y"]], on=["date", "sid"], how="left").merge(panel[["date", "sid", "ewma"]], on=["date", "sid"], how="left")
out = pd.DataFrame({"決策日": x.date.dt.date, "股票": x.sid, "可交易": x.tradable.fillna(False).astype(bool),
                    "預測波動_Lasso": x.ml, "區間下限_90%": x.ml * np.exp(-q), "區間上限_90%": x.ml * np.exp(q),
                    "HV20": x.rv20, "GARCH": x.garch_oo5, "EWMA": x.ewma, "實際波動_未來5日": x.y})
out["實際落在區間內"] = np.where(out["實際波動_未來5日"].notna(), (out["實際波動_未來5日"] >= out["區間下限_90%"]) & (out["實際波動_未來5日"] <= out["區間上限_90%"]), np.nan)
out = out.sort_values(["決策日", "股票"]).round(4)
out.to_csv(VTAB / "volatility_predictions_weekly.csv", index=False, encoding="utf-8-sig")  # utf-8-sig：Excel 開啟中文不亂碼
t = out[out.可交易]; t = t.assign(年=pd.to_datetime(t.決策日).dt.year)
print("筆數", len(out), "｜可交易", len(t), "｜期間", out.決策日.min(), "～", out.決策日.max())
print(t.groupby("年")[["預測波動_Lasso", "HV20", "實際波動_未來5日"]].mean().round(3).to_string())
