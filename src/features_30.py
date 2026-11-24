# features_30.py — 組員提出的 30 項特徵（6 類），全部只用過去 20 個交易日以內的資料，寫入 data/processed/features_30.parquet
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC                                 # 載入：設定與路徑
from features import rmean, rsum, rstd                       # 載入：移動平均、加總、標準差（窗內有空值就給空值）

N = 20                                                       # 主要回看期：20 個交易日
F30 = ["Volatility_5", "Volatility_10", "Volatility_20", "Volatility_Ratio_5_20", "EWMA_Volatility_20",
       "Volatility_of_Volatility_20", "Stock_Volatility_Shock",                       # A. 歷史波動與持續性
       "Return_1", "Return_5", "Return_20", "Downside_Semivariance_20", "Max_Absolute_Return_20",  # B. 報酬衝擊與不對稱性
       "Parkinson_Volatility_20", "NATR_14", "Close_Open_Range", "Volume_Ratio_20", "VolumeCV_20",  # C. 價格區間與成交量
       "Market_Volatility_20", "Market_Return_5", "Market_Beta_20", "Market_Correlation_20",       # D. 市場共同風險
       "Idiosyncratic_Volatility_20", "Systematic_Volatility_20", "Idiosyncratic_Volatility_Ratio",
       "Relative_Volatility_20", "Cross_Sectional_Volatility_Rank",                    # E. 個股特有與相對風險
       "Volatility_5_10_Ratio", "Upside_Semivariance_20", "Negative_Return_Ratio_20", "Market_Volatility_Ratio_5_20"]  # F. 波動結構與市場狀態

def vol(r, n): return np.sqrt(252 / n * rsum(r ** 2, n))     # 定義：n 日年化實現波動（與標籤同算法）

def stock30(g, rm):                                          # 定義：單一股票的 30 項特徵（g＝依日期排序的單檔資料，rm＝0050 日報酬）
    C, O, H, L = g.aclose, g.aopen, g.ahigh, g.alow          # 還原後收、開、高、低
    r = np.log(C / C.shift(1))                               # 每日收盤對數報酬
    f = pd.DataFrame(index=g.index)                          # 準備空表
    # --- A. 歷史波動與持續性 ---
    for n in (5, 10, 20): f[f"Volatility_{n}"] = vol(r, n)   # 5、10、20 日波動
    f["Volatility_Ratio_5_20"] = f.Volatility_5 / f.Volatility_20  # 短期 ÷ 中期波動（>1 代表最近變熱）
    ew = (r ** 2).ewm(span=N, adjust=False, min_periods=N).mean()  # 報酬平方的指數加權平均（越近的日子權重越大）
    f["EWMA_Volatility_20"] = np.sqrt(252 * ew)              # EWMA 波動（span=20）
    f["Volatility_of_Volatility_20"] = rstd(r.abs(), N) / rmean(r.abs(), N)  # 每日漲跌幅的變異係數（波動本身穩不穩）
    f["Stock_Volatility_Shock"] = r.abs() / (f.Volatility_20 / np.sqrt(252))  # 今天的漲跌是近期日波動的幾倍（衝擊大小）
    # --- B. 報酬衝擊與不對稱性 ---
    f["Return_1"] = r                                        # 1 日報酬
    f["Return_5"] = np.log(C / C.shift(5))                   # 5 日報酬
    f["Return_20"] = np.log(C / C.shift(N))                  # 20 日報酬
    f["Downside_Semivariance_20"] = 252 / N * rsum(r.clip(upper=0) ** 2, N)  # 下方半變異數（只算下跌日，年化變異數）
    f["Max_Absolute_Return_20"] = r.abs().rolling(N, min_periods=N).max()  # 20 日內最大單日漲跌幅
    # --- C. 價格區間與成交量 ---
    f["Parkinson_Volatility_20"] = np.sqrt(252 / (4 * np.log(2)) * rmean(np.log(H / L) ** 2, N))  # Parkinson 波動（高低價）
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1, skipna=False)  # 真實波幅
    f["NATR_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / C  # 正規化 ATR（Wilder 平滑 14 日 ÷ 收盤價）
    f["Close_Open_Range"] = (C - O) / O                      # 當日開盤到收盤的漲跌
    f["Volume_Ratio_20"] = g.avolume / rmean(g.avolume, N)   # 成交量 ÷ 20 日平均
    f["VolumeCV_20"] = rstd(g.avolume, N) / rmean(g.avolume, N)  # 成交量變異係數
    # --- D. 市場共同風險（0050） ---
    mv20 = vol(rm, N)                                        # 大盤 20 日波動
    f["Market_Volatility_20"] = mv20                         # 大盤 20 日波動
    f["Market_Return_5"] = rm.rolling(5, min_periods=5).sum()  # 大盤 5 日報酬（對數報酬可相加）
    cov = rmean(r * rm, N) - rmean(r, N) * rmean(rm, N)      # 與大盤的 20 日共變異（母體）
    var_m = rmean(rm ** 2, N) - rmean(rm, N) ** 2            # 大盤 20 日變異（母體）
    var_s = rmean(r ** 2, N) - rmean(r, N) ** 2              # 個股 20 日變異（母體）
    beta = cov / var_m                                       # beta
    f["Market_Beta_20"] = beta                               # 對大盤的敏感度
    f["Market_Correlation_20"] = cov / np.sqrt(var_m * var_s)  # 與大盤的相關係數
    # --- E. 個股特有與相對風險 ---
    idio_var = (var_s - beta ** 2 * var_m).clip(lower=0)     # 個股特有變異 = 總變異 − 大盤解釋的部分
    f["Idiosyncratic_Volatility_20"] = np.sqrt(252 * idio_var)  # 個股特有波動（年化）
    f["Systematic_Volatility_20"] = np.sqrt(252 * beta ** 2 * var_m)  # 系統性波動（大盤帶來的部分）
    f["Idiosyncratic_Volatility_Ratio"] = np.sqrt(idio_var / var_s)  # 特有波動占總波動的比例
    f["Relative_Volatility_20"] = f.Volatility_20 / mv20     # 個股波動 ÷ 大盤波動
    # Cross_Sectional_Volatility_Rank 需要同一天所有股票，於主程式計算
    # --- F. 波動結構與市場狀態 ---
    f["Volatility_5_10_Ratio"] = f.Volatility_5 / f.Volatility_10  # 5 日 ÷ 10 日波動
    f["Upside_Semivariance_20"] = 252 / N * rsum(r.clip(lower=0) ** 2, N)  # 上方半變異數（只算上漲日）
    f["Negative_Return_Ratio_20"] = (r < 0).astype(float).where(r.notna()).rolling(N, min_periods=N).mean()  # 20 日中下跌天數比例
    f["Market_Volatility_Ratio_5_20"] = vol(rm, 5) / mv20    # 大盤短期 ÷ 中期波動
    return f                                                 # 回傳（索引為日期）

if __name__ == "__main__":                                   # 直接執行時：計算全部股票並存檔
    STK, BM = CFG["universe"]["stocks"], CFG["universe"]["passive_benchmark"]  # 股票與 0050
    panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])  # 清理後日資料
    panel["avolume"] = (panel.volume / panel.adj).where(panel.valid)  # 還原成交量（無效日空值）
    b = panel[panel.sid == BM].set_index("date"); rm = np.log(b.aclose / b.aclose.shift(1))  # 0050 日報酬
    X = pd.concat([stock30(g.set_index("date"), rm.reindex(g.set_index("date").index)).assign(sid=sid).reset_index()
                   for sid, g in panel[panel.sid.isin(STK)].groupby("sid")])  # 各檔特徵
    elig = pd.read_parquet(PROC / "samples.parquet", columns=["date", "sid", "eligible"])  # 合格標記
    MK = ["Market_Volatility_20", "Market_Return_5", "Market_Volatility_Ratio_5_20"]  # 市場層級特徵
    mk = pd.DataFrame({"Market_Volatility_20": vol(rm, N), "Market_Return_5": rm.rolling(5, min_periods=5).sum()})  # 用 0050 自己的交易日曆計算（個股停牌不影響）
    mk["Market_Volatility_Ratio_5_20"] = vol(rm, 5) / mk.Market_Volatility_20  # 大盤短期 ÷ 中期波動
    X = X.drop(columns=MK).merge(mk.rename_axis("date").reset_index(), on="date", how="left")  # 換成同一天所有股票同值的版本
    X["Relative_Volatility_20"] = X.Volatility_20 / X.Market_Volatility_20  # 相對波動也用新的大盤波動
    X = X.merge(elig, on=["date", "sid"], how="left")        # 合併
    X["Cross_Sectional_Volatility_Rank"] = X[X.eligible.fillna(False)].groupby("date").Volatility_20.rank(pct=True)  # 當天合格股票中的波動百分位
    X = X.drop(columns="eligible")[["date", "sid"] + F30]    # 欄位順序
    X.to_parquet(PROC / "features_30.parquet", index=False)  # 寫出
    print(X.shape, "| 缺值比例（合格列）：", X.merge(elig[elig.eligible], on=["date", "sid"])[F30].isna().mean().round(3).max())  # 印出
