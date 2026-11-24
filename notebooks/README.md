# notebooks：依課程範本（ML4T ETF case study 01–07）重做的資料與特徵檢查

| notebook | 對應範本 | 回答的問題 |
|---|---|---|
| `01_feasibility_analysis` | etfs/01 | 每個決策日有幾檔可交易？成本 vs 變動？波動有持續性嗎？fold 有沒有碰到封存期？ |
| `02_labels` | etfs/02 | 標籤公式對嗎？窗口有效性四項斷言、分布、重疊與有效樣本、HV20 基準線 |
| `03_financial_features` | etfs/03 | 時點契約、暖身稽核、扣掉封存期重建比對、尺度、離散度、冗餘、持續性 |
| `04_model_based_features` | etfs/04（GARCH 一節） | 每檔 GARCH(1,1) walk-forward、封存凍結、刪資料不變性、增量 IC |
| `05_evaluation` | etfs/05 | 驗證年度上每項特徵的 IC、HAC、BH、三年一致性、五分位、冗餘、triage ledger |
| `03b_feature_redesign` | （延伸） | 27 個候選特徵只用 2016-11～2020 篩選、2021 比較 → v2 特徵清單 |

執行：先跑完 `src/02`、`src/03`（產生 `data/processed`），再依序開啟 01→05（或 `./run_nb.sh 0*.py`）。
輸出：`outputs/notebooks/`（圖、`triage_ledger.csv`、`eligibility.csv`、`ic_timeseries.parquet`、`garch_fits.csv`）。

## 跟範本 06_linear / 07_gbm 的差距（尚未重跑）

| 範本做法 | 我們目前 | 差距 |
|---|---|---|
| Ridge 懲罰跨 10 個數量級，另有 Lasso、ElasticNet（alpha_frac） | Ridge 0.01–10⁴（7 個數量級），沒有 Lasso／ElasticNet | 補 10⁻⁴–10⁶ 與兩種稀疏懲罰 |
| 每個設定記錄 IC 涵蓋日數（ic_n_days），只比較全涵蓋的設定 | 未記錄 | 補欄位 |
| GBM：葉數 default/7/15/31/63 × 損失 mse/mae/huber，每個設定 10 個 checkpoint | 葉數 7/15 × 葉樣本 50/100 × 樹 100/300，只用 MAE 損失 | 補損失函數與 checkpoint 學習曲線；checkpoint 要算進試驗次數 |
| **選模在回測（驗證期回測 Sharpe），不在 IC** | 用驗證期平均 MAE 選 | 選模準則不同：須在報告說明，或改成依驗證期回測選 |
| 「標籤結束日」切驗證／封存邊界 | 已修正（`models.split`）；v1 重算後選出的參數不變 | 已解決 |
