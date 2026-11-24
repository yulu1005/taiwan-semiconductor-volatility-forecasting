# 09_excel.py — 把主要表格彙整成一個 Excel 檔，方便貼進簡報與報告
import pandas as pd                                          # 載入：表格
from common import ROOT, TAB                                 # 載入：路徑
SHEETS = [("00_總表", "headline_table.csv", "總表：驗證 vs 封存（訊號、投組、成本、風險覆蓋、推論）"),
          ("01_PIT檢查", "pit_checklist.csv", "時間點正確性三層檢查"),
          ("02_資料品質", "data_quality_by_stock.csv", "每檔資料品質：缺值、零量、OHLC 異常、跳動、公司行動"),
          ("03_剔除紀錄", "data_quality_log.csv", "上市前興櫃資料剔除"),
          ("04_異常跳動", "unexplained_jumps.csv", "還原後超過 ±10.5% 的跳動與說明"),
          ("05_每期合格股數", "coverage_eligible_names.csv", "每個決策日合格股票數"),
          ("06_特徵篩選", "triage_table.csv", "五道關卡 triage（訓練期 2016-11～2021）"),
          ("07_特徵高相關", "feature_high_corr_pairs.csv", "|Spearman|>0.8 的特徵配對"),
          ("08_有效樣本", "effective_sample.csv", "名目與有效樣本"),
          ("09_參數試驗", "model_grid_validation.csv", "所有模型設定的三年平均驗證指標"),
          ("10_分年度預測", "validation_by_year.csv", "驗證期每年 MAE/RMSE/IC"),
          ("11_驗證合併", "validation_pooled.csv", "驗證期三年合併"),
          ("12_分股票MAE", "validation_mae_by_stock.csv", "驗證期每檔 MAE"),
          ("13_成對比較_驗證", "paired_comparison.csv", "每一階 vs 前一階：ΔMAE、ΔIC（HAC 95%）"),
          ("14_回看期穩定", "lookback_robustness.csv", "10/20/40 日回看期"),
          ("15_SHAP穩定度", "shap_stability_folds.csv", "LightGBM 各 fold SHAP 重要度與方向"),
          ("16_預測區間", "conformal_coverage.csv", "名目 90% 預測區間實際涵蓋率與寬度"),
          ("17_IC每年", "ic_by_year_hac.csv", "每年 IC 與 HAC 區間"),
          ("18_封存預測", "holdout_prediction_metrics.csv", "封存期預測指標"),
          ("19_成對比較_封存", "holdout_paired_comparison.csv", "封存期成對比較"),
          ("20_風險門檻", "tier_thresholds_validation.csv", "各年風險門檻（訓練期 P33/P67）"),
          ("21_風險等級次數", "tier_counts.csv", "主策略低/中/高風險次數"),
          ("22_回測_驗證", "backtest_validation_perf.csv", "驗證期全部策略與變體績效"),
          ("23_回測_封存", "backtest_holdout_perf.csv", "封存期策略績效"),
          ("24_成本掃描_驗證", "cost_sweep_validation.csv", "成本 0～4 倍"),
          ("25_成本掃描_封存", "cost_sweep_holdout.csv", "成本 0～4 倍"),
          ("26_兩平成本", "breakeven_both.csv", "毛/淨 Sharpe 與兩平成本"),
          ("27_Sharpe差異_驗證", "sharpe_diff_bootstrap_validation.csv", "週區塊重抽樣 95% 區間"),
          ("28_Sharpe差異_封存", "sharpe_diff_bootstrap_holdout.csv", "週區塊重抽樣 95% 區間"),
          ("29_市場狀態", "regime_slice.csv", "波動 × 趨勢狀態切片"),
          ("30_VaR_Beta", "risk_var_cvar_beta.csv", "VaR/CVaR、高波動 VaR、60 日 beta"),
          ("31_壓力測試", "stress_tests.csv", "2022 升息、2025/4 關稅；滑價 ×3"),
          ("32_風險控制矩陣", "risk_control_matrix.csv", "風險、代理指標、控制、頻率"),
          ("33_公司行動", "corporate_actions.csv", "除權息、減資、分割事件與因子")]
with pd.ExcelWriter(ROOT/"outputs"/"MLQT_group5_results.xlsx") as w:  # 開 Excel 檔
    pd.DataFrame([(s, d) for s, _, d in SHEETS], columns=["工作表", "內容"]).to_excel(w, sheet_name="目錄", index=False)  # 目錄頁
    for s, f, _ in SHEETS:                                   # 逐一表格
        t = pd.read_csv(TAB/f, encoding="utf-8-sig")         # 讀 CSV
        t.to_excel(w, sheet_name=s, index=False)             # 寫入工作表
    pd.read_csv(ROOT/"run_log.csv").to_excel(w, sheet_name="研究帳本run_log", index=False)  # 研究帳本
print("ok")                                                  # 完成
