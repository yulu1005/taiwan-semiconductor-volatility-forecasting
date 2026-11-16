# 台灣半導體股價波動度預測

本專案以台灣半導體股票未來 5 個交易日的波動度為研究目標。目前可執行第 1 階段資料可行性分析，第 2 至 8 階段為待實作的骨架。

## 第 1 階段資料可行性分析

在專案目錄執行下列指令。此階段只使用 Python 標準函式庫。

```powershell
python 01_feasibility_analysis.py
```

若只要核對候選股票，加入 `--pool-only` 即可離線產生股票池。使用 `--start YYYY-MM-DD --end YYYY-MM-DD` 可指定包含首尾日期的下載區間。

- `results/tables/stock_pool.csv` 記錄全部 15 檔候選股的證交所產業分類、上市日、暫定研究分組、業務說明與來源。`in_scope` 空白表示最終研究資格待確認，所有候選股均接受行情品質檢查。
- `data/raw/` 依股票及研究期間儲存每日開、高、低、收及成交股數，價格為未還原行情。預設使用 FinMind 的 `TaiwanStockPrice`（原始上市交易資訊由臺灣證券交易所提供、由 FinMind 處理），也可用 `--source twse` 直接查詢證交所。
- `results/tables/feasibility_report.csv` 彙整實際日期範圍、缺漏交易日、缺失欄位、重複日期及異常值，並附上初篩判定。行情連線失敗會保留錯誤原因，程式會回傳非零結束碼。

研究期間為 2016-11-01 至 2026-11-01，初篩要求至少 500 個有效交易日及 95% 的覆蓋率。覆蓋率以每檔股票首末行情之間的台積電（2330）行情日期為分母，上市前交易日與上市後的首尾缺口另外列出。有效日需有唯一日期、合理的開高低收及正成交量。報告另列首末有效行情日，作為各股可用研究期間。分析採證交所上市行情，起點不早於該股證交所上市日。

相鄰有效行情的原始收盤價變動超過 20% 會列為人工複核。兩筆行情之間有參考交易日缺口時，另列為跨期變動。報告中的 `pass` 表示通過資料品質初篩，`review` 需要人工核對，`insufficient` 表示樣本或覆蓋不足，`download_failed` 表示下載失敗。產業資格由股票池另行審核。

FinMind 以股票與已結束的查詢區間快取至 `data/raw/.price_cache/finmind/`。直接查詢證交所時，每個已結束月份成功下載並驗證後會存入 `data/raw/monthly_cache/<股票代碼>/YYYY-MM.json`，原有 `.twse_monthly_cache/` 中的有效月份也可重用。證交所每月最多嘗試三次，空資料或異常回應不寫入快取。若下載失敗，報告會標示錯誤，重跑可從已下載月份接續。

### 異常資料審核

執行 `python 01b_data_quality_review.py`，離線讀取已下載的行情並輸出 `results/tables/data_quality_review.csv`。明細包含原始 OHLCV、前後有效收盤、缺口交易日數及查核來源。`gap_sessions` 是兩個比較端點之間的參考交易日數，0 表示連續交易日。

研究資料保留原始行情及停牌缺口。後續以經股息、除權、減資與分割調整的收盤價計算對數報酬，先抽核公司行動調整因子；僅在兩端皆有有效行情且為相鄰參考交易日時建立單日報酬。零價行情與跨停牌的報酬不填零，跨缺口的五日視窗留為缺失。調整因子可對照[證交所除權息](https://wwwc.twse.com.tw/zh/announcement/ex-right/twt49u.html)及[減資恢復買賣參考價](https://www.twse.com.tw/exchangeReport/TWTAUU?response=html)。

## 調整後日報酬與五日波動率

```powershell
python test_yahoo.py
python 01c_prepare_adjusted_prices.py
python 02_labels.py
```

Yahoo 下載輸出 `data/external/adjusted_close.csv`、含空價格的 `yahoo_daily.csv` 及逐檔下載稽核。01c 以台積電原始行情日期為共同日曆，以各股上市日與原始 OHLCV 判定每日可用性。01c 輸出 `data/processed/adjusted_daily_returns.csv`，另有 `adjusted_price_audit.csv` 與 `adjusted_price_discrepancies.csv` 記錄分股涵蓋率所需的計數、收盤價差異、公司行動與待查日期。Yahoo Close 可能使用不同的歷史價格尺度，固定比例差異留供核對。報酬採 Yahoo Adj Close 的日比率。跨缺口或調整後單日變動超過 20% 的報酬留空。

02 讀取日報酬檔，對日期 t 計算 `sqrt((252 / 5) * sum(r[t+i] ** 2 for i in 1..5))`。此為未來五個交易日平方對數報酬的年化實現波動率，公式不扣除平均報酬。五個參考交易日都須有有效報酬。輸出為 `data/processed/labels.csv` 及 `results/tables/label_audit.csv`，波動率以小數表示。大於 1（100%）的值列入統計，供檢查高波動視窗。

Yahoo 歷史調整價格可能隨後續公司行動回溯修訂。建立 Feature 前須核對當時可得的資料版本與調整因子，Label 欄位只作為預測目標。

## 特徵稽核與樣本資格

`python 03_financial_features.py` 產出 15 項特徵及 `results/tables/feature_audit.csv`。`RSI_14` 以固定 14 筆有效價格變動的簡單平均漲跌幅計算，平盤視窗為 50。成交量與相對成交金額沿用原始行情檢查，即使該日調整後價格經補值，仍可使用通過檢查的原始成交資料。

ATR 稽核的 `TR_*` 欄位以單日為單位，每個上市後交易日歸入一個原因。若同時有多項異常，依程式檢查順序記下第一項。`ATR_window_*` 欄位以 14 日 ATR 視窗為單位，區分初始歷史不足與視窗內存在無效 TR。

`data/processed/model_dataset.csv` 是 Purging 後 Train、Validation、Test 的合併檔。Purging 前的候選列可在 `merged_feature_labels.csv` 以 `valid=1` 查詢。05 階段從 `train.csv` 擬合，使用 `validation.csv` 選擇模型與調整參數。

執行 `python 04_feature_evaluation.py` 會重建完整合併檔 `merged_feature_labels.csv`、篩選後的 `model_dataset.csv`，以及 Train（2016/11/01–2021/12/31）、Validation（2022/01/01–2023/12/31）、Test（2024/01/01–2026/11/01）三份 CSV。各組以預測日期切分，並排除 Label 結束日超過組別截止日的樣本。Feature 以當日收盤後可得為研究假設，供後續配置與回測決定可交易時點。

`dataset_audit.csv` 記錄合併與排除數量。`feature_coverage.csv` 以完整合併資料中的上市後交易日計算有效比例。`feature_correlation.csv` 與 `feature_feature_correlation.csv` 都以 Purging 後的 Train 計算 Pearson 與 Spearman。單一特徵線性迴歸只在 Train 擬合，預測限制為非負，再與直接使用 `Volatility_5` 的基準在相同 Validation 樣本比較。Test 只輸出與稽核樣本資格，未參與相關性或 MAE 評估。三份切分檔中的 Label 起訖日是切分稽核欄位，模型輸入取 15 個 Feature 欄位。

每個單一特徵 OLS 都將所有 Train 股票合併擬合一組係數，MAE 衡量這個跨股票共同模型的表現。若要做正式顯著性檢定，需處理同日股票之間的相關性，以及重疊五日 Label 帶來的時間相依性。

## 模型訓練與選擇

在專案虛擬環境安裝 `requirements.txt`，執行 `python 05_model_training.py`。05 讀取已切分及 Purging 的 Train、Validation。特徵組 A 含 15 項特徵，B 移除 `Turnover`，兩組使用相同樣本。Ridge 的標準化與迴歸係數一起擬合。LightGBM 最多 500 棵樹，以 Validation 的非負預測 MAE 早停，耐心值為 40 輪。

在 Ridge 與 LightGBM 中依 Validation MAE、RMSE、試驗編號依序選擇模型，基準模型供比較。選定後以 Train＋Validation 重新擬合，LightGBM 迭代次數固定為選擇階段的最佳值。Validation 報表保留重訓前的預測。Test 保留給 06。

06 可用 `joblib.load()` 載入 `models/final_model.joblib`，按同目錄 `model_metadata.json` 的 `features` 順序傳入特徵，並將預測下限設為 0。也可直接沿用 `05_model_training.py` 的 `predict_volatility()`。

## 分析流程

| 腳本 | 用途 |
| --- | --- |
| `01_feasibility_analysis.py` | 檢查股價資料的涵蓋期間、完整性與可用性 |
| `02_labels.py` | 建立波動度預測目標 |
| `03_financial_features.py` | 建立技術指標特徵 |
| `04_feature_evaluation.py` | 評估特徵的預測能力 |
| `05_model_training.py` | 訓練波動度預測模型 |
| `06_model_evaluation.py` | 評估模型預測績效 |
| `07_portfolio_construction.py` | 依模型預測配置投資組合 |
| `08_backtest.py` | 回測投資組合策略 |

## 資料與輸出

將原始股價放在 `data/raw/`，清理後的資料放在 `data/processed/`。特徵與訓練完成的模型分別存放於 `features/` 與 `models/`，圖表與表格則分別放在 `results/figures/` 與 `results/tables/`。

`config.py` 集中定義專案路徑與研究參數。`requirements.txt` 可在實作後續階段時補上所需套件。
