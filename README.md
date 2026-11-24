# 第五組：台灣半導體相關股票的波動率預測與動態資產配置

本專案依研究計畫書 V7，以台灣半導體相關股票為研究對象。Model 1 預測個股未來五個交易日的年化波動率。Model 2 結合個股波動率、股票間相關性與波動率目標，計算股票權重與現金比例，再比較 A–D 四組策略的扣成本回測結果。

## V7 現行版本

`setup.yaml` 的 `features.active` 設為 `v7`。2022–2024 年用於逐年擴張式驗證與選模。2025–2026 年資料曾被早期版本使用，V7 將這段結果列為事後評估。獨立前瞻結果由規則凍結後的紙上交易另行累積。

### 查看現有儀表板

執行 V7 後，在該次執行的資料夾中開啟 `outputs/v7/dashboard/research_dashboard_preview.html`。PowerShell 可用 `Invoke-Item` 開啟：

```powershell
Invoke-Item ".\outputs\v7\dashboard\research_dashboard_preview.html"
```

四頁式儀表板包含資產配置總覽、波動率預測、策略配置與績效，以及研究驗證與診斷。首頁的日期選單使用 2025–2026 事後回測的決策日。原版歷史報表仍可從 `outputs/v7/dashboard/dashboard.html` 開啟。兩份 HTML 皆內嵌資料，可直接用瀏覽器查看。

### 下載資料並在本機重跑

使用 Python 3.12，在包含 `setup.yaml` 的資料夾開啟 PowerShell，依序執行：

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python src/01_download.py
& $python src/01b_listing.py
& $python reproduce_v7.py --use-current-python
```

前兩支資料腳本需要連線至 FinMind 和證交所。若手上已有同一版本的原始資料快照，可將它放入 `data/raw/` 再直接重跑。每次執行會在 `reproductions/` 建立帶時間戳的獨立資料夾。完成後開啟該次產出的儀表板，各步驟的日誌也會保留在該資料夾。

若你的 Python 已裝好 `requirements.txt` 的套件，可加上 `--use-current-python`。若只想確認輸入檔案能成功複製，可加上 `--prepare-only`。

### 特徵篩選與 Model 2 試驗紀錄

- `src/16_feature_selection_v7.py` 只使用 2016 年 11 月至 2020 年底的每週資料，要求標籤也在 2020 年底前結束。它將原本 17 項與組員 30 項合併，依既有的 Spearman > 0.999 規則去重，對個股 IC 的 HAC p 值做 BH 校正，再檢查 2017–2020 每年的 IC 方向，以控制 HV20 後的偏 IC 分層，並剔除高度相關項目。第一層包含 HV20 錨點和通過增量檢定者，第二層依增量 t 值的絕對值補選 8 項。市場層級特徵採每週時間序列 HAC 迴歸。兩種 GARCH 口徑的逐項明細、去重對照、入選名單與摘要位於 `outputs/v7/tables/feature_selection/`。
- `src/07_model2_v7.py` 將驗證期 2 個窗口、4 個 α 的設定與分數寫入當次 `run_log.csv` 和 `outputs/v7/tables/model2/model2_trials.csv`。Deflated Sharpe 依該次六筆試驗計算。策略 B 共用凍結的目標波動率和成本假設，在 2025–2026 比較 20／60 日收縮窗口，結果為 `window_posthoc.csv`。該期間僅作事後評估。

### V7 前瞻紙上交易

以已成功重跑 V7 的獨立工作副本操作。進入該次執行的資料夾，沿用已安裝套件的 Python，先執行 `src/17_paper_v7.py freeze`。產出的 `paper_v7/freeze.json` 記錄模型係數、前處理參數、策略設定和程式雜湊。

```powershell
$python = "..\..\.venv\Scripts\python.exe"
& $python src/17_paper_v7.py refresh
$rate = Read-Host "輸入當日臺銀一年期機動定存年率（小數，例如 1.69% 輸入 0.0169）"
& $python src/17_paper_v7.py capture --cash-rate $rate
& $python src/17_paper_v7.py report
```

從凍結日期之後第一個要納入的每週決策日開始，臺北時間收盤後執行，當天在 `capture` 加上 `--decision`。遇到休市縮短的交易週，確認當天是該週最後一個交易日後，再加 `--short-week`。之後每個交易日都登記行情。程式保留當日行情及已登記的目標權重，從下一個交易日開盤依同一回測引擎模擬成交、扣成本並產生 A–D 的 NAV 與交易明細。未在當日登記的交易日應記為資料缺口，前瞻日誌拒絕事後補寫。

首次執行 `refresh` 需要 FinMind 網路存取，並更新該工作副本的原始資料。實際機動定存年率以當日取得的數值輸入。紙上交易績效只使用凍結後登記的行情，至少 20 筆日超額報酬才顯示年化 Sharpe。

### V7 手動重跑與輸出

在要寫入新結果的專案副本根目錄執行。表格、圖、PIT 檢查與 Model 2 報表分別位於 `outputs/v7/` 下的 `tables/`、`figures/`、`tables/pit/` 與 `tables/model2/`。

```bash
python -m pip install -r requirements.txt
python src/02_build_dataset.py         # 還原權值
python src/03_features_labels.py       # 開盤到開盤 5 日標籤
python src/03b_features_v2.py
python src/features_30.py
python src/universe.py                 # 可交易名單（250 日均額 ≥ 3.5 億、≥ 200 有效日）與每週決策日
python src/03d_garch_oo.py             # V7 2.4.1：第 18 項特徵「開盤到開盤 GARCH」
python src/16_feature_selection_v7.py  # 重現 18 項特徵篩選
python src/05_model1_v7.py             # 117 個候選、三年 QLIKE 選模、事後評估與預測區間
python src/07_model2_v7.py             # 窗口 → 目標波動率 → α → A–D 回測、DSR、敏感度
python src/10_pit_check.py             # 截斷原始資料後重算，比較截止日前的特徵
python src/11_figures_v7.py            # 九張圖
python src/14_export_predictions.py    # 每週每檔的預測波動率整理成 CSV（outputs/v7/tables/volatility_predictions_weekly.csv）
python src/12_correlation_checks.py    # 特徵相關、每週相關矩陣檢查與權重穩定性
python src/15_build_dashboard.py       # 產生可用瀏覽器開啟的 HTML 儀表板
python src/19_build_research_dashboard.py # 用同一次 V7 輸出產生四頁式儀表板
```

上述步驟使用已下載至 `data/raw/` 的資料快照。更新資料時可參閱上方的下載步驟，用 `data/raw/manifest.json` 核對資料版本。

| 程式 | 做什麼 | 主要輸出 |
|---|---|---|
| `src/03d_garch_oo.py` | 開盤到開盤 GARCH(1,1)，每 63 日擴張視窗重估、至少 250 個報酬 | `data/processed/features_garch_oo.parquet`、`outputs/v7/tables/garch_oo_fits.csv` |
| `src/05_model1_v7.py` | 4 個基準＋Ridge／Lasso／ElasticNet／LightGBM，共 117 個候選 | `outputs/v7/tables/model1_*.csv`、`selected_configs.csv` |
| `src/model2_v7.py` | Model 2 的數學：收縮相關、風險預算求解（凸問題→超過 30% 改 SLSQP）、波動目標、Nₜ<4 例外 | （被 07 呼叫） |
| `src/07_model2_v7.py` | V7 5.5 → 7.5 → 4.4 的選擇程序與 A–D 回測（隔日開盤成交、手續費、證交稅、滑價、臺銀定存利率） | `outputs/v7/tables/model2/` |
| `src/10_pit_check.py` | 預設截斷四個截止日，重算並比較截止日前的特徵 | `outputs/v7/tables/pit/` |
| `src/11_figures_v7.py` | 結果圖 | `outputs/v7/figures/v7_fig*.png` |
| `src/12_correlation_checks.py` | 特徵相關、相關矩陣數值檢查、權重穩定性 | `outputs/v7/tables/correlation/`、`v7_fig10`、`v7_fig11` |

V7 回測中涉及缺值、例外週、停牌與模型重訓頻率的實作決定，列在 `src/07_model2_v7.py` 開頭。

## 舊版研究紀錄

以下列出早期版本和原研究工作區的歷史輸出位置。GitHub checkout 包含程式來源。重現 V7 請使用上方步驟。

| 版本 | 特徵 | 驗證期 | 封存期 | 結果位置 |
|---|---|---|---|---|
| **v1** | 文件原始 15 項 | 已跑 | **已正式執行一次（唯一一次）** | `outputs/tables`、`outputs/figures`、`outputs/MLQT_group5_results.xlsx` |
| **v2** | 第二版 17 項（`notebooks/03b` 只用 2016-11～2020 篩選） | 已跑 | **不重跑**：封存期已用過，v2 若在封存期評估必須標註為事後評估 | `outputs/v2/` |
| **v3** | 沿用 v2 的 17 項特徵，加上當時提案的交易規則 | 保留模型驗證結果及回測程式 | 如使用該期間，列為事後評估 | `outputs/v3/`、規格見 `docs/strategy_v3.md` |
| **v4** | 從 17 項與 30 項候選中篩出 18 項，使用原版 GARCH 特徵 | 保留歷史驗證結果 | 如使用該期間，列為事後評估 | `outputs/v4/` |

v1 vs v2 驗證期比較：`outputs/v2/tables/v1_vs_v2_validation.xlsx`、`outputs/v2/figures/v1_vs_v2_validation.png`。

### 舊版程式入口

下列命令供查閱各舊版的入口。各版本使用對應的設定與輸出，請依研究版本選擇所需步驟。

```bash
pip install -r requirements.txt
python src/01_download.py        # 下載 FinMind 原始資料（免費版有次數限制，程式會自動等待重試）
python src/01b_listing.py        # 證交所上市日期與產業別、手動分割事件（0050 2025-06-18 一拆四）
python src/02_build_dataset.py   # 清理、自行還原權值、資料品質報告
python src/03_features_labels.py # v1 的 15 項特徵（10/20/40 日）與開盤到開盤 5 日波動標籤
python src/03b_features_v2.py    # v2 候選特徵 27 項（含每檔 GARCH 走勢前進估計、行事曆）
python src/04_triage.py          # v1 特徵五道關卡（只用 2016-11～2021 訓練期）
python src/05_validate.py        # 三個驗證年度 walk-forward、參數選擇、成對比較、SHAP（依 features.active 決定特徵組）
python src/06_backtest_validation.py  # 驗證期回測、變體、成本掃描
python src/06b_compare_versions.py    # （v2 時）v1 vs v2 比較
python src/universe.py           # （v3）時點正確的可交易股票池（250 日均額 ≥ 3.5 億）與每週決策日
python src/05_validate_v3.py     # （v3）115 個候選模型、QLIKE 選模、兩種風險等級規則比較（不交易）
python src/06_backtest_v3.py     # （v3）驗證期回測——已寫好，尚未執行
python src/07_holdout.py         # 封存測試：只能執行一次（outputs/HOLDOUT_USED.txt 存在時會拒絕執行；v1 已用過）
python src/08_report.py          # v1 總表、六張圖、風險、預測區間、DSR
python src/09_excel.py           # v1 表格彙整成 Excel
```

`notebooks/` 依課程範本（ETF case study 01–05）做資料與特徵檢查，見 `notebooks/README.md`。
資料版本可查 `data/raw/manifest.json`。比較不同執行結果時，同時核對資料快照、程式版本及套件版本。

## 設定與研究帳本

- `setup.yaml`：資料期間、特徵組、驗證與模型參數，以及各版本相關設定。
- `run_log.csv`：研究帳本（每次執行的 git 版本、資料快照、特徵組、評估區間、指標、決策）
- `run_log_before_fix.csv`：帳本修正前的原檔（見下方「修正紀錄」）

## 舊版修正紀錄（2026-11-17）
1. **驗證期碰到封存期**：2024-12-25 決策日的 15 筆驗證樣本，標籤在 2025/1 才結束。已改為依「標籤結束日」切（`models.split`）。
   重算後 v1 選出的參數完全相同，因此 v1 的回測與封存結果不受影響。
2. **研究帳本欄位錯位**：早期 append 時欄位沒有對齊，回看期、回測、封存期紀錄的數值落在錯的欄位，第一次驗證期回測也被重複記錄。
   以 `src/tools_fix_run_log.py` 依各類紀錄的原始欄位順序重建（原檔保留為 `run_log_before_fix.csv`），`log_runs` 改為依欄名對齊。
3. **Deflated Sharpe 的試驗數**：原本誤把封存期紀錄算成試驗（N=69），改為只算封存測試執行之前的試驗（N=57）；DSR 幾乎不變。

## 舊版實作決定（V1–V3）
1. 資料來源 FinMind（TWSE/TPEx 日成交）；FinMind 免費版沒有還原價，自行以除權息與減資參考價回溯調整。
2. 5222、6770、8028 只使用證交所上市日之後的資料（之前為興櫃，無漲跌幅限制）。
3. 訓練樣本使用所有交易日（標籤重疊，以 purge 處理）；評估只用每 5 日的決策日（時間上不重疊）。
4. Ridge alpha 依老師要求擴為 0.01～10000（含簡報的 0.1/1/10）；LightGBM 使用 MAE 損失、簡報的 8 組參數，未用 early stopping。
5. 股利處理：持有期間以還原價評價（等同股利再投入同一股票，最多 5 日），成交用原始開盤價、整張。
6. 主策略事先指定為「驗證 MAE 最低的模型（LightGBM）＋股票等權」，在看回測結果之前決定。
7. 本金採簡報第 11 頁的 3 億元；備註中的 100 萬元列為變體（整張限制下多數股票買不到）。
8. v2 的股票池規則與 v1 相同；v2 特徵的缺值（新股前 60 天、GARCH 暖身期）在 Ridge 用訓練期中位數補，LightGBM 直接處理缺值。
9. v2 有兩項特徵用到超過 20 日的歷史（60 日波動、GARCH），與文件「過去 20 個交易日」不同，需在報告說明。
10. v3 的選模規則原文是「QLIKE 差距 1% 內選較簡單者」，第一次執行時 Lasso 與 ElasticNet 同分，原文沒有定義跨家族誰較簡單；
    補寫為「先比超參數個數，再比懲罰強度／樹數」（`setup.yaml` 的 `v3.models.tie_break`）。這個補充只用到驗證期資訊。
11. v3 的模型用所有合格股票訓練（學「波動怎麼變」），只有交易、門檻與評估限制在可交易股票。
