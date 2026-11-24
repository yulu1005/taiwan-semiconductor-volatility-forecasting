# tools_fix_run_log.py — 一次性修正：研究帳本早期 append 時欄位沒有對齊，依各類紀錄的原始欄位順序重建
import csv, shutil, pandas as pd                             # 載入：CSV 讀寫、檔案複製、表格
from common import ROOT                                      # 專案路徑

f = ROOT / "run_log.csv"                                     # 帳本路徑
if not (ROOT / "run_log_before_fix.csv").exists(): shutil.copy(f, ROOT / "run_log_before_fix.csv")  # 先備份原檔（保留稽核軌跡）
f_in = ROOT / "run_log_before_fix.csv"                      # 一律從備份重建
rows = list(csv.reader(open(f_in, encoding="utf-8-sig")))    # 逐列讀成字串清單
head, body = rows[0], rows[1:]                               # 表頭與內容
META = ["logged_at", "git_commit", "data_snapshot"]          # 每列開頭的三個欄位
MET = ["MAE", "RMSE", "IC", "IC_lo", "IC_hi", "ICIR", "IC_pos_share", "TS_corr_level", "n", "n_dates"]  # 預測指標欄位
LOOK = META + ["family", "config", "fset", "lookback", "fold", "eval_window", "trial_family", "decision"] + MET  # 回看期紀錄的原始順序
BT = META + ["trial_family", "family", "config", "fold", "eval_window", "decision", "Sharpe", "CAGR", "MDD", "turnover_2way"]  # 回測紀錄的原始順序
HO = BT + MET                                                # 封存期紀錄的原始順序
out = []                                                     # 修正後的列
for r in body:                                               # 逐列判斷類型
    if len(r) == len(head) and r[9] == "lookback": out.append(dict(zip(LOOK, r[:len(LOOK)])))  # 回看期列（被補成 24 欄的錯位列）
    elif len(r) == len(head) and r[3].startswith("backtest"): out.append(dict(zip(BT, r[:len(BT)]), _dup=True))  # 第一次回測（被補成 24 欄）：與第二次完全相同，標記為重複
    elif len(r) == len(head): out.append(dict(zip(head, r)))  # 模型網格列（原本就對齊）
    elif len(r) == len(BT): out.append(dict(zip(BT, r)))      # 驗證期回測列
    elif len(r) == len(HO): out.append(dict(zip(HO, r)))      # 封存期列
    else: raise ValueError(f"無法辨識的列（{len(r)} 欄）")      # 其他長度直接停止
df = pd.DataFrame(out)                                       # 轉成表格
print(f"刪除重複的第一次驗證期回測紀錄 {int(df._dup.eq(True).sum())} 列（與重跑結果完全相同）")  # 印出
df = df[df._dup.ne(True)].drop(columns="_dup")               # 刪除重複列
df.to_csv(f, index=False, encoding="utf-8-sig")              # 寫回（欄位為各類欄位的聯集）
print(df.groupby("trial_family").size())                     # 印出各類筆數
