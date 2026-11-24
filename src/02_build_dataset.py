# 02_build_dataset.py — 清理原始資料、自行還原權值、產生資料品質報告
import pathlib, yaml, numpy as np, pandas as pd              # 載入：路徑、設定檔、數值運算、表格
import matplotlib; matplotlib.use("Agg")                     # 設定：不開視窗，直接畫到檔案
import matplotlib.pyplot as plt                              # 載入：畫圖工具
from common import ROOT, CFG, setup_fonts                    # 載入：共用的路徑、設定與中文字型

setup_fonts()                                                # 讓圖表可顯示中文
RAW, PROC = ROOT/"data"/"raw", ROOT/"data"/"processed"       # 原始與處理後資料夾
TAB, FIG = ROOT/"outputs"/"tables", ROOT/"outputs"/"figures" # 表格與圖檔資料夾
for d in (PROC, TAB, FIG): d.mkdir(parents=True, exist_ok=True)  # 不存在就建立

p = pd.read_parquet(RAW/"price.parquet")                     # 讀原始日成交資料
p = p.rename(columns={"stock_id": "sid", "max": "high", "min": "low",
                      "Trading_Volume": "volume", "Trading_money": "amount"})  # 欄位改成英文短名
p["date"] = pd.to_datetime(p["date"])                        # 日期轉成日期格式
p = p[["date", "sid", "open", "high", "low", "close", "volume", "amount"]]  # 只留需要的欄位

# ---------- 1. 交易日曆與上市日 ----------
cal = pd.DatetimeIndex(sorted(p.loc[p.sid == "0050", "date"].unique()))  # 以 0050 交易日當作市場日曆
lst = pd.read_csv(RAW/"listing_dates.csv", dtype={"sid": str}, parse_dates=["listing_date"])  # 證交所上市日期
LISTING = dict(zip(lst.sid, lst.listing_date))               # 代號 → 上市日
qlog = []                                                    # 準備收集品質紀錄
for sid, d0 in LISTING.items():                              # 逐一處理每檔股票
    n = ((p.sid == sid) & (p.date < d0)).sum()               # 計算上市前（興櫃）筆數
    if n: qlog.append({"sid": sid, "check": f"上市日 {d0.date()} 前（興櫃，無漲跌幅限制）資料剔除", "count": int(n)})  # 記錄剔除筆數
    p = p[~((p.sid == sid) & (p.date < d0))]                 # 刪除上市前資料
NOLIMIT = {sid: cal[cal.searchsorted(d0)] for sid, d0 in LISTING.items() if d0 > cal[0]}  # 新上市首日（前 5 日無漲跌幅限制）

# ---------- 2. 建立完整面板（日曆 × 代號），缺的日子保留為空值 ----------
ids = sorted(p.sid.unique())                                 # 全部代號
first = p.groupby("sid").date.min()                          # 每檔第一筆資料日
idx = pd.MultiIndex.from_tuples([(d, s) for s in ids for d in cal if d >= first[s]], names=["date", "sid"])  # 只從各檔第一天開始
panel = p.set_index(["date", "sid"]).reindex(idx).sort_index()  # 對齊到完整面板（沒資料的日子為空）
panel["missing_row"] = panel["close"].isna()                 # 標記：當天完全沒有資料列（停牌）

# ---------- 3. 無效列：零成交量或價格為 0 ----------
bad_zero = (panel.volume.fillna(0) == 0) | (panel[["open", "high", "low", "close"]].fillna(0) == 0).any(axis=1)  # 零量或零價
bad_ohlc = (panel.high < panel[["open", "close"]].max(axis=1) - 1e-9) | (panel.low > panel[["open", "close"]].min(axis=1) + 1e-9)  # 開高低收不合理
panel["valid"] = ~(bad_zero | bad_ohlc)                      # 有效列＝非零量、非零價、OHLC 合理
for c in ["open", "high", "low", "close"]:                   # 逐一處理價格欄
    panel.loc[~panel.valid, c] = np.nan                      # 無效列價格設為空值（絕不往前補值）

dup = p.duplicated(["date", "sid"]).sum()                    # 重複筆數
qtab = panel.groupby("sid").agg(                             # 依代號彙總品質指標
    first_date=("open", lambda s: s.index.get_level_values(0).min().date()),  # 第一天
    n_days=("open", "size"),                                 # 面板天數
    missing_rows=("missing_row", "sum"),                     # 沒有資料列的天數
    invalid_rows=("valid", lambda s: int((~s).sum())))       # 無效天數（含缺資料）
qtab["ohlc_violations"] = bad_ohlc.groupby("sid").sum()     # OHLC 異常筆數
qtab["zero_volume_or_price"] = (bad_zero & ~panel.missing_row).groupby("sid").sum()  # 零量/零價（不含缺列）
qtab["duplicates"] = dup                                     # 重複筆數（全體）

# ---------- 4. 還原權值：除權息與減資 ----------
div = pd.read_parquet(RAW/"div.parquet"); div["date"] = pd.to_datetime(div["date"])   # 讀除權息結果
cap = pd.read_parquet(RAW/"capred.parquet"); cap["date"] = pd.to_datetime(cap["date"])  # 讀減資參考價
ev = pd.concat([                                             # 合併兩類事件
    pd.DataFrame({"sid": div.stock_id, "date": div.date, "before": div.before_price, "after": div.after_price, "type": "除權息"}),
    pd.DataFrame({"sid": cap.stock_id, "date": cap.date, "before": cap.ClosingPriceonTheLastTradingDay,
                  "after": cap.PostReductionReferencePrice, "type": "減資"}),
    pd.read_csv(RAW/"manual_splits.csv", dtype={"sid": str}, parse_dates=["date"])], ignore_index=True)  # 加上手動分割事件
ev = ev[(ev.before > 0) & (ev.after > 0)].copy()             # 去掉缺值事件
ev["date"] = ev.date.apply(lambda d: cal[cal.searchsorted(d)] if d <= cal[-1] else pd.NaT)  # 移到當天或之後第一個交易日
ev["factor"] = ev.after / ev.before                          # 調整因子＝參考價 ÷ 前一日收盤
ev = ev.dropna(subset=["date"])                              # 去掉超出日曆的事件
ev.to_csv(TAB/"corporate_actions.csv", index=False, encoding="utf-8-sig")  # 存下事件清單

panel["adj"] = 1.0                                           # 先把調整因子設為 1
for sid, g in ev.groupby("sid"):                             # 逐一處理每檔的事件
    if sid not in ids: continue                              # 不在名單就跳過
    dates = panel.xs(sid, level="sid").index                 # 這檔的所有日期
    f = pd.Series(1.0, index=dates)                          # 每天的累積因子
    for _, e in g.iterrows():                                # 逐一事件
        f[dates < e.date] *= e.factor                        # 事件日之前的價格乘上因子（回溯調整）
    panel.loc[(slice(None), sid), "adj"] = f.values          # 寫回面板
for c in ["open", "high", "low", "close"]:                   # 逐一價格欄
    panel["a" + c] = panel[c] * panel["adj"]                 # 還原後價格（aopen, ahigh, alow, aclose）

# ---------- 5. 檢查：還原後是否還有超過漲跌幅的跳動 ----------
r = panel.aclose.groupby("sid").pct_change(fill_method=None) # 還原後收盤簡單報酬（遇空值不補）
big = r[r.abs() > 0.105]                                     # 超過 ±10.5% 的跳動（台股漲跌幅 10%）
big_tab = big.reset_index().rename(columns={"aclose": "ret"})  # 轉成表格
big_tab["explanation"] = ""                                  # 準備填說明
for i, row in big_tab.iterrows():                            # 逐筆說明
    d0 = NOLIMIT.get(row.sid)                                # 這檔的上市首日
    if d0 is not None and 0 <= cal.get_loc(row.date) - cal.get_loc(d0) < 5: big_tab.at[i, "explanation"] = "新上市前 5 日無漲跌幅限制"
    elif row.sid in ("00635U",): big_tab.at[i, "explanation"] = "追蹤國外指數 ETF 無漲跌幅限制"
big_tab.to_csv(TAB/"unexplained_jumps.csv", index=False, encoding="utf-8-sig")  # 存檔供人工檢查
qtab["jumps_gt_10.5pct"] = big.groupby("sid").size().reindex(qtab.index).fillna(0).astype(int)  # 每檔跳動筆數
qtab["corporate_events"] = ev.groupby("sid").size().reindex(qtab.index).fillna(0).astype(int)  # 每檔公司行動數
qtab.to_csv(TAB/"data_quality_by_stock.csv", encoding="utf-8-sig")  # 存品質報告
pd.DataFrame(qlog).to_csv(TAB/"data_quality_log.csv", index=False, encoding="utf-8-sig")  # 存剔除紀錄

panel.reset_index().to_parquet(PROC/"panel.parquet", index=False)  # 存處理後面板
print(qtab); print("jumps:\n", big_tab)                      # 印出結果

# ---------- 6. 圖：還原前後價格比較（南亞科有多次除息） ----------
s = panel.xs("2408", level="sid")                            # 取出南亞科
fig, ax = plt.subplots(figsize=(9, 4))                       # 開一張圖
ax.plot(s.index, s.close, lw=1, label="原始收盤價")            # 畫原始收盤價
ax.plot(s.index, s.aclose, lw=1, label="還原後收盤價")          # 畫還原收盤價
for d in ev[ev.sid == "2408"].date: ax.axvline(d, color="grey", lw=.5, alpha=.5)  # 標出除權息日
ax.set_title("公司行動檢查：2408 南亞科 原始 vs 還原價格（灰線＝除權息日）"); ax.legend()  # 標題與圖例
fig.tight_layout(); fig.savefig(FIG/"data_corporate_action_check.png", dpi=150); plt.close(fig)  # 存圖
