# 01_download.py — 從 FinMind 下載原始日成交資料、除權息結果與減資參考價
import json, time, datetime as dt, pathlib, urllib.request, urllib.parse  # 載入：JSON、等待、時間、路徑、網路請求工具
import argparse
import yaml, pandas as pd                                                 # 載入：讀設定檔、表格處理

ROOT = pathlib.Path(__file__).resolve().parents[1]                        # 專案根目錄
CFG = yaml.safe_load(open(ROOT / "setup.yaml", encoding="utf-8"))         # 讀入 setup.yaml
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)       # 原始資料資料夾，不存在就建立
API = "https://api.finmindtrade.com/api/v4/data"                          # FinMind API 網址
parser = argparse.ArgumentParser()
parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 11, 12))
START, END = CFG["data"]["start"], parser.parse_args().end.isoformat()
IDS = CFG["universe"]["stocks"] + [CFG["universe"]["gold_etf"], CFG["universe"]["passive_benchmark"]]  # 要下載的代號清單

def get(dataset, sid, start=START):                                       # 定義：下載單一資料集、單一代號
    q = urllib.parse.urlencode({"dataset": dataset, "data_id": sid, "start_date": start, "end_date": END})  # 組合查詢參數
    for attempt in range(5):                                              # 最多重試 5 次
        try:                                                              # 嘗試連線
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:   # 送出請求
                js = json.load(r)                                         # 解析回傳的 JSON
            if js.get("status") == 200:                                   # 成功時
                return pd.DataFrame(js["data"])                           # 轉成表格回傳
            print(dataset, sid, js.get("msg")); time.sleep(30)            # 失敗（多半是次數上限）就等 30 秒
        except Exception as e:                                            # 網路錯誤時
            print("retry", dataset, sid, e); time.sleep(10)               # 印出錯誤並等 10 秒
    raise RuntimeError(f"下載失敗 {dataset} {sid}")                        # 5 次都失敗就停止程式

frames = {"price": [], "div": [], "capred": []}                           # 準備三個清單收集結果
for sid in IDS:                                                           # 逐一處理每個代號
    frames["price"].append(get("TaiwanStockPrice", sid)); time.sleep(1.5) # 下載日成交資料（未還原）
    frames["div"].append(get("TaiwanStockDividendResult", sid, "2010-01-01")); time.sleep(1.5)  # 下載除權息結果
    frames["capred"].append(get("TaiwanStockCapitalReductionReferencePrice", sid, "2010-01-01")); time.sleep(1.5)  # 下載減資參考價
    print("done", sid, len(frames["price"][-1]))                          # 回報這個代號完成與筆數

stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # 記錄下載時間（UTC）
for k, lst in frames.items():                                             # 逐一存檔
    df = pd.concat([d for d in lst if len(d)], ignore_index=True)         # 合併所有代號（略過空表）
    df["downloaded_at"] = stamp                                           # 加上擷取時間欄位
    df.to_parquet(RAW / f"{k}.parquet", index=False)                      # 存成 Parquet
    print(k, df.shape)                                                    # 印出資料大小
json.dump({"vendor": "FinMind", "downloaded_at": stamp, "start": START, "end": END, "ids": IDS},
          open(RAW / "manifest.json", "w"), ensure_ascii=False, indent=2) # 寫下資料版本說明檔
