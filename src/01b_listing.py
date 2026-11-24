# 01b_listing.py — 從證交所 OpenAPI 取得上市日期與產業別，並記錄手動補充的分割事件
import json, urllib.request, datetime as dt, pandas as pd   # 載入：JSON、網路、時間、表格
from common import ROOT, CFG                                 # 載入：專案路徑與設定

url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"   # 證交所「上市公司基本資料」
js = json.load(urllib.request.urlopen(url, timeout=60))      # 下載並解析 JSON
ids = CFG["universe"]["stocks"]                              # 15 檔候選股票
rows = [{"sid": r["公司代號"], "name": r["公司簡稱"], "listing_date": pd.to_datetime(r["上市日期"]),
         "twse_industry_code": r["產業別"]} for r in js if r["公司代號"] in ids]  # 只留需要的欄位
df = pd.DataFrame(rows).sort_values("sid")                   # 轉成表格並排序
df["semiconductor_by_twse"] = df.twse_industry_code.eq("24") # 產業代碼 24＝半導體業
df["downloaded_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # 擷取時間
df.to_csv(ROOT/"data"/"raw"/"listing_dates.csv", index=False, encoding="utf-8-sig")   # 存檔
print(df)                                                    # 印出結果

# 手動補充：FinMind 免費版無分割資料；0050 於 2025-06-18 以 1 拆 4 恢復交易（元大投信公告）
split = pd.DataFrame([{"sid": "0050", "date": "2025-06-18", "before": 4.0, "after": 1.0, "type": "分割"}])  # 因子 = 1/4
split.to_csv(ROOT/"data"/"raw"/"manual_splits.csv", index=False, encoding="utf-8-sig")  # 存檔
