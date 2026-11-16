"""離線整理缺漏交易日、無效行情和價格變動的審核明細。"""

import argparse
import csv
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date
from importlib import import_module

import config


feasibility = import_module("01_feasibility_analysis")
PRICE_FIELDS = feasibility.PRICE_FIELDS
FIELDS = (
    "code", "name", "date", "issue", "raw_row_count", *PRICE_FIELDS,
    "previous_valid_date", "previous_valid_close", "next_valid_date", "next_valid_close",
    "gap_sessions", "raw_price_change_pct", "finding", "handling", "source_url",
)

# 起點為停止交易日，終點為恢復交易日。來源欄保留公司公告或年報。
CAPITAL_REDUCTIONS = {
    "2337": [("2017-08-15", "2017-08-28",
              "https://www.macronix.com/zh-tw/about/investor-relations/Documents/annualreport_2017_chinese.pdf")],
    "2340": [
        ("2017-09-22", "2017-10-06", "https://news.cnyes.com/news/id/3929566"),
        ("2019-09-11", "2019-09-23", "https://news.cnyes.com/news/id/4370319"),
    ],
    "2371": [("2025-06-12", "2025-06-23",
              "https://goodinfo.tw/tw/StockAnnounceDetail.asp?CLAIM_TIME=2025%2F05%2F15+22%3A30%3A38&STOCK_ID=2371&SUBJECT=%E6%9C%AC%E5%85%AC%E5%8F%B8%E8%91%A3%E4%BA%8B%E9%95%B7%E8%A8%82%E5%AE%9A%E7%8F%BE%E9%87%91%E6%B8%9B%E8%B3%87%E6%8F%9B%E8%82%A1%E5%9F%BA%E6%BA%96%E6%97%A5%E6%9A%A8%E6%8F%9B%E7%99%BC%E8%82%A1%E7%A5%A8%E4%BD%9C%E6%A5%AD%E8%A8%88%E7%95%AB%E7%AD%89%E7%9B%B8%E9%97%9C%E4%BA%8B%E5%AE%9C")],
    "2434": [
        ("2017-09-20", "2017-09-30", "https://news.cnyes.com/news/id/3924773"),
        ("2020-09-08", "2020-09-15",
         "https://www.mospec.com.tw/big5/data-evaluation/financial-information/year-report/110/110-year-report.pdf"),
    ],
    "3532": [("2018-09-13", "2018-09-25", "https://news.cnyes.com/news/id/4188089")],
}
TWSE_PLACEHOLDER_CHECKS = {
    ("2303", "2018-06-29"), ("2371", "2019-03-27"), ("4919", "2019-11-28"),
    ("2434", "2019-08-28"), ("2434", "2020-04-20"), ("2434", "2020-04-24"),
}


def load_prices(code, start, end, source):
    suffix = "_finmind" if source == "finmind" else ""
    path = config.RAW_DATA_DIR / f"{code}_{start}_{end}{suffix}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def describe_issue(code, day, issue, raw):
    if issue == "unusable_quote":
        values = [feasibility.parse_number(raw[field]) for field in PRICE_FIELDS]
        if values[:4] == [0] * 4:
            volume = "成交量亦為零" if values[-1] == 0 else "成交量非零"
            finding = f"原始 CSV 的 OHLC 全為零，{volume}"
            source = ""
            if (code, day) in TWSE_PLACEHOLDER_CHECKS:
                finding += "，證交所 OHLC 為 --"
                month = date.fromisoformat(day).strftime("%Y%m01")
                source = f"{feasibility.TWSE_STOCK_DAY}?response=json&date={month}&stockNo={code}"
            return finding, "排除涉及該日的單日報酬，保留原始列供查核", source
        return "行情未通過有效性檢查", "排除該日報酬，保留原始列供查核", ""
    for start, restart, source in CAPITAL_REDUCTIONS.get(code, ()):
        if start <= day <= restart:
            event = "減資換股暫停交易" if day < restart else "減資換股後恢復交易"
            return event, "保留股票與缺口，核對公司行動調整因子", source
    if issue in {"missing_session", "boundary_missing_session"}:
        return "參考交易日缺少該股行情，原因待查", "保留缺口，核對停牌與證交所原始行情", ""
    return "價格變動超過門檻，原因待查", "核對公司行動與交易日距離", ""


def review_stock(stock, prices, sessions):
    code = stock["code"]
    inputs = defaultdict(list)
    for row in prices:
        inputs[row["date"]].append(row)
    metrics = feasibility.inspect_prices(prices, sessions, stock["twse_listing_date"])
    unusable = set(filter(None, metrics["unusable_quote_dates"].split("|")))
    closes = {
        day: feasibility.parse_number(rows[0]["close"])
        for day, rows in inputs.items() if day not in unusable
    }
    valid_dates = sorted(closes)
    reference = sorted(sessions)
    positions = {day: index for index, day in enumerate(reference)}

    def record(day, issue, prior="", following="", gap=None, change=None):
        for raw in inputs.get(day, [{}]):
            finding, handling, source_url = describe_issue(code, day, issue, raw)
            yield {
                "code": code, "name": stock["name"], "date": day, "issue": issue,
                "raw_row_count": len(inputs.get(day, ())),
                **{field: raw.get(field, "") for field in PRICE_FIELDS},
                "previous_valid_date": prior, "previous_valid_close": closes.get(prior),
                "next_valid_date": following, "next_valid_close": closes.get(following),
                "gap_sessions": gap,
                "raw_price_change_pct": round(100 * change, 6) if change is not None else None,
                "finding": finding, "handling": handling, "source_url": source_url,
            }

    for issue, key in (
        ("unusable_quote", "unusable_quote_dates"),
        ("missing_session", "missing_session_dates"),
        ("boundary_missing_session", "boundary_missing_dates"),
    ):
        for day in filter(None, metrics[key].split("|")):
            left, right = bisect_left(valid_dates, day), bisect_right(valid_dates, day)
            prior = valid_dates[left - 1] if left else ""
            following = valid_dates[right] if right < len(valid_dates) else ""
            gap = positions[following] - positions[prior] - 1 if prior in positions and following in positions else None
            yield from record(day, issue, prior, following, gap)

    for prior, day, change, gap in feasibility.price_jumps(closes, sessions):
        issue = "unknown_gap_jump" if gap is None else "cross_gap_jump" if gap else "daily_jump"
        right = bisect_right(valid_dates, day)
        following = valid_dates[right] if right < len(valid_dates) else ""
        yield from record(day, issue, prior, following, gap, change)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    parser.add_argument("--source", choices=("finmind", "twse"), default="finmind")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start 必須早於或等於 --end")
    try:
        prices = {
            code: load_prices(code, args.start, args.end, args.source) for code in config.STOCK_CODES
        }
    except OSError as error:
        parser.exit(1, f"無法讀取原始行情，請先執行 01_feasibility_analysis.py。{error}\n")
    sessions = {row["date"] for row in prices[config.REFERENCE_STOCK]}
    if not sessions:
        parser.exit(1, "沒有參考交易日，無法審核缺口。\n")
    rows = [
        row for stock in feasibility.stock_pool()
        for row in review_stock(stock, prices[stock["code"]], sessions)
    ]
    path = config.TABLES_DIR / "data_quality_review.csv"
    feasibility.write_csv(path, FIELDS, sorted(rows, key=lambda row: (row["code"], row["date"], row["issue"])))
    print(f"已輸出 {len(rows)} 筆審核明細至 {path}")


if __name__ == "__main__":
    main()
