"""下載 Yahoo Finance 日行情與公司行動。"""

import argparse
import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config


TAIPEI = timezone(timedelta(hours=8))
FIELDS = ("date", "code", "adjusted_close", "yahoo_close", "dividends", "stock_splits")


def positive(value):
    return value if isinstance(value, (float, int)) and math.isfinite(value) and value > 0 else None


def download(code, start, end):
    def timestamp(day):
        return int(datetime.combine(day, datetime.min.time(), TAIPEI).timestamp())

    query = urlencode({
        "period1": timestamp(start), "period2": timestamp(end + timedelta(days=1)),
        "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true",
    })
    for attempt, host in enumerate(("query1", "query2", "query1")):
        request = Request(f"https://{host}.finance.yahoo.com/v8/finance/chart/{code}.TW?{query}",
                          headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=20) as response:
                chart = json.load(response)["chart"]
            if chart["error"]:
                raise ValueError(str(chart["error"]))
            data = chart["result"][0]
            break
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            if attempt == 2:
                raise ValueError(str(error)) from error
            sleep(2 ** attempt)

    def trading_day(value):
        return datetime.fromtimestamp(value, TAIPEI).date()

    events = data.get("events", {})
    dividends = {}
    for event in events.get("dividends", {}).values():
        day = trading_day(event["date"])
        dividends[day] = dividends.get(day, 0) + event["amount"]
    splits = {trading_day(event["date"]): event["numerator"] / event["denominator"]
              for event in events.get("splits", {}).values()}
    indicator = data["indicators"]
    rows = {}
    for stamp, close, adjusted in zip(data["timestamp"], indicator["quote"][0]["close"],
                                      indicator["adjclose"][0]["adjclose"], strict=True):
        day = trading_day(stamp)
        if not start <= day <= end:
            continue
        if day in rows:
            raise ValueError(f"日期重複 {day}")
        rows[day] = dict(zip(FIELDS, (day, code, positive(adjusted), positive(close),
                                     dividends.get(day, 0), splits.get(day, 0)), strict=True))
    return [rows[day] for day in sorted(rows)]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start 必須早於或等於 --end")
    all_rows, audit = [], []
    for code in config.STOCK_CODES:
        try:
            rows = download(code, args.start, args.end)
            valid = [row for row in rows if row["adjusted_close"] is not None]
            error = "" if valid else "Yahoo 無正數調整後價格"
        except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError) as exc:
            rows, valid, error = [], [], str(exc)
        audit.append({
            "code": code, "ticker": f"{code}.TW", "status": "failed" if error else "ok",
            "error": error, "yahoo_rows": len(rows), "valid_adjusted_rows": len(valid),
            "first_valid_date": valid[0]["date"] if valid else "",
            "last_valid_date": valid[-1]["date"] if valid else "",
        })
        all_rows.extend(rows)
        print(code, error or f"{len(valid)} 個有效價格", flush=True)
    write_csv(config.TABLES_DIR / "yahoo_download_audit.csv", list(audit[0]), audit)
    write_csv(config.EXTERNAL_DATA_DIR / "yahoo_daily.csv", FIELDS, all_rows)
    write_csv(config.ADJUSTED_CLOSE_PATH, FIELDS,
              (row for row in all_rows if row["adjusted_close"] is not None))
    return int(any(row["error"] for row in audit))


if __name__ == "__main__":
    raise SystemExit(main())
