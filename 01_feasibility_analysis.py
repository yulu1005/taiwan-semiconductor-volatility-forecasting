"""核對候選股、下載上市日行情並輸出資料可行性報告。"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config


TWSE_STOCK_DAY = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
FINMIND_DATA = "https://api.finmindtrade.com/api/v4/data"
CACHE_DIR = config.RAW_DATA_DIR / "monthly_cache"
PROFILE_URL = "https://www.twse.com.tw/pdf/ch/{}_ch.pdf"
PRICE_FIELDS = ("open", "high", "low", "close", "volume")
RAW_FIELDS = ("date", *PRICE_FIELDS)
TWSE_FIELDS = ("日期", "開盤價", "最高價", "最低價", "收盤價", "成交股數")

# 業務分組是研究用判定，證交所產業分類另列。
STOCK_PROFILES = {
    "2303": ("聯電", "半導體業", "晶圓代工", "提供積體電路晶圓代工服務"),
    "2330": ("台積電", "半導體業", "晶圓代工", "依客戶設計製造積體電路"),
    "2337": ("旺宏", "半導體業", "記憶體", "設計與製造非揮發性記憶體"),
    "2340": ("台亞", "半導體業", "光電與半導體元件", "製造光電半導體及電子元件"),
    "2344": ("華邦電", "半導體業", "記憶體", "開發及製造記憶體晶片"),
    "2371": ("大同", "電機機械", "機電與能源", "研究指定候選股，主要業務為機電能源系統及消費產品"),
    "2408": ("南亞科", "半導體業", "記憶體", "開發及製造 DRAM"),
    "2434": ("統懋", "半導體業", "功率半導體", "製造二極體及功率電晶體"),
    "2455": ("全新", "通信網路業", "化合物半導體磊晶", "製造 III-V 族半導體磊晶片"),
    "3016": ("嘉晶", "半導體業", "半導體磊晶", "開發製造矽及化合物半導體磊晶晶圓"),
    "3532": ("台勝科", "半導體業", "矽晶圓材料", "製造半導體用矽晶圓"),
    "4919": ("新唐", "半導體業", "IC 產品與晶圓代工", "具自有 IC 產品及六吋晶圓代工業務"),
    "5222": ("全訊", "半導體業", "微波半導體", "研發與製造微波半導體元件、IC 及次系統"),
    "6770": ("力積電", "半導體業", "晶圓代工", "提供邏輯及記憶體晶圓代工"),
    "8028": ("昇陽半導體", "半導體業", "晶圓再生與薄化", "提供晶圓再生及薄化加工服務"),
}
BUSINESS_SOURCES = {
    "2434": "https://www.mospec.com.tw/",
    "2455": "https://ic.tpex.org.tw/company_data.php?column=vision_story&table=company_vision&userId=2455",
}
LISTING_DATES = {
    "2303": date(1985, 7, 16), "2330": date(1994, 9, 5),
    "2337": date(1995, 3, 15), "2340": date(1995, 5, 2),
    "2344": date(1995, 10, 18), "2371": date(1962, 2, 9),
    "2408": date(2000, 8, 17), "2434": date(2000, 9, 11),
    "2455": date(2002, 1, 24), "3016": date(2002, 12, 24),
    "3532": date(2007, 12, 10), "4919": date(2010, 9, 27),
    "5222": date(2021, 10, 25), "6770": date(2021, 12, 6),
    "8028": date(2018, 7, 10),
}
POOL_GROUPS = {
    "2303": "core_foundry",
    "2330": "core_foundry",
    "6770": "core_foundry",
    "4919": "review_external_foundry",
}


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stock_pool():
    for code in config.STOCK_CODES:
        name, industry, business, reason = STOCK_PROFILES[code]
        yield {
            "code": code, "name": name, "market": "TWSE", "exchange_industry": industry,
            "business_group": business, "in_scope": None,
            "pool_group": POOL_GROUPS.get(code, "review"),
            "twse_listing_date": LISTING_DATES[code],
            "reason": reason, "source_url": PROFILE_URL.format(code),
            "business_source_url": BUSINESS_SOURCES.get(code, PROFILE_URL.format(code)),
        }


def parse_month(payload, month):
    rows = []
    status = payload.get("stat", "")
    if "沒有符合條件" in status:
        return rows
    if status != "OK":
        raise ValueError(status or "無法辨識證交所回應")
    columns = [payload["fields"].index(field) for field in TWSE_FIELDS]
    for raw in payload["data"]:
        day, *values = [raw[index] for index in columns]
        year, number, day_of_month = map(int, day.split("/"))
        trade_date = date(year + 1911, number, day_of_month)
        if trade_date.replace(day=1) != month or not all(isinstance(value, str) for value in values):
            raise ValueError("行情月份或欄位格式不符")
        rows.append(dict(zip(RAW_FIELDS, (trade_date.isoformat(), *values), strict=True)))
    return rows


def fetch_json(url, cache, parse, cacheable, *, legacy_cache=None, attempts=4, request_delay=0.4):
    if cacheable:
        for candidate in (cache, legacy_cache):
            if candidate is None or not candidate.is_file():
                continue
            try:
                rows = parse(json.loads(candidate.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue
            if rows:
                return rows

    for attempt in range(attempts):
        try:
            sleep(request_delay)
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            rows = parse(payload)
            break
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            permanent = isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429
            if permanent or attempt == attempts - 1:
                raise ValueError(str(error)) from error
            retry_after = error.headers.get("Retry-After", "") if isinstance(error, HTTPError) else ""
            base_delay = 10 if isinstance(error, HTTPError) and error.code in (307, 308, 429) else 2
            sleep(max(base_delay * 2 ** attempt, int(retry_after) if retry_after.isdecimal() else 0))
    if cacheable and rows:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache.parent, encoding="utf-8", mode="w",
                                delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False)
        temporary_path = cache.parent / temp.name
        try:
            temporary_path.replace(cache)
        finally:
            temporary_path.unlink(missing_ok=True)
    return rows


def fetch_month(code, month):
    monthly_file = f"{month:%Y-%m}.json"
    cache = CACHE_DIR / code / monthly_file
    legacy_cache = config.RAW_DATA_DIR / ".twse_monthly_cache" / code / monthly_file
    current_month = datetime.now(timezone(timedelta(hours=8))).date().replace(day=1)
    query = urlencode({"response": "json", "date": month.strftime("%Y%m%d"), "stockNo": code})
    try:
        return fetch_json(
            f"{TWSE_STOCK_DAY}?{query}", cache, lambda payload: parse_month(payload, month),
            month < current_month, legacy_cache=legacy_cache, attempts=3, request_delay=1,
        )
    except ValueError as error:
        raise ValueError(f"{month:%Y-%m} {error}") from error


def fetch_finmind_prices(code, start, end):
    start = max(start, LISTING_DATES[code])
    if start > end:
        return []
    fields = ("open", "max", "min", "close", "Trading_Volume")

    def parse(payload):
        if payload.get("status") != 200:
            raise ValueError(payload.get("msg", "無法辨識 FinMind 回應"))
        rows = []
        for raw in payload["data"]:
            day = date.fromisoformat(raw["date"])
            if raw["stock_id"] != code:
                raise ValueError("FinMind 回傳股票代號不符")
            if start <= day <= end:
                values = ["" if raw[field] is None else str(raw[field]) for field in fields]
                rows.append(dict(zip(RAW_FIELDS, (day.isoformat(), *values), strict=True)))
        return sorted(rows, key=lambda row: row["date"])

    query = urlencode({
        "dataset": "TaiwanStockPrice", "data_id": code, "start_date": start, "end_date": end,
    })
    cache = config.PRICE_CACHE_DIR / "finmind" / f"{code}_{start}_{end}.json"
    today = datetime.now(timezone(timedelta(hours=8))).date()
    return fetch_json(f"{FINMIND_DATA}?{query}", cache, parse, end < today)


def fetch_prices(code, start, end):
    start = max(start, LISTING_DATES[code])
    rows = []
    month = start.replace(day=1)
    while start <= end and month <= end:
        rows.extend(fetch_month(code, month))
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return [row for row in rows if start.isoformat() <= row["date"] <= end.isoformat()]


def parse_number(value):
    value = value.strip().replace(",", "")
    if value in {"", "--", "---", "----", "—"}:
        return None
    try:
        return float(value)
    except ValueError:
        return math.nan


def price_jumps(closes, sessions):
    positions = {day: index for index, day in enumerate(sorted(sessions))}
    valid_dates = sorted(closes)
    for prior, current in zip(valid_dates, valid_dates[1:]):
        change = closes[current] / closes[prior] - 1
        if abs(change) <= config.PRICE_JUMP_THRESHOLD:
            continue
        gap_sessions = None
        if prior in positions and current in positions:
            gap_sessions = positions[current] - positions[prior] - 1
        yield prior, current, change, gap_sessions


def inspect_prices(rows, sessions, listing_date):
    day_counts = Counter(row["date"] for row in rows)
    missing = Counter()
    invalid = Counter()
    closes = {}
    inconsistent = 0
    zero_volume = 0
    for row in rows:
        values = {field: parse_number(row[field]) for field in PRICE_FIELDS}
        missing.update(field for field, value in values.items() if value is None)
        invalid_fields = [
            field for field, value in values.items()
            if value is not None and (
                not math.isfinite(value)
                or (value < 0 or not value.is_integer() if field == "volume" else value <= 0)
            )
        ]
        invalid.update(invalid_fields)
        if None in values.values() or invalid_fields:
            continue
        open_, high, low, close, volume = values.values()
        if not low <= min(open_, close) <= max(open_, close) <= high:
            inconsistent += 1
            continue
        if volume == 0:
            zero_volume += 1
        elif day_counts[row["date"]] == 1:
            closes[row["date"]] = close

    valid_dates = sorted(closes)
    jumps = list(price_jumps(closes, sessions))
    days = sorted(day_counts)
    listed_sessions = {day for day in sessions if day >= listing_date.isoformat()}
    active_sessions = {day for day in listed_sessions if days and days[0] <= day <= days[-1]}
    coverage = len(active_sessions & closes.keys()) / len(active_sessions) if active_sessions else None
    missing_sessions = sorted(active_sessions - day_counts.keys())
    boundary_gaps = sorted(listed_sessions - active_sessions)
    return {
        "first_date": days[0] if days else "",
        "last_date": days[-1] if days else "",
        "effective_start": valid_dates[0] if valid_dates else "",
        "effective_end": valid_dates[-1] if valid_dates else "",
        "rows": len(rows),
        "valid_days": len(closes),
        "reference_days": len(active_sessions) if sessions else None,
        "prelisting_reference_days": len(sessions - listed_sessions) if sessions else None,
        "listed_reference_days": len(listed_sessions) if sessions else None,
        "missing_session_days": len(missing_sessions) if sessions else None,
        "missing_session_dates": "|".join(missing_sessions),
        "boundary_missing_days": len(boundary_gaps) if sessions else None,
        "boundary_missing_dates": "|".join(boundary_gaps),
        "unusable_quote_dates": "|".join(sorted(day_counts.keys() - closes.keys())),
        "coverage": round(coverage, 6) if coverage is not None else None,
        "duplicate_rows": len(rows) - len(days),
        **{f"missing_{field}": missing[field] for field in PRICE_FIELDS},
        **{f"invalid_{field}": invalid[field] for field in PRICE_FIELDS},
        "inconsistent_ohlc_rows": inconsistent,
        "zero_volume_rows": zero_volume,
        "price_jump_count": sum(gap == 0 for _, _, _, gap in jumps),
        "price_jump_dates": "|".join(day for _, day, _, gap in jumps if gap == 0),
        "cross_gap_jump_count": sum(gap is not None and gap > 0 for _, _, _, gap in jumps),
        "cross_gap_jump_dates": "|".join(day for _, day, _, gap in jumps if gap is not None and gap > 0),
        "unknown_gap_jump_count": sum(gap is None for _, _, _, gap in jumps),
    }


def verdict(metrics, error):
    """判定行情資料品質。股票的研究資格另見股票池。"""
    if error:
        return "download_failed", error
    if metrics["valid_days"] < config.MIN_VALID_DAYS:
        return "insufficient", "有效交易日少於初篩門檻"
    if metrics["coverage"] is None:
        return "review", "缺少參考交易日，無法判定覆蓋率"
    if metrics["coverage"] < config.MIN_COVERAGE:
        return "insufficient", "有效期間覆蓋率低於初篩門檻"
    quality_fields = (
        "duplicate_rows", "inconsistent_ohlc_rows", "zero_volume_rows", "price_jump_count",
        "boundary_missing_days", "cross_gap_jump_count", "unknown_gap_jump_count",
        *(f"{kind}_{field}" for kind in ("missing", "invalid") for field in PRICE_FIELDS),
    )
    if any(metrics[field] for field in quality_fields):
        return "review", "存在資料異常，需人工核對"
    return "pass", "符合資料初篩門檻"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-only", action="store_true", help="只輸出候選股票清單")
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    parser.add_argument("--source", choices=("finmind", "twse"), default="finmind",
                        help="行情來源，預設為 FinMind TaiwanStockPrice")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start 必須早於或等於 --end")

    pool = list(stock_pool())
    write_csv(config.TABLES_DIR / "stock_pool.csv", list(pool[0]), pool)
    if args.pool_only:
        return 0

    prices = {}
    errors = {}
    downloader = fetch_finmind_prices if args.source == "finmind" else fetch_prices

    def download(stock):
        code = stock["code"]
        print(f"下載 {code} {stock['name']}", flush=True)
        try:
            rows = downloader(code, args.start, args.end)
            suffix = "_finmind" if args.source == "finmind" else ""
            write_csv(config.RAW_DATA_DIR / f"{code}_{args.start}_{args.end}{suffix}.csv", RAW_FIELDS, rows)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            print(f"{code} {error}", file=sys.stderr)
            return code, [], str(error)
        return code, rows, ""

    with ThreadPoolExecutor(max_workers=3 if args.source == "finmind" else 1) as executor:
        for code, rows, error in executor.map(download, pool):
            if error:
                errors[code] = error
            else:
                prices[code] = rows
                print(f"完成 {code}，{len(rows)} 筆", flush=True)

    sessions = {row["date"] for row in prices.get(config.REFERENCE_STOCK, [])}
    report = []
    for stock in pool:
        code = stock["code"]
        error = errors.get(code, "")
        metrics = inspect_prices(prices.get(code, []), sessions, stock["twse_listing_date"])
        status, reason = verdict(metrics, error)
        report.append({
            **stock, "requested_start": args.start, "requested_end": args.end,
            "price_source": args.source,
            **(dict.fromkeys(metrics) if error else metrics),
            "status": status, "decision_reason": reason, "download_error": error,
        })
    write_csv(config.TABLES_DIR / "feasibility_report.csv", list(report[0]), report)
    print(f"報告已輸出至 {config.TABLES_DIR / 'feasibility_report.csv'}")
    return int(bool(errors) or not sessions)


if __name__ == "__main__":
    sys.exit(main())
