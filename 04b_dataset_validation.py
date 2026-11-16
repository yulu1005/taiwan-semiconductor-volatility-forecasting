"""唯讀驗證 04 的三份時序資料集及其稽核數字。"""

import csv
import math
import sys
from collections import Counter
from datetime import date
from importlib import import_module

import config


FEATURE_COLUMNS = import_module("03_financial_features").FEATURE_COLUMNS
TARGET = "future_5d_volatility"
REQUIRED = {
    "date", "code", "session_index", "label_start_date", "label_end_date",
    TARGET, *FEATURE_COLUMNS,
}
PERIODS = (
    ("train", date(2016, 11, 1), date(2021, 12, 31)),
    ("validation", date(2022, 1, 1), date(2023, 12, 31)),
    ("test", date(2024, 1, 1), date(2026, 11, 1)),
)


def read_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} 缺少欄位 {', '.join(sorted(missing))}")
        return list(reader)


def parse_date(value):
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError(f"日期必須為 YYYY-MM-DD，收到 {value!r}")
    return parsed


def reference_calendar():
    rows = read_rows(config.PROCESSED_DATA_DIR / "adjusted_daily_returns.csv",
                     {"code", "date", "session_index"})
    calendar = {}
    for row in rows:
        if row["code"] == config.REFERENCE_STOCK:
            index = int(row["session_index"])
            if index in calendar:
                raise ValueError(f"參考日曆交易日索引重複 {index}")
            calendar[index] = parse_date(row["date"])
    days = [calendar[i] for i in sorted(calendar)]
    if not days or days != sorted(set(days)) or sorted(calendar) != list(range(1, len(calendar) + 1)):
        raise ValueError("參考交易日曆為空、日期重複或索引不連續")
    return calendar


def validate_split(name, lower, upper, calendar, audit, used_dates):
    path = config.PROCESSED_DATA_DIR / f"{name}.csv"
    rows = read_rows(path, REQUIRED)
    if not rows:
        raise ValueError(f"{path.name} 無樣本")
    keys, dates, targets, end_dates = set(), set(), [], []
    counts = Counter()
    for line, row in enumerate(rows, start=2):
        try:
            code = row["code"]
            day, label_start, label_end = (
                parse_date(row[field]) for field in ("date", "label_start_date", "label_end_date")
            )
            index = int(row["session_index"])
            key = (day, code)
            if code not in config.STOCK_CODES or key in keys or day in used_dates:
                raise ValueError("股票代號無效、鍵重複或預測日期跨組重疊")
            if not lower <= day < label_start <= label_end <= upper:
                raise ValueError("預測日期或 Label 視窗超出切分期間")
            if (calendar.get(index) != day or calendar.get(index + 1) != label_start
                    or calendar.get(index + config.FORECAST_HORIZON) != label_end):
                raise ValueError("交易日索引或 Label 起訖與參考日曆不符")
            for field in (*FEATURE_COLUMNS, TARGET):
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{field} 缺失或格式無效") from error
                if not math.isfinite(value) or (field == TARGET and value < 0):
                    raise ValueError(f"{field} 數值無效")
            keys.add(key)
            dates.add(day)
            counts[code] += 1
            targets.append(float(row[TARGET]))
            end_dates.append(label_end)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path.name} 第 {line} 列，{error}") from error

    if len(rows) != audit.get((name, "ALL")):
        raise ValueError(f"{path.name} 樣本總數與 dataset_audit.csv 不符")
    for code in config.STOCK_CODES:
        if counts[code] != audit.get((name, code)):
            raise ValueError(f"{path.name} 股票 {code} 的樣本數與 dataset_audit.csv 不符")
    used_dates.update(dates)
    return len(rows), min(dates), max(dates), max(end_dates), sum(value > 1 for value in targets)


def main():
    try:
        if len(FEATURE_COLUMNS) != 15 or config.FORECAST_HORIZON != 5:
            raise ValueError("研究規格須為 15 個 Feature 與五交易日 Label")
        calendar = reference_calendar()
        audit = {}
        for row in read_rows(config.TABLES_DIR / "dataset_audit.csv", {"split", "code", "retained"}):
            key = (row["split"], row["code"])
            if key in audit or int(row["retained"]) < 0:
                raise ValueError(f"dataset_audit.csv 的筆數或鍵無效 {key}")
            audit[key] = int(row["retained"])
        used_dates = set()
        for name, lower, upper in PERIODS:
            count, first, last, label_end, above_one = validate_split(
                name, lower, upper, calendar, audit, used_dates,
            )
            print(f"PASS {name}: {count:,} 筆，{first} ~ {last}，"
                  f"最晚 Label 結束日 {label_end}，年化波動率 > 100% 共 {above_one} 筆")
    except (OSError, TypeError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("PASS 三組切分的日期、筆數、重複鍵、數值及五日 Label 邊界均通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
