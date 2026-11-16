"""由未來五個交易日的平方對數報酬計算年化實現波動率。"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import date
from importlib import import_module
from pathlib import Path
from statistics import median

import config


write_csv = import_module("01_feasibility_analysis").write_csv


def load_returns(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "code", "session_index", "price_valid", "status", "log_return"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("每日報酬資料缺少必要欄位")
        groups = defaultdict(dict)
        calendar = {}
        for row in reader:
            index = int(row["session_index"])
            day = date.fromisoformat(row["date"])
            code = row["code"]
            if index < 1 or row["date"] != day.isoformat() or row["price_valid"] not in {"0", "1"}:
                raise ValueError(f"無效的交易日資料 {code} {row['date']}")
            if calendar.setdefault(index, day) != day or index in groups[code]:
                raise ValueError(f"交易日索引衝突或重複 {code} {index}")
            groups[code][index] = row
    dates = [calendar[index] for index in sorted(calendar)]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("參考交易日日期重複、順序異常或無資料")
    return groups, max(calendar)


def calculate_label(rows, index, last_session, horizon, annual_days):
    if rows[index]["price_valid"] != "1":
        return None, "invalid_anchor", ()
    if index + horizon > last_session:
        return None, "insufficient_future_sessions", ()
    future = [rows.get(i) for i in range(index + 1, index + horizon + 1)]
    if any(row is None for row in future):
        return None, "nonconsecutive_future_sessions", ()
    if any(row["status"] != "ok" or row["price_valid"] != "1" for row in future):
        return None, "invalid_future_return", ()
    try:
        returns = [float(row["log_return"]) for row in future]
        value = math.sqrt(annual_days / horizon) * math.hypot(*returns)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_future_return", ()
    if not all(map(math.isfinite, returns)) or not math.isfinite(value):
        return None, "invalid_future_return", ()
    return value, "ok", (future[0]["date"], future[-1]["date"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=config.PROCESSED_DATA_DIR / "adjusted_daily_returns.csv")
    parser.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON)
    parser.add_argument("--annual-days", type=int, default=config.ANNUAL_TRADING_DAYS)
    args = parser.parse_args()
    if args.horizon != 5 or args.annual_days < 1:
        parser.error("研究固定使用未來五日，年化交易日數必須大於零")
    try:
        groups, last_session = load_returns(args.input)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    result, audit = [], []
    for code, rows in sorted(groups.items()):
        valid, reasons = [], Counter()
        for index, row in sorted(rows.items()):
            label, reason, window = calculate_label(
                rows, index, last_session, args.horizon, args.annual_days
            )
            reasons[reason] += 1
            if label is not None:
                valid.append(label)
            result.append({
                "date": row["date"], "symbol": code, "future_5d_volatility": label,
                "label_valid": int(label is not None), "label_reason": reason,
                "label_start_date": window[0] if window else "",
                "label_end_date": window[1] if window else "",
            })
        audit.append({
            "symbol": code, "rows": len(rows), "valid_labels": len(valid),
            "invalid_labels": len(rows) - len(valid),
            **{reason: reasons[reason] for reason in (
                "invalid_anchor", "insufficient_future_sessions",
                "nonconsecutive_future_sessions", "invalid_future_return",
            )},
            "min_volatility": min(valid, default=None),
            "median_volatility": median(valid) if valid else None,
            "max_volatility": max(valid, default=None),
            "volatility_above_100pct": sum(value > 1 for value in valid),
        })
    write_csv(config.PROCESSED_DATA_DIR / "labels.csv", list(result[0]), result)
    write_csv(config.TABLES_DIR / "label_audit.csv", list(audit[0]), audit)
    print(f"已輸出 {len(result)} 筆資料，其中 {sum(item['valid_labels'] for item in audit)} 筆有效 Label")


if __name__ == "__main__":
    main()
