"""依交易日建立十五項市場特徵與覆蓋率稽核。"""

import argparse
import json
import math
from collections import Counter
from datetime import date
from importlib import import_module
from pathlib import Path
from statistics import fmean, pstdev

import config


load_returns = import_module("02_labels").load_returns
read_csv = import_module("01c_prepare_adjusted_prices").read_csv
positive_number = import_module("01c_prepare_adjusted_prices").number
feasibility = import_module("01_feasibility_analysis")
write_csv = feasibility.write_csv

FEATURE_COLUMNS = (
    "Volatility_5", "Volatility_10", "Volatility_20",
    "Return_1", "Return_5", "Return_20", "High_Low_Range", "Close_Open_Range",
    "MA_Gap_5", "MA_Gap_20", "Volume_Ratio", "Volume_Volatility", "Turnover",
    "RSI_14", "ATR_14",
)
TR_REASONS = (
    "valid", "invalid_raw_ohlcv", "missing_current_session", "missing_previous_session",
    "imputed_price", "company_action_review", "invalid_adjusted_price_or_return",
    "factor_mismatch",
)
ATR_INVALID_COLUMNS = (
    "code", "date", "session_index", "tr_exclusion_reason", "status", "review_reason",
    "price_source", "previous_session_date", "previous_status", "previous_review_reason",
    "previous_price_source",
)


def eligible_price(row):
    return row["price_valid"] == "1" and row["price_source"] == "yahoo"


def eligible_intraday_price(row):
    return (
        eligible_price(row)
        and row["status"] in {"ok", "no_previous_valid_session", "nonconsecutive_sessions"}
        and "adjusted_cross_gap_review" not in row["review_reason"].split("|")
    )


def feature_returns(rows):
    result = {}
    for index, row in rows.items():
        prior = rows.get(index - 1)
        if (
            prior is None or row["status"] != "ok"
            or not (eligible_price(prior) and eligible_price(row))
        ):
            continue
        try:
            value = float(row["log_return"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[index] = value
    return result


def valid_quotes(raw_rows):
    counts = Counter(row["date"] for row in raw_rows)
    quotes = {}
    for row in raw_rows:
        day = row["date"]
        if counts[day] != 1:
            continue
        open_, high, low, close, volume = (
            positive_number(row[field]) for field in ("open", "high", "low", "close", "volume")
        )
        if (
            None in (open_, high, low, close, volume) or not volume.is_integer()
            or not low <= min(open_, close) <= max(open_, close) <= high
        ):
            continue
        quotes[day] = (open_, high, low, close, volume)
    return quotes


def load_trading_money(code, start, end):
    start = max(start, feasibility.LISTING_DATES[code])
    if start > end:
        return {}
    path = config.PRICE_CACHE_DIR / "finmind" / f"{code}_{start}_{end}.json"
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("status") != 200:
        raise ValueError(f"FinMind 成交金額快取無效 {path}")
    money = {}
    for row in payload["data"]:
        day = row["date"]
        if row["stock_id"] != code or day in money:
            raise ValueError(f"FinMind 成交金額代號不符或日期重複 {code} {day}")
        money[day] = positive_number(row.get("Trading_money"))
    return money


def trailing_window(values, index, length):
    window = [values.get(i) for i in range(index - length + 1, index + 1)]
    return window if None not in window else None


def true_ranges(rows, quotes, closes, returns, audit, invalid_dates):
    ranges = {}
    for index, row in rows.items():
        if row["status"] == "prelisting":
            continue
        previous = rows.get(index - 1)
        quote = quotes.get(row["date"])
        prior_quote = quotes.get(previous["date"]) if previous else None
        if row["status"] == "missing_raw_session":
            reason = "missing_current_session"
        elif previous is None or previous["status"] in {"prelisting", "missing_raw_session"}:
            reason = "missing_previous_session"
        elif quote is None or prior_quote is None:
            reason = "invalid_raw_ohlcv"
        elif any(item["price_source"].startswith("imputed") for item in (row, previous)):
            reason = "imputed_price"
        elif "company_action" in row["review_reason"].split("|"):
            reason = "company_action_review"
        elif index not in returns or index not in closes or index - 1 not in closes:
            reason = "invalid_adjusted_price_or_return"
        else:
            _, high, low, close, _ = quote
            factor = closes[index] / close
            prior_close = closes[index - 1]
            if math.isclose(factor, prior_close / prior_quote[3], rel_tol=1e-4):
                high, low = high * factor, low * factor
                ranges[index] = max(high - low, abs(high - prior_close), abs(low - prior_close))
                reason = "valid"
            else:
                reason = "factor_mismatch"
        audit[f"TR_{reason}"] += 1
        if reason != "valid":
            previous = previous or {}
            invalid_dates.append({
                "code": row["code"], "date": row["date"], "session_index": index,
                "tr_exclusion_reason": reason, "status": row["status"],
                "review_reason": row["review_reason"], "price_source": row["price_source"],
                "previous_session_date": previous.get("date"),
                "previous_status": previous.get("status"),
                "previous_review_reason": previous.get("review_reason"),
                "previous_price_source": previous.get("price_source"),
            })
    return ranges


def build_features(code, rows, raw_rows, trading_money, atr_audit, invalid_dates):
    returns = feature_returns(rows)
    quotes = valid_quotes(raw_rows)
    closes, volumes, amounts = {}, {}, {}
    for index, row in rows.items():
        day = row["date"]
        if row["status"] == "prelisting":
            quotes.pop(day, None)
            continue
        if day in quotes:
            volumes[index] = quotes[day][4]
            amounts[index] = trading_money.get(day)
        if eligible_price(row):
            close = positive_number(row["adjusted_close"])
            if close is not None:
                closes[index] = close

    ranges = true_ranges(rows, quotes, closes, returns, atr_audit, invalid_dates)
    first_listed_index = min(
        (index for index, row in rows.items() if row["status"] != "prelisting"), default=None,
    )
    changes = {
        i: closes[i] - closes[i - 1]
        for i in returns if i in closes and i - 1 in closes
    }
    for index, row in sorted(rows.items()):
        day = row["date"]
        values = dict.fromkeys(FEATURE_COLUMNS)
        values["Return_1"] = returns.get(index)
        for length in (5, 10, 20):
            window = trailing_window(returns, index, length)
            if window is None:
                continue
            values[f"Volatility_{length}"] = (
                math.sqrt(config.ANNUAL_TRADING_DAYS / length) * math.hypot(*window)
            )
            if length in (5, 20):
                values[f"Return_{length}"] = math.fsum(window)
        if day in quotes and eligible_intraday_price(row):
            open_, high, low, close, _ = quotes[day]
            values["High_Low_Range"] = (high - low) / close
            values["Close_Open_Range"] = (close - open_) / open_

        for length in (5, 20):
            window = trailing_window(closes, index, length)
            if window is not None:
                values[f"MA_Gap_{length}"] = window[-1] / fmean(window) - 1
        window = trailing_window(volumes, index, 20)
        if window is not None:
            average = fmean(window)
            values["Volume_Ratio"] = window[-1] / average
            values["Volume_Volatility"] = pstdev(window) / average
        window = trailing_window(amounts, index, 20)
        if window is not None:
            values["Turnover"] = window[-1] / fmean(window)
        window = trailing_window(changes, index, 14)
        if window is not None:
            gains = math.fsum(max(value, 0) for value in window)
            losses = math.fsum(max(-value, 0) for value in window)
            values["RSI_14"] = 100 * gains / (gains + losses) if gains + losses else 50.0
        window = trailing_window(ranges, index, 14)
        if window is not None:
            values["ATR_14"] = fmean(window) / closes[index]
        elif row["status"] != "prelisting":
            reason = "warmup" if index - first_listed_index < 14 else "invalid_tr"
            atr_audit[f"ATR_window_{reason}"] += 1

        yield {"date": day, "code": code, "session_index": index, **values}


def audit_stock(code, rows, features, atr_audit):
    eligible = [row for row in features if rows[row["session_index"]]["status"] != "prelisting"]
    complete = sum(all(row[field] is not None for field in FEATURE_COLUMNS) for row in eligible)
    valid = {field: sum(row[field] is not None for row in eligible) for field in FEATURE_COLUMNS}
    statuses = Counter(row["status"] for row in rows.values())
    # 20 日報酬使用 t-20 到 t 的價格，補值可影響最長 21 個預測日。
    imputed_windows = {
        anchor
        for index, row in rows.items() if row["price_source"].startswith("imputed")
        for anchor in range(index, index + 21)
        if anchor in rows and rows[anchor]["status"] != "prelisting"
    }
    return {
        "code": code, "total_sessions": len(features), "complete_features": complete,
        "rows": len(features), "listed_sessions": len(eligible), "complete_rows": complete,
        "complete_coverage": complete / len(eligible) if eligible else None,
        "imputed_window_excluded": len(imputed_windows),
        **{f"sessions_{status}": statuses[status] for status in (
            "prelisting", "missing_raw_session", "duplicate_raw_session", "invalid_raw_quote",
            "missing_adjusted_close", "adjusted_jump_review", "nonconsecutive_sessions",
            "no_previous_valid_session",
        )},
        **{f"{field}_valid": valid[field] for field in FEATURE_COLUMNS},
        **{f"missing_{field}": len(eligible) - valid[field] for field in FEATURE_COLUMNS},
        **{f"TR_{reason}": atr_audit[f"TR_{reason}"] for reason in TR_REASONS},
        "ATR_window_warmup": atr_audit["ATR_window_warmup"],
        "ATR_window_invalid_tr": atr_audit["ATR_window_invalid_tr"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.PROCESSED_DATA_DIR / "adjusted_daily_returns.csv")
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    parser.add_argument("--source", choices=("finmind", "twse"), default="finmind")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("開始日期不可晚於結束日期")
    suffix = "_finmind" if args.source == "finmind" else ""
    try:
        groups, _ = load_returns(args.input)
        output, audit, invalid_dates = [], [], []
        for code, rows in sorted(groups.items()):
            if any(not args.start.isoformat() <= row["date"] <= args.end.isoformat() for row in rows.values()):
                raise ValueError("日報酬日期與指定研究區間不符")
            raw_path = config.RAW_DATA_DIR / f"{code}_{args.start}_{args.end}{suffix}.csv"
            atr_audit = Counter()
            features = list(build_features(
                code, rows, read_csv(raw_path), load_trading_money(code, args.start, args.end), atr_audit,
                invalid_dates,
            ))
            output.extend(features)
            audit.append(audit_stock(code, rows, features, atr_audit))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    path = config.FEATURES_DIR / "financial_features.csv"
    write_csv(path, ("date", "code", "session_index", *FEATURE_COLUMNS), output)
    write_csv(config.TABLES_DIR / "feature_audit.csv", list(audit[0]), audit)
    write_csv(
        config.TABLES_DIR / "atr_invalid_dates.csv", ATR_INVALID_COLUMNS,
        sorted(invalid_dates, key=lambda row: (row["code"], row["date"])),
    )
    complete = sum(row["complete_rows"] for row in audit)
    print(f"已輸出 {len(output)} 筆至 {path}，其中 {complete} 筆十五項齊全")


if __name__ == "__main__":
    main()
