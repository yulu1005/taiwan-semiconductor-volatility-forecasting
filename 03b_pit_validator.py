"""驗證特徵日曆、歷史視窗、截斷重算及來源可得性。"""

import argparse
import math
from collections import Counter, defaultdict
from datetime import date
from importlib import import_module
from inspect import signature
from pathlib import Path
from statistics import StatisticsError, correlation

import config


features = import_module("03_financial_features")
read_csv, write_csv = features.read_csv, features.write_csv
FIELDS = features.FEATURE_COLUMNS
REPORT_FIELDS = ("code", "feature", "test", "result", "observations", "detail")


def agree(observed, expected):
    if expected is None:
        return observed in ("", None)
    try:
        return math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-12)
    except (TypeError, ValueError, OverflowError):
        return False


def validate_order(published, calendar):
    dates = [date.fromisoformat(row["date"]) for row in published]
    ordered = all(a < b for a, b in zip(dates, dates[1:]))
    actual = Counter((int(row["session_index"]), row["date"]) for row in published)
    expected = Counter(calendar.items())
    return ordered, sum((expected - actual).values()), sum((actual - expected).values())


def historical_values(daily, raw, money):
    """獨立重算公式，沿用 03 的原始 OHLCV 品質檢查。"""
    quotes = features.valid_quotes(raw)
    closes, volumes, amounts, intraday = {}, {}, {}, {}
    for i, row in daily.items():
        day, status, source = row["date"], row["status"], row["price_source"]
        tags = row["review_reason"].split("|")
        if row["price_valid"] == "1" and source == "yahoo":
            price = features.positive_number(row["adjusted_close"])
            if price is not None:
                closes[i] = price
        if day not in quotes or status == "prelisting":
            continue
        opening, high, low, close, volume = quotes[day]
        volumes[i] = volume
        if money.get(day) is not None:
            amounts[i] = money[day]
        if (i in closes and status in {"ok", "no_previous_valid_session", "nonconsecutive_sessions"}
                and "adjusted_cross_gap_review" not in tags):
            intraday[i] = ((high - low) / close, (close - opening) / opening)

    returns, changes, tr = {}, {}, {}
    for i, current in closes.items():
        row = daily[i]
        if i - 1 not in closes or row["status"] != "ok":
            continue
        previous = closes[i - 1]
        try:
            approved_return = float(row["log_return"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(approved_return):
            continue
        returns[i] = approved_return
        changes[i] = current - previous
        if ("company_action" in row["review_reason"].split("|")
                or row["date"] not in quotes or daily[i - 1]["date"] not in quotes):
            continue
        _, high, low, close, _ = quotes[row["date"]]
        factor = current / close
        if math.isclose(factor, previous / quotes[daily[i - 1]["date"]][3], rel_tol=1e-4):
            bounds = (high * factor, low * factor, previous)
            tr[i] = max(bounds) - min(bounds)

    def window(series, i, length):
        keys = range(i - length + 1, i + 1)
        return [series[j] for j in keys] if all(j in series for j in keys) else []

    for i in sorted(daily):
        values = dict.fromkeys(FIELDS)
        values["Return_1"] = returns.get(i)
        for length in (5, 10, 20):
            xs = window(returns, i, length)
            if xs:
                values[f"Volatility_{length}"] = math.sqrt(
                    config.ANNUAL_TRADING_DAYS * sum(x * x for x in xs) / length
                )
                if length != 10:
                    values[f"Return_{length}"] = sum(xs)
        if i in intraday:
            values["High_Low_Range"], values["Close_Open_Range"] = intraday[i]
        for length in (5, 20):
            xs = window(closes, i, length)
            if xs:
                values[f"MA_Gap_{length}"] = length * xs[-1] / sum(xs) - 1
        xs = window(volumes, i, 20)
        if xs:
            mean = sum(xs) / 20
            values["Volume_Ratio"] = xs[-1] / mean
            values["Volume_Volatility"] = math.sqrt(sum((x - mean) ** 2 for x in xs) / 20) / mean
        xs = window(amounts, i, 20)
        if xs:
            values["Turnover"] = 20 * xs[-1] / sum(xs)
        xs = window(changes, i, 14)
        if xs:
            magnitude = sum(map(abs, xs))
            values["RSI_14"] = 100 * sum(x for x in xs if x > 0) / magnitude if magnitude else 50
        xs = window(tr, i, 14)
        if xs:
            values["ATR_14"] = sum(xs) / (14 * closes[i])
        yield i, values


def cutoff_sessions(daily):
    indices = sorted(daily)
    listed = [i for i in indices if daily[i]["status"] != "prelisting"]
    if not listed:
        return {indices[-1]}
    cutoffs = {listed[k * (len(listed) - 1) // 8] for k in range(9)}
    cutoffs.update(i for i in indices if daily[i]["date"] == "2024-06-14")
    for i in listed:
        if daily[i]["price_source"].startswith("imputed"):
            cutoffs.update(j for j in (i - 1, i, i + 1, i + 14, i + 20, i + 21) if j in daily)
    gaps = [i for i in listed if daily[i]["status"] == "missing_raw_session"]
    if gaps:
        end = next((i for i in listed if i > gaps[0] and daily[i]["status"] != "missing_raw_session"), None)
        cutoffs.add(gaps[0])
        if end is not None:
            cutoffs.add(end)
    return cutoffs


def check_future_leakage(published, daily, name):
    """Pearson 為尺度不變的篩檢。尋找近乎等於次日價格或次日價格變動的特徵。"""
    observations = []
    for i, row in published.items():
        if i + 1 not in daily:
            continue
        current = features.positive_number(daily[i]["adjusted_close"])
        following = features.positive_number(daily[i + 1]["adjusted_close"])
        if not row[name] or current is None or following is None:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            observations.append((value, current, following, math.log(following / current)))
    if len(observations) < 30:
        return "REVIEW", len(observations), "可配對樣本少於 30"
    series, current, following, change = zip(*observations)

    def pearson(other):
        try:
            return correlation(series, other)
        except StatisticsError:
            return None

    now, lead, lead_change = (pearson(x) for x in (current, following, change))
    suspicious = (
        lead_change is not None and abs(lead_change) >= 0.90
        or lead is not None and now is not None and abs(lead) >= 0.98 and abs(lead) - abs(now) >= 0.05
    )
    coefficients = [f"{value:.6f}" if value is not None else "NA" for value in (now, lead, lead_change)]
    result = "REVIEW" if suspicious or all(value is None for value in (now, lead, lead_change)) else "PASS"
    return result, len(observations), "corr_t, corr_t+1, corr_return_t+1 = " + ", ".join(coefficients)


IMPUTED_PRICE_LOOKBACK = {
    "Volatility_5": 5, "Volatility_10": 10, "Volatility_20": 20,
    "Return_1": 1, "Return_5": 5, "Return_20": 20,
    "MA_Gap_5": 4, "MA_Gap_20": 19, "RSI_14": 14, "ATR_14": 14,
}


def recalculate_prefix(code, daily, raw, money, index):
    day = daily[index]["date"]
    # 產出 ATR 排除清單的 03 版本需要額外的收集器。
    options = {}
    if "invalid_dates" in signature(features.build_features).parameters:
        options["invalid_dates"] = []
    return features.build_features(
        code, {i: row for i, row in daily.items() if i <= index},
        [row for row in raw if row["date"] <= day],
        {key: value for key, value in money.items() if key <= day}, Counter(), **options,
    )


def validate_stock(code, published, daily, calendar, raw, money):
    def report(feature, test, result, observations, detail):
        return dict(zip(REPORT_FIELDS, (code, feature, test, result, observations, detail), strict=True))

    ordered, missing, extra = validate_order(published, calendar)
    yield report("*", "Monotonic Dates", "PASS" if ordered else "FAIL", len(published), "逐股嚴格遞增")
    daily_calendar = {i: row["date"] for i, row in daily.items()}
    calendar_ok = missing == extra == 0 and daily_calendar == calendar
    yield report("*", "Session Calendar", "PASS" if calendar_ok else "FAIL", len(calendar),
                 f"缺少 {missing}、額外或重複 {extra}；01c 與參考日曆一致={daily_calendar == calendar}")
    if not calendar_ok:
        return

    by_index = {int(row["session_index"]): row for row in published}
    errors = defaultdict(list)
    for i, expected in historical_values(daily, raw, money):
        for name, value in expected.items():
            if not agree(by_index[i][name], value):
                errors[name].append(daily[i]["date"])

    truncated_errors = defaultdict(list)
    imputed_indices = {i for i, row in daily.items() if row["price_source"].startswith("imputed")}
    cutoffs = sorted(cutoff_sessions(daily))
    for index in cutoffs:
        day = daily[index]["date"]
        cutoff_rows = recalculate_prefix(code, daily, raw, money, index)
        last = None
        for last in cutoff_rows:
            pass
        for name in FIELDS:
            if last is None or not agree(by_index[index][name], last[name]):
                truncated_errors[name].append(day)

    for name in FIELDS:
        for test, failures, count in (
            ("Historical Window", errors[name], len(published)),
            ("Truncated Recalculation", truncated_errors[name], len(cutoffs)),
        ):
            yield report(name, test, "FAIL" if failures else "PASS", count,
                         f"差異 {len(failures)}" + (f"；首筆 {failures[0]}" if failures else ""))
        result, count, detail = check_future_leakage(by_index, daily, name)
        yield report(name, "Future Correlation Screen", result, count, detail)

        if name in {"Volume_Ratio", "Volume_Volatility", "Turnover"}:
            source = "FinMind 原始成交資料；無逐日發布版本與時間快照"
        elif name in {"High_Low_Range", "Close_Open_Range"}:
            source = "FinMind 原始 OHLCV 與 Yahoo 可用性篩選；無逐日發布時間與歷史版本"
        else:
            source = "Yahoo 調整後收盤價；無歷史版本可核對追溯調整"
        yield report(name, "Data Availability", "REVIEW", None, source)
        if imputed_indices and name in IMPUTED_PRICE_LOOKBACK:
            affected = {
                i for seed in imputed_indices
                for i in range(seed, seed + IMPUTED_PRICE_LOOKBACK[name] + 1) if i in by_index
            }
            present = sum(bool(by_index[i][name]) for i in affected)
            yield report(name, "Imputed Price", "FAIL" if present else "PASS", len(affected),
                         f"有事後補值的價格視窗應留空；非空 {present}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.FEATURES_DIR / "financial_features.csv")
    parser.add_argument("--returns", type=Path, default=config.PROCESSED_DATA_DIR / "adjusted_daily_returns.csv")
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    args = parser.parse_args()
    expected_module = Path(__file__).with_name("03_financial_features.py").resolve()
    if (Path(features.__file__).resolve() != expected_module or len(FIELDS) != 15
            or not {"Turnover", "ATR_14", "Volatility_20"}.issubset(FIELDS)):
        parser.error(f"請使用同目錄的 15 項台股 Feature 程式 {expected_module}")
    if args.start > args.end:
        parser.error("開始日期不可晚於結束日期")

    def raw_prices(code):
        return read_csv(config.RAW_DATA_DIR / f"{code}_{args.start}_{args.end}_finmind.csv")

    try:
        daily, _ = features.load_returns(args.returns)
        published = defaultdict(list)
        for row in read_csv(args.features):
            published[row["code"]].append(row)
        dates = sorted({row["date"] for row in raw_prices(config.REFERENCE_STOCK)})
        if not dates:
            raise ValueError("參考交易日曆為空")
        calendar = dict(enumerate(dates, start=1))
        report = []
        for code in sorted(set(config.STOCK_CODES) | daily.keys() | published.keys()):
            if code not in config.STOCK_CODES or code not in daily or not published[code]:
                report.append({"code": code, "feature": "*", "test": "Session Calendar", "result": "FAIL",
                               "detail": "股票代號未配置，或缺少日報酬／特徵資料"})
                continue
            report.extend(validate_stock(
                code, published[code], daily[code], calendar, raw_prices(code),
                features.load_trading_money(code, args.start, args.end),
            ))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        parser.error(str(error))

    counts = defaultdict(Counter)
    for item in report:
        counts[item["test"]][item["result"]] += 1
    summary = [
        {"test": test, **{status: totals[status] for status in ("PASS", "FAIL", "REVIEW")}}
        for test, totals in counts.items()
    ]
    write_csv(config.TABLES_DIR / "pit_validation_report.csv", REPORT_FIELDS, report)
    write_csv(config.TABLES_DIR / "pit_validation_summary.csv", ("test", "PASS", "FAIL", "REVIEW"), summary)
    return int(any(totals["FAIL"] for totals in counts.values()))


if __name__ == "__main__":
    raise SystemExit(main())
