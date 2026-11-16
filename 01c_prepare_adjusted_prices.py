"""對齊原始行情與調整後價格，建立連續交易日的對數報酬。"""

import argparse
import csv
import math
from collections import Counter
from datetime import date
from importlib import import_module
from pathlib import Path

import config


feasibility = import_module("01_feasibility_analysis")
VERIFIED_ACTIONS = {
    ("2340", "2019-09-11"): (
        "verified_ratio_0.85_cash_1.5",
        "https://www.tascsemi.com/wp-content/uploads/2022/04/108%E5%B9%B4_%E8%AD%B0%E4%BA%8B%E9%8C%84.pdf",
    ),
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    feasibility.write_csv(path, columns, rows)


def number(value):
    try:
        value = float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def load_adjusted(path, start, end):
    rows = read_csv(path)
    if not rows or not {"date", "code", "adjusted_close"} <= set(rows[0]):
        raise ValueError("調整後價格 CSV 必須包含 date、code、adjusted_close")
    result = {code: {} for code in config.STOCK_CODES}
    for row in rows:
        code, day = row["code"].strip(), date.fromisoformat(row["date"].strip())
        if code in result and start <= day <= end:
            key = day.isoformat()
            if key in result[code]:
                raise ValueError(f"調整後價格日期重複 {code} {key}")
            result[code][key] = row
    return result


def corporate_action_notes():
    path = config.TABLES_DIR / "data_quality_review.csv"
    if not path.is_file():
        return {}
    return {
        (row["code"], row["next_valid_date"]): row["source_url"]
        for row in read_csv(path)
        if row["issue"] == "missing_session" and row["finding"] == "減資換股暫停交易"
    }


def load_decisions(path, start, end):
    decisions = {}
    if path is None:
        return decisions
    for row in read_csv(path):
        day = date.fromisoformat(row["date"])
        if start <= day <= end:
            key = row["code"], day.isoformat()
            if key in decisions or not row["reason"].strip() or not row["source_url"].strip():
                raise ValueError(f"審核紀錄重複或缺少依據 {key}")
            decisions[key] = row
    return decisions


def checked_override(record, raw_close, neighbors, yahoo, actions):
    day = record["date"]
    if neighbors != (record["previous_date"], record["next_date"]):
        raise ValueError(f"{day} 補值日期與相鄰交易日不符")
    factors = []
    for adjacent in neighbors:
        row = yahoo.get(adjacent, {})
        close, adjusted = number(row.get("yahoo_close")), number(row.get("adjusted_close"))
        if close is None or adjusted is None:
            raise ValueError(f"{day} 找不到相鄰 Yahoo 調整因子")
        factors.append(adjusted / close)
    for adjacent in (*neighbors, day):
        event = actions.get(adjacent, {})
        if any(event.get(field) not in {"0", "0.0"} for field in ("dividends", "stock_splits")):
            raise ValueError(f"{day} 鄰近行情有公司行動或缺少事件紀錄")
    expected = [number(record[field]) for field in ("previous_factor", "next_factor", "raw_close")]
    if (None in expected or not math.isclose(*factors, rel_tol=1e-6)
            or not all(math.isclose(actual, saved, rel_tol=1e-10)
                       for actual, saved in zip((*factors, raw_close), expected))):
        raise ValueError(f"{day} 來源價格或調整因子已變更，請重審補值")
    factor = sum(factors) / 2
    adjusted = number(record["adjusted_close"])
    saved_factor = number(record["adjustment_factor"])
    if (adjusted is None or saved_factor is None or not math.isclose(factor, saved_factor, rel_tol=1e-10)
            or not math.isclose(adjusted, raw_close * factor, rel_tol=1e-10)):
        raise ValueError(f"{day} 補值與核准調整因子不符")
    return adjusted


def prepare_stock(code, raw, yahoo, reference, action_notes, overrides, actions, verified_jumps):
    listing = feasibility.LISTING_DATES[code].isoformat()
    counts = Counter(row["date"] for row in raw)
    raw_by_date = {row["date"]: row for row in raw}
    last_valid = None
    output, discrepancies = [], []
    stats = Counter()
    for position, day in enumerate(reference, start=1):
        quote = raw_by_date.get(day, {})
        external = yahoo.get(day, {})
        raw_open, raw_high, raw_low, raw_close, volume = (
            number(quote.get(field)) for field in feasibility.PRICE_FIELDS
        )
        yahoo_close = number(external.get("yahoo_close"))
        adj = number(external.get("adjusted_close"))
        dividend = number(external.get("dividends"))
        split = number(external.get("stock_splits"))
        action_source = action_notes.get((code, day), "")
        action_review, verification_source = VERIFIED_ACTIONS.get((code, day), ("", ""))
        action_source = verification_source or action_source
        override = overrides.get((code, day))
        verified_jump = verified_jumps.get((code, day))
        price_source = "yahoo" if adj is not None else "missing"
        status = "ok"
        if day < listing:
            status = "prelisting"
        elif not quote:
            status = "missing_raw_session"
        elif counts[day] > 1:
            status = "duplicate_raw_session"
        elif None in (raw_open, raw_high, raw_low, raw_close, volume) or not (
            raw_low <= min(raw_open, raw_close) <= max(raw_open, raw_close) <= raw_high
            and volume.is_integer()
        ):
            status = "invalid_raw_quote"
        elif adj is None:
            status = "missing_adjusted_close"
            if override and 1 < position < len(reference):
                adj = checked_override(
                    override, raw_close, (reference[position - 2], reference[position]), yahoo, actions,
                )
                price_source = "imputed_from_finmind_factor"
                status = "ok"

        price_valid = status == "ok"
        gap = change = log_return = None
        if price_valid:
            if last_valid is not None:
                prior_position, prior_adjusted = last_valid
                gap = position - prior_position - 1
                change = adj / prior_adjusted - 1
            if gap != 0:
                status = "nonconsecutive_sessions" if gap else "no_previous_valid_session"
            elif abs(change) > config.PRICE_JUMP_THRESHOLD and not verified_jump:
                status = "adjusted_jump_review"
            else:
                log_return = math.log(adj) - math.log(prior_adjusted)
            last_valid = position, adj

        issues = []
        if day >= listing and status in {
            "missing_raw_session", "duplicate_raw_session", "invalid_raw_quote", "missing_adjusted_close",
        }:
            issues.append(status)
        if day >= listing and yahoo_close is None:
            issues.append("missing_yahoo_close")
        if raw_close is not None and yahoo_close is not None and not math.isclose(
            yahoo_close, raw_close, rel_tol=1e-5, abs_tol=0.001,
        ):
            issues.append("close_mismatch")
        if change is not None and abs(change) > config.PRICE_JUMP_THRESHOLD:
            issues.append("adjusted_cross_gap_review" if gap else
                          "verified_large_return" if verified_jump else "adjusted_jump_review")
        if price_source == "imputed_from_finmind_factor":
            issues.append(price_source)
        if dividend or split or action_source:
            issues.append("company_action")
        stats.update(issues)
        result = {
            "date": day, "code": code, "session_index": position, "raw_close": raw_close,
            "adjusted_close": adj, "log_return": log_return, "price_valid": int(price_valid),
            "status": status, "price_source": price_source, "gap_sessions": gap,
            "adjusted_change_pct": 100 * change if change is not None else None,
            "review_reason": "|".join(issues),
        }
        output.append(result)
        if issues:
            discrepancies.append({
                "date": day, "code": code, "issues": "|".join(issues), "status": status,
                "raw_close": raw_close, "yahoo_close": yahoo_close, "adjusted_close": adj,
                "yahoo_to_raw_ratio": yahoo_close / raw_close if yahoo_close and raw_close else None,
                "dividends": dividend, "stock_splits": split, "gap_sessions": gap,
                "adjusted_change_pct": 100 * change if change is not None else None,
                "company_action_source": action_source,
                "company_action_review": action_review,
                "price_source": price_source,
                "decision_reason": (override["reason"] if price_source == "imputed_from_finmind_factor"
                                    else verified_jump["reason"] if "verified_large_return" in issues else ""),
                "decision_source": (override["source_url"] if price_source == "imputed_from_finmind_factor"
                                    else verified_jump["source_url"] if "verified_large_return" in issues else ""),
            })

    for day in sorted(yahoo.keys() - set(reference)):
        stats["yahoo_outside_reference"] += 1
        discrepancies.append({
            "date": day, "code": code, "issues": "yahoo_outside_reference",
            "yahoo_close": yahoo[day].get("yahoo_close"), "adjusted_close": yahoo[day]["adjusted_close"],
        })
    valid_dates = [row["date"] for row in output if row["price_valid"]]
    summary = {
        "code": code,
        "first_usable_date": valid_dates[0] if valid_dates else "",
        "last_usable_date": valid_dates[-1] if valid_dates else "",
        "reference_sessions": sum(day >= listing for day in reference),
        "valid_adjusted_sessions": len(valid_dates),
        "valid_returns": sum(row["log_return"] is not None for row in output),
        **{key: stats[key] for key in (
            "missing_raw_session", "duplicate_raw_session", "invalid_raw_quote",
            "missing_adjusted_close", "missing_yahoo_close", "close_mismatch",
            "adjusted_jump_review", "adjusted_cross_gap_review", "company_action",
            "yahoo_outside_reference", "imputed_from_finmind_factor", "verified_large_return",
        )},
    }
    return output, summary, discrepancies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjusted", type=Path, default=config.ADJUSTED_CLOSE_PATH)
    parser.add_argument("--start", type=date.fromisoformat, default=config.START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=config.END_DATE)
    parser.add_argument("--source", choices=("finmind", "twse"), default="finmind")
    parser.add_argument("--overrides", type=Path, default=config.ADJUSTED_PRICE_OVERRIDES_PATH)
    parser.add_argument("--verified-jumps", type=Path, help="已核對的真實漲跌 CSV，含 date,code,reason,source_url")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("開始日期不可晚於結束日期")
    suffix = "_finmind" if args.source == "finmind" else ""
    try:
        yahoo = load_adjusted(args.adjusted, args.start, args.end)
        raw = {code: read_csv(config.RAW_DATA_DIR / f"{code}_{args.start}_{args.end}{suffix}.csv")
               for code in config.STOCK_CODES}
        notes = corporate_action_notes()
        overrides = load_decisions(args.overrides, args.start, args.end)
        verified_jumps = load_decisions(args.verified_jumps, args.start, args.end)
        actions = {}
        if overrides:
            for row in read_csv(config.EXTERNAL_DATA_DIR / "yahoo_daily.csv"):
                actions.setdefault(row["code"], {})[row["date"]] = row
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    reference = sorted({row["date"] for row in raw[config.REFERENCE_STOCK]})
    if not reference:
        parser.error("參考股票沒有交易日資料")
    output, summary, details = [], [], []
    for code in config.STOCK_CODES:
        try:
            daily, audit, discrepancies = prepare_stock(
                code, raw[code], yahoo[code], reference, notes, overrides, actions.get(code, {}), verified_jumps,
            )
        except (ValueError, KeyError) as error:
            parser.error(f"{code} {error}")
        output.extend(daily)
        summary.append(audit)
        details.extend(discrepancies)
    write_csv(config.PROCESSED_DATA_DIR / "adjusted_daily_returns.csv", output, list(output[0]))
    write_csv(config.TABLES_DIR / "adjusted_price_audit.csv", summary, list(summary[0]))
    write_csv(config.TABLES_DIR / "adjusted_price_discrepancies.csv", details, (
        "date", "code", "issues", "status", "raw_close", "yahoo_close", "adjusted_close",
        "yahoo_to_raw_ratio", "dividends", "stock_splits", "gap_sessions", "adjusted_change_pct",
        "company_action_source", "company_action_review", "price_source", "decision_reason", "decision_source",
    ))
    print(f"已輸出 {len(output)} 筆日報酬資料與 {len(details)} 筆比對明細")


if __name__ == "__main__":
    main()
