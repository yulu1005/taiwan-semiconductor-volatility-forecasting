"""整合特徵與標籤，依時間切分並使用 Train、Validation 評估特徵。"""

import csv
import math
from collections import Counter
from datetime import date
from importlib import import_module
from itertools import combinations
from statistics import StatisticsError, correlation, fmean, linear_regression

import config


FEATURES = import_module("03_financial_features")
FIELDS = FEATURES.FEATURE_COLUMNS
TARGET = "future_5d_volatility"
SPLITS = {
    "train": (date(2016, 11, 1), date(2021, 12, 31)),
    "validation": (date(2022, 1, 1), date(2023, 12, 31)),
    "test": (date(2024, 1, 1), date(2026, 11, 1)),
}
MODEL_FIELDS = ("date", "code", "session_index", *FIELDS, TARGET, "label_start_date", "label_end_date")
MERGED_FIELDS = (*MODEL_FIELDS, "has_feature_row", "has_label_row", "features_complete", "label_valid",
                 "label_reason", "valid", "listed", "split", "purged")


def load_keyed(path, code_field, required):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"date", code_field, *required} <= set(reader.fieldnames or []):
            raise ValueError(f"{path.name} 缺少必要欄位")
        for row in reader:
            day = date.fromisoformat(row["date"])
            code = row[code_field]
            key = (day, code)
            if key in result or code not in config.STOCK_CODES or row["date"] != day.isoformat():
                raise ValueError(f"{path.name} 日期、代號無效或鍵重複 {key}")
            result[key] = row
    if not result:
        raise ValueError(f"{path.name} 無資料")
    return result


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def merged_rows(feature_rows, label_rows):
    calendar = {}
    for (day, _), row in feature_rows.items():
        index = int(row["session_index"])
        if index < 1 or calendar.setdefault(index, day) != day:
            raise ValueError(f"交易日索引不一致 {index} {day}")
    ordered_days = [calendar[index] for index in sorted(calendar)]
    if ordered_days != sorted(set(ordered_days)) or sorted(calendar) != list(range(1, len(calendar) + 1)):
        raise ValueError("交易日曆必須連續編號且日期嚴格遞增")
    indices = {day: index for index, day in calendar.items()}

    for day, code in sorted(feature_rows.keys() | label_rows.keys()):
        feature = feature_rows.get((day, code))
        label = label_rows.get((day, code))
        values = {name: number(feature.get(name)) if feature else None for name in FIELDS}
        listed = day >= FEATURES.feasibility.LISTING_DATES[code]
        complete = listed and all(value is not None for value in values.values())
        target = number(label[TARGET]) if label else None
        valid_label = False
        label_end = None
        split = next((name for name, (start, end) in SPLITS.items() if start <= day <= end), None)
        if split is None:
            raise ValueError(f"預測日期超出正式研究區間 {day} {code}")
        if label:
            if label["label_valid"] not in {"0", "1"}:
                raise ValueError(f"無效的 label_valid {day} {code}")
            if label["label_valid"] == "1":
                index = indices.get(day)
                start = date.fromisoformat(label["label_start_date"])
                label_end = date.fromisoformat(label["label_end_date"])
                if (not listed or target is None or target < 0 or index is None or not day < start <= label_end
                        or start != calendar.get(index + 1)
                        or label_end != calendar.get(index + config.FORECAST_HORIZON)):
                    raise ValueError(f"有效 Label 的數值或五日時間對齊異常 {day} {code}")
                valid_label = True
        candidate = complete and valid_label
        yield {
            "date": day.isoformat(), "code": code, "session_index": indices.get(day), **values,
            TARGET: target, "label_start_date": label["label_start_date"] if label else "",
            "label_end_date": label["label_end_date"] if label else "",
            "has_feature_row": int(feature is not None), "has_label_row": int(label is not None),
            "features_complete": int(complete), "label_valid": int(valid_label),
            "label_reason": label["label_reason"] if label else "missing_label_row",
            "valid": int(candidate), "listed": int(listed), "split": split,
            "purged": int(candidate and label_end > SPLITS[split][1]),
        }


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def safe_correlation(x, y):
    try:
        return correlation(x, y)
    except StatisticsError:
        return None


def audit_row(scope, code, rows):
    reasons = Counter(row["label_reason"] for row in rows if row["listed"] and not row["label_valid"])
    candidates = sum(row["valid"] for row in rows)
    purged = sum(row["purged"] for row in rows)
    return {
        "split": scope, "code": code, "total_rows": len(rows),
        "listed_sessions": sum(row["listed"] for row in rows),
        "feature_rows": sum(row["has_feature_row"] for row in rows),
        "label_rows": sum(row["has_label_row"] for row in rows),
        "unmatched_rows": sum(not (row["has_feature_row"] and row["has_label_row"]) for row in rows),
        "complete_features": sum(row["features_complete"] for row in rows),
        "valid_labels": sum(row["label_valid"] for row in rows),
        "candidates_before_purge": candidates, "purged": purged, "retained": candidates - purged,
        "invalid_label_reasons_listed": "|".join(f"{key}={value}" for key, value in sorted(reasons.items())),
    }


def coverage_rows(rows, scope, code):
    listed = [row for row in rows if row["listed"]]
    for field in FIELDS:
        valid = sum(row[field] is not None for row in listed)
        yield {
            "split": scope, "code": code, "feature": field, "listed_sessions": len(listed),
            "valid": valid, "missing": len(listed) - valid, "coverage": valid / len(listed) if listed else None,
        }


def evaluate(train, validation):
    if len(train) < 2 or not validation:
        raise ValueError("Purging 後 Train 或 Validation 樣本不足")
    train_values = {name: [row[name] for row in train] for name in (*FIELDS, TARGET)}
    ranked = {name: ranks(values) for name, values in train_values.items()}

    def pair(a, b):
        return {
            "feature": a, "comparison": b, "split": "train", "samples": len(train),
            "pearson": safe_correlation(train_values[a], train_values[b]),
            "spearman": safe_correlation(ranked[a], ranked[b]),
        }

    correlations = [pair(name, TARGET) for name in FIELDS]
    redundancy = [pair(a, b) for a, b in combinations(FIELDS, 2)]
    observed = [row[TARGET] for row in validation]
    baseline = fmean(abs(row["Volatility_5"] - row[TARGET]) for row in validation)
    models = [{
        "feature": "Volatility_5", "method": "persistence_baseline", "train_samples": 0,
        "validation_samples": len(validation), "validation_mae": baseline,
        "mae_minus_baseline": 0.0, "intercept": None, "slope": None,
    }]
    for name in FIELDS:
        x, y = train_values[name], train_values[TARGET]
        try:
            slope, intercept = linear_regression(x, y)
        except StatisticsError:
            slope, intercept = 0.0, fmean(y)
        error = fmean(abs(max(0.0, intercept + slope * row[name]) - actual)
                      for row, actual in zip(validation, observed, strict=True))
        models.append({
            "feature": name, "method": "single_feature_ols", "train_samples": len(train),
            "validation_samples": len(validation), "validation_mae": error,
            "mae_minus_baseline": error - baseline, "intercept": intercept, "slope": slope,
        })
    return correlations, redundancy, models


def main():
    feature_rows = load_keyed(config.FEATURES_DIR / "financial_features.csv", "code",
                              ("session_index", *FIELDS))
    labels = load_keyed(config.PROCESSED_DATA_DIR / "labels.csv", "symbol",
                        (TARGET, "label_valid", "label_reason", "label_start_date", "label_end_date"))
    full = list(merged_rows(feature_rows, labels))
    complete_split = {name: [row for row in full if row["split"] == name] for name in SPLITS}
    retained = {name: [row for row in rows if row["valid"] and not row["purged"]]
                for name, rows in complete_split.items()}
    correlations, redundancy, scores = evaluate(retained["train"], retained["validation"])

    audit, coverage = [], []
    for scope, rows in {"all": full, **complete_split}.items():
        for code in ("ALL", *config.STOCK_CODES):
            subset = rows if code == "ALL" else [row for row in rows if row["code"] == code]
            audit.append(audit_row(scope, code, subset))
            coverage.extend(coverage_rows(subset, scope, code))

    write = FEATURES.write_csv
    write(config.PROCESSED_DATA_DIR / "merged_feature_labels.csv", MERGED_FIELDS, full)
    model_dataset = [row for rows in retained.values() for row in rows]
    for name, rows in {"model_dataset": model_dataset, **retained}.items():
        write(config.PROCESSED_DATA_DIR / f"{name}.csv", MODEL_FIELDS,
              ({field: row[field] for field in MODEL_FIELDS} for row in rows))
    for name, rows in (
        ("dataset_audit", audit), ("feature_coverage", coverage), ("feature_correlation", correlations),
        ("feature_feature_correlation", redundancy), ("feature_evaluation", scores),
    ):
        write(config.TABLES_DIR / f"{name}.csv", list(rows[0]), rows)
    print(f"完整合併 {len(full)}、Purging 後 {len(model_dataset)}；切分保留 "
          + "、".join(f"{name} {len(rows)}" for name, rows in retained.items()))


if __name__ == "__main__":
    main()
