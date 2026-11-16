"""使用封存模型在 Test 上預測，並與兩種波動率基準比較。"""

import json
import math
from hashlib import sha256
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


training = import_module("05_model_training")
validation = import_module("04b_dataset_validation")
TEST_PERIOD = next((start, end) for name, start, end in validation.PERIODS if name == "test")
ACTUAL = "actual_5d_volatility"
PREDICTED = "predicted_5d_volatility"
HISTORICAL = "historical_5d_volatility"
TRAIN_MEAN = "train_mean_5d_volatility"
MIN_RANKING_STOCKS = 3


def check_provenance(info, first_test_date):
    selections = info["selection_splits"]
    if set(selections) != set(training.PERIODS):
        raise ValueError("05 的模型選擇來源應為 Train 與 Validation")
    if not (info["refit_anchor_start"] <= info["refit_anchor_end"] < info["refit_label_end"]
            <= training.PERIODS["validation"][1] < first_test_date):
        raise ValueError("模型重訓日期或 Label 結束日與 Test 重疊")
    if info["refit_rows"] != sum(item["rows"] for item in selections.values()):
        raise ValueError("模型重訓筆數與來源筆數不符")

    for split, item in selections.items():
        expected_file = f"{split}.csv"
        if item["file"] != expected_file or tuple(item["period"]) != training.PERIODS[split]:
            raise ValueError(f"{split} 模型來源或期間不符")
        path = config.PROCESSED_DATA_DIR / expected_file
        if sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"目前的 {expected_file} 與模型中保存的來源雜湊不符")

    comparison = pd.read_csv(config.TABLES_DIR / "model_comparison.csv")
    selected = comparison.loc[comparison["selected"].eq(1)]
    if len(selected) != 1:
        raise ValueError("05 的 Validation 比較結果須恰有一個選定模型")
    row = selected.iloc[0]
    if (
        (row["model"], row["feature_set"], row["trial_id"])
        != (info["model"], info["feature_set"], info["selection_trial"]["trial_id"])
        or any(not math.isclose(float(row[field]), info[field], rel_tol=0, abs_tol=1e-12)
               for field in ("validation_mae", "validation_rmse"))
    ):
        raise ValueError("05 Validation 報告與最終模型設定不符")


def load_test():
    audit = {}
    rows = validation.read_rows(
        config.TABLES_DIR / "dataset_audit.csv", {"split", "code", "retained"},
    )
    for row in rows:
        if row["split"] == "test":
            if (row["split"], row["code"]) in audit:
                raise ValueError("Test 稽核中的股票代號重複")
            audit[row["split"], row["code"]] = int(row["retained"])
    calendar = validation.reference_calendar()
    validation.validate_split("test", *TEST_PERIOD, calendar, audit, set())
    test = training.load_split(
        config.PROCESSED_DATA_DIR / "test.csv", tuple(day.isoformat() for day in TEST_PERIOD),
    )
    return test, calendar


def next_execution_dates(test, calendar):
    indices = pd.to_numeric(test["session_index"], errors="coerce")
    if not np.isfinite(indices.to_numpy(dtype=float)).all() or (indices % 1 != 0).any():
        raise ValueError("Test 交易日索引須為有效整數")
    if (indices < 1).any() or not (indices + 1).isin(calendar.keys()).all():
        raise ValueError("Test 的下一交易日超出交易日曆範圍")
    dates = indices.map(lambda index: calendar[int(index) + 1].isoformat())
    if not (dates > test["date"]).all():
        raise ValueError("下一交易日須晚於預測日")
    return dates


def error_metrics(actual, prediction):
    residual = actual - prediction
    return {
        "test_mae": float(np.mean(np.abs(residual))),
        "test_rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def improvement(baseline_mae, model_mae):
    return 100 * (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0 else None


def stock_metrics(predictions):
    rows = []
    for code, group in predictions.groupby("code", sort=True):
        actual = group[ACTUAL].to_numpy()
        score = error_metrics(actual, group[PREDICTED].to_numpy())
        baseline = error_metrics(actual, group[HISTORICAL].to_numpy())
        rows.append({
            "code": code, "test_samples": len(group), **score,
            "baseline_mae": baseline["test_mae"], "baseline_rmse": baseline["test_rmse"],
            "mae_improvement": improvement(baseline["test_mae"], score["test_mae"]),
        })
    return pd.DataFrame(rows)


def volatility_regimes(predictions, model_name):
    actual = predictions[ACTUAL]
    cutoff = float(actual.quantile(0.90))
    rows = []
    # 最高一成與超過 100% 的分組可以重疊，分類僅用於事後評估。
    for regime, group in (
        ("below_p90", predictions.loc[actual < cutoff]),
        ("top_decile", predictions.loc[actual >= cutoff]),
        ("above_100pct", predictions.loc[actual > 1.0]),
    ):
        realized = group[ACTUAL].to_numpy()
        for model, field in ((model_name, PREDICTED),
                             ("Historical Volatility Persistence", HISTORICAL),
                             ("Train Mean", TRAIN_MEAN)):
            forecast = group[field].to_numpy()
            rows.append({
                "regime": regime, "actual_p90_cutoff": cutoff, "model": model,
                "test_samples": len(group),
                **(error_metrics(realized, forecast) if len(group)
                   else {"test_mae": None, "test_rmse": None}),
                "mean_actual": float(realized.mean()) if len(group) else None,
                "mean_predicted": float(forecast.mean()) if len(group) else None,
                "mean_bias": float(np.mean(forecast - realized)) if len(group) else None,
                "underprediction_pct": 100 * float(np.mean(forecast < realized)) if len(group) else None,
            })
    return pd.DataFrame(rows)


def daily_risk_ranking(predictions):
    rows = []
    for day, group in predictions.groupby("date", sort=True):
        ranks = group[[ACTUAL, PREDICTED, HISTORICAL]].rank(method="average")
        row = {"date": day, "stocks": len(group)}
        for field, name in ((PREDICTED, "model"), (HISTORICAL, "baseline")):
            score = None
            if len(group) < MIN_RANKING_STOCKS:
                status = "insufficient_stocks"
            elif ranks[ACTUAL].nunique() < 2 or ranks[field].nunique() < 2:
                status = "constant_rank"
            else:
                status = "ok"
                score = float(ranks[ACTUAL].corr(ranks[field]))
            row[f"{name}_spearman"] = score
            row[f"{name}_status"] = status
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(info, model, test, calendar):
    if info["model"] == "Ridge":
        if not isinstance(model, Pipeline):
            raise ValueError("保存的 Ridge 模型不是 Pipeline")
        if not isinstance(model.named_steps.get("ridge"), Ridge):
            raise ValueError("Pipeline 中的迴歸器不是 Ridge")
        if not isinstance(model.named_steps.get("standardscaler"), StandardScaler):
            raise ValueError("Ridge Pipeline 缺少 StandardScaler")
    names = tuple(info["features"])
    if (names != training.FEATURE_SETS[info["feature_set"]]
            or names != tuple(model.feature_names_in_)
            or tuple(field for field in test.columns if field in names) != names
            or info["target"] != training.TARGET or info["prediction_floor"] != 0):
        raise ValueError("Test 欄位、模型特徵順序或預測規則與 05 設定不符")
    if info["model"] == "Ridge":
        fitted_rows = model.named_steps["standardscaler"].n_samples_seen_
        if fitted_rows != info["refit_rows"] or model.named_steps["ridge"].alpha != info["parameters"]["alpha"]:
            raise ValueError("Ridge 的重訓筆數或 alpha 與 05 設定不符")

    next_session = next_execution_dates(test, calendar)
    actual = test[training.TARGET].to_numpy()
    prediction = training.predict_volatility(model, test, names)
    historical = test["Volatility_5"].to_numpy()
    train_mean = float(info["train_mean"])
    if (historical < 0).any() or not math.isfinite(train_mean) or train_mean < 0:
        raise ValueError("基準波動率無效")
    mean_predictions = np.full(len(test), train_mean)
    historical_mae = error_metrics(actual, historical)["test_mae"]
    comparison = []
    for name, feature_set, values in (
        (info["model"], info["feature_set"], prediction),
        ("Historical Volatility Persistence", "baseline", historical),
        ("Train Mean", "baseline", mean_predictions),
    ):
        score = error_metrics(actual, values)
        comparison.append({
            "model": name, "feature_set": feature_set, "test_samples": len(test),
            **score, "mae_improvement": improvement(historical_mae, score["test_mae"]),
        })

    predictions = test[["date", "code", "session_index", "label_start_date", "label_end_date"]].assign(
        actual_5d_volatility=actual, predicted_5d_volatility=prediction,
        historical_5d_volatility=historical, train_mean_5d_volatility=mean_predictions,
        model=info["model"], feature_set=info["feature_set"],
        prediction_available="after_close", earliest_execution_date=next_session,
    )
    return predictions, pd.DataFrame(comparison), stock_metrics(predictions)


def main():
    info = json.loads((config.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    test, calendar = load_test()
    check_provenance(info, test["date"].min())
    model = joblib.load(config.MODELS_DIR / "final_model.joblib")
    predictions, comparison, by_stock = evaluate(info, model, test, calendar)
    regimes = volatility_regimes(predictions, info["model"])
    ranking = daily_risk_ranking(predictions)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("test_predictions", predictions), ("test_model_comparison", comparison),
        ("test_metrics_by_stock", by_stock), ("test_volatility_regimes", regimes),
        ("test_risk_ranking", ranking),
    ):
        frame.to_csv(config.TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"Test {len(test):,} 筆，{test['date'].min()} ~ {test['date'].max()}，"
          f"最晚 Label 結束日 {test['label_end_date'].max()}")
    print(comparison.to_string(index=False))
    paired = ranking.dropna(subset=["model_spearman", "baseline_spearman"])
    if not paired.empty:
        print(f"每日 Spearman 共同可比較 {len(paired)} 日，"
              f"模型 {paired['model_spearman'].mean():.6f}，"
              f"歷史基準 {paired['baseline_spearman'].mean():.6f}")


if __name__ == "__main__":
    main()
