"""使用既有 Train、Validation 切分訓練跨股票波動率模型。"""

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
from itertools import product

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


FEATURES = import_module("03_financial_features").FEATURE_COLUMNS
FEATURE_SETS = {
    "A": FEATURES,
    "B": tuple(name for name in FEATURES if name != "Turnover"),
}
TARGET = "future_5d_volatility"
SEED = 42
TRAINING_THREADS = 2
EARLY_STOPPING_ROUNDS = 40
MAX_ESTIMATORS = 500
RIDGE_ALPHAS = (0.01, 0.1, 1, 10, 100)
LIGHTGBM_GRID = {
    "learning_rate": (0.03, 0.05),
    "num_leaves": (15, 31),
    "max_depth": (3, 5),
    "min_child_samples": (30, 60),
    "reg_lambda": (0, 1),
}
PERIODS = {"train": ("2016-11-01", "2021-12-31"), "validation": ("2022-01-01", "2023-12-31")}


def load_split(path, period):
    frame = pd.read_csv(path, dtype={"code": str, "date": str, "label_start_date": str, "label_end_date": str})
    required = {"date", "code", "label_start_date", "label_end_date", TARGET, *FEATURES}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{path.name} 缺少必要欄位或無資料")
    if frame.duplicated(["date", "code"]).any() or not frame["code"].isin(config.STOCK_CODES).all():
        raise ValueError(f"{path.name} 包含重複股票日期鍵或未知股票")
    dates = frame[["date", "label_start_date", "label_end_date"]].apply(
        lambda column: pd.to_datetime(column, format="%Y-%m-%d", errors="coerce")
    )
    start, end = map(pd.Timestamp, period)
    if (dates.isna().any().any()
            or not ((start <= dates["date"]) & (dates["date"] < dates["label_start_date"])
                    & (dates["label_start_date"] <= dates["label_end_date"])
                    & (dates["label_end_date"] <= end)).all()):
        raise ValueError(f"{path.name} 日期或 Label 超出切分區間")
    numeric = frame[[*FEATURES, TARGET]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[TARGET] < 0).any():
        raise ValueError(f"{path.name} 存在無效的特徵或目標值")
    frame[[*FEATURES, TARGET]] = numeric
    return frame


def predict_volatility(model, frame, feature_columns):
    """供 06 沿用欄位順序及非負預測規則。"""
    predictions = np.asarray(model.predict(frame[list(feature_columns)]), dtype=float)
    if predictions.ndim != 1 or len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise ValueError("模型預測值的筆數、維度或數值無效")
    return np.maximum(predictions, 0.0)


def metrics(actual, predicted):
    residual = np.asarray(actual, dtype=float) - predicted
    return {"validation_mae": float(np.mean(np.abs(residual))),
            "validation_rmse": float(np.sqrt(np.mean(np.square(residual))))}


def clipped_mae(actual, predicted):
    return "clipped_mae", metrics(actual, np.maximum(predicted, 0.0))["validation_mae"], False


def make_model(name, parameters):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**parameters))
    return lgb.LGBMRegressor(
        objective="regression", metric="None", random_state=SEED, n_jobs=TRAINING_THREADS,
        deterministic=True, force_col_wise=True, verbosity=-1, **parameters,
    )


def ranked(result):
    return result["validation_mae"], result["validation_rmse"], result["trial_id"]


def search_models(train, validation):
    trials, best_groups = [], []
    lightgbm_grid = [dict(zip(LIGHTGBM_GRID, values, strict=True), n_estimators=MAX_ESTIMATORS)
                     for values in product(*LIGHTGBM_GRID.values())]
    for feature_set, names in FEATURE_SETS.items():
        for model_name, grid in (("Ridge", [{"alpha": value} for value in RIDGE_ALPHAS]),
                                 ("LightGBM", lightgbm_grid)):
            best = None
            for number, parameters in enumerate(grid, start=1):
                model = make_model(model_name, parameters)
                fit_options = {}
                if model_name == "LightGBM":
                    fit_options = {
                        "eval_set": [(validation[list(names)], validation[TARGET])],
                        "eval_metric": clipped_mae,
                        "callbacks": [lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True,
                                                         verbose=False)],
                    }
                model.fit(train[list(names)], train[TARGET], **fit_options)
                prediction = predict_volatility(model, validation, names)
                measured = {
                    "trial_id": f"{model_name}_{feature_set}_{number:02d}",
                    "model": model_name, "feature_set": feature_set,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "best_iteration": model.best_iteration_ if model_name == "LightGBM" else None,
                    **metrics(validation[TARGET], prediction),
                }
                trials.append(measured)
                if best is None or ranked(measured) < ranked(best[0]):
                    best = measured, model, prediction
            best_groups.append(best)
            print(f"{model_name}/{feature_set} {len(grid)} 組，最低 Validation MAE "
                  f"{best[0]['validation_mae']:.6f}", flush=True)
    return trials, best_groups


def importance(model, feature_set, stage):
    booster = model.booster_
    return [
        {"model": "LightGBM", "feature_set": feature_set, "stage": stage, "feature": feature,
         "gain": float(gain), "split_count": int(splits)}
        for feature, gain, splits in zip(booster.feature_name(), booster.feature_importance("gain"),
                                         booster.feature_importance("split"), strict=True)
    ]


def write_table(name, rows):
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def main():
    if len(FEATURES) != 15 or len(FEATURE_SETS["B"]) != 14:
        raise ValueError("需要 15 項 Feature，B 組移除 Turnover")
    paths = {name: config.PROCESSED_DATA_DIR / f"{name}.csv" for name in PERIODS}
    train, validation = (load_split(paths[name], PERIODS[name]) for name in PERIODS)
    actual = validation[TARGET].to_numpy()
    historical = validation["Volatility_5"].to_numpy()
    if (historical < 0).any():
        raise ValueError("歷史波動率基準有負值")
    train_mean = float(train[TARGET].mean())
    baseline = metrics(actual, historical)["validation_mae"]
    reference = config.TABLES_DIR / "feature_evaluation.csv"
    reference_mae = None
    if reference.is_file():
        scores = pd.read_csv(reference)
        reference_mae = float(scores.loc[scores["method"] == "persistence_baseline", "validation_mae"].item())
        if not math.isclose(baseline, reference_mae, rel_tol=0.0, abs_tol=5e-7):
            raise ValueError("05 重算的歷史基準 MAE 與 04 報告不符")
    print(f"Train {len(train):,} 筆、Validation {len(validation):,} 筆，"
          f"歷史波動率基準 MAE {baseline:.6f}", flush=True)
    trials, groups = search_models(train, validation)
    selected = min((result for result, _, _ in groups), key=ranked)

    comparison, predictions, by_stock, importances = [], [], [], []
    evaluation_runs = [
        ({"model": "Historical Volatility Persistence", "feature_set": "baseline",
          "parameters": json.dumps({"feature": "Volatility_5"})}, None, historical),
        ({"model": "Train Mean", "feature_set": "baseline",
          "parameters": json.dumps({"train_mean": train_mean})}, None, np.full(len(validation), train_mean)),
        *groups,
    ]
    for result, estimator, predicted in evaluation_runs:
        name, feature_set = result["model"], result["feature_set"]
        score = metrics(actual, predicted)
        comparison.append({
            "model": name, "feature_set": feature_set, "parameters": result["parameters"],
            "trial_id": result.get("trial_id"), "best_iteration": result.get("best_iteration"), **score,
            "mae_improvement": 100 * (baseline - score["validation_mae"]) / baseline if baseline else None,
            "selected": int(result is selected),
        })
        predictions.append(validation[["date", "code"]].assign(
            actual_volatility=actual, predicted_volatility=predicted, model=name, feature_set=feature_set,
        ))
        for code, data in predictions[-1].groupby("code", sort=True):
            by_stock.append({
                "code": code, "model": name, "feature_set": feature_set, "validation_samples": len(data),
                **metrics(data["actual_volatility"], data["predicted_volatility"].to_numpy()),
            })
        if name == "LightGBM":
            importances.extend(importance(estimator, feature_set, "train_before_refit"))

    for trial in trials:
        trial["best_in_group"] = int(any(trial is result for result, _, _ in groups))
        trial["selected"] = int(trial is selected)

    names = FEATURE_SETS[selected["feature_set"]]
    parameters = json.loads(selected["parameters"])
    if selected["model"] == "LightGBM":
        parameters["n_estimators"] = selected["best_iteration"]
    final = make_model(selected["model"], parameters)
    refit = pd.concat([train, validation], ignore_index=True)
    final.fit(refit[list(names)], refit[TARGET])
    if selected["model"] == "LightGBM":
        importances.extend(importance(final, selected["feature_set"], "train_validation_refit"))

    model_info = {
        "model": selected["model"], "feature_set": selected["feature_set"], "features": list(names),
        "target": TARGET, "parameters": parameters,
        "selection_trial": selected,
        "selection_rule": "min Validation MAE, then RMSE, then trial ID, among Ridge and LightGBM",
        "validation_mae": selected["validation_mae"], "validation_rmse": selected["validation_rmse"],
        "refit_rows": len(refit), "refit_anchor_start": refit["date"].min(),
        "refit_anchor_end": refit["date"].max(), "refit_label_end": refit["label_end_date"].max(),
        "selection_splits": {
            name: {"period": PERIODS[name], "rows": len(frame), "file": paths[name].name,
                   "sha256": sha256(paths[name].read_bytes()).hexdigest()}
            for name, frame in (("train", train), ("validation", validation))
        },
        "historical_mae": baseline, "historical_mae_04": reference_mae, "train_mean": train_mean,
        "pooled": True, "prediction_floor": 0.0,
        "seed": SEED, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "configuration_count": len(trials),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {name: metadata.version(name) for name in
                             ("joblib", "lightgbm", "numpy", "pandas", "scikit-learn")},
    }
    sample = validation.iloc[:10]
    expected = predict_volatility(final, sample, names)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = config.MODELS_DIR / "final_model.joblib.tmp"
    try:
        joblib.dump(final, temporary)
        restored = predict_volatility(joblib.load(temporary), sample, names)
        if not np.allclose(restored, expected, rtol=0, atol=1e-12):
            raise ValueError("重載模型後預測與原模型不一致")
        temporary.replace(config.MODELS_DIR / "final_model.joblib")
    finally:
        temporary.unlink(missing_ok=True)
    (config.MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_table("model_comparison", comparison)
    write_table("hyperparameter_results", trials)
    write_table("validation_metrics_by_stock", by_stock)
    write_table("feature_importance", importances)
    pd.concat(predictions, ignore_index=True).to_csv(
        config.TABLES_DIR / "validation_predictions.csv", index=False, encoding="utf-8-sig",
    )
    print(f"選定 {selected['model']}/{selected['feature_set']}，以 {len(refit):,} 筆重訓並保存")


if __name__ == "__main__":
    main()
