"""Reconstruct historical A/B allocation data from the V7 run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from model2_v7 import allocate, window_correlation
from models import load


def historical_data():
    tables = ROOT / "outputs/v7/tables/model2"
    proc = ROOT / "data/processed"
    rules = json.loads((tables / "frozen_rules.json").read_text())
    forecasts = pd.read_parquet(proc / "model2_sigma_ml_v7.parquet")
    frame = load().merge(forecasts, on=["date", "sid"], validate="one_to_one")
    frame = frame[frame.tradable & frame.is_decision_wk & frame.rv20.notna() & frame.ml.notna()]
    panel = pd.read_parquet(proc / "panel.parquet", columns=["sid", "date", "valid", "aopen"])
    panel = panel[panel.valid]
    calendar = pd.DatetimeIndex(sorted(panel.loc[panel.sid.eq("0050"), "date"]))
    opens = panel.pivot(index="date", columns="sid", values="aopen").reindex(calendar)
    returns = np.log(opens / opens.shift())
    weekly = {key: pd.read_csv(tables / f"weekly_{key}_2025.csv", parse_dates=["date", "exec"]).set_index("date")
              for key in ("A", "B")}
    dates = weekly["A"].index.intersection(weekly["B"].index).sort_values()
    codes = sorted(frame.sid.unique().tolist())
    data = []
    max_difference = 0.0
    for date in dates:
        stocks = frame[frame.date.eq(date)].set_index("sid").sort_index()
        names = stocks.index.tolist()
        window = rules["window"]
        matrix, _, _, _, _ = window_correlation(returns.loc[:date, names].tail(window), window)
        row = {"date": date.date().isoformat()}
        for strategy, column in (("A", "rv20"), ("B", "ml")):
            record = weekly[strategy].loc[date]
            sigma = stocks[column]
            result = allocate(sigma.to_numpy(), matrix, 0.0, rules["sigma_target"])
            if len(stocks) != record.n:
                raise ValueError(f"{date.date()} {strategy}: {len(stocks)} stocks, expected {record.n}")
            difference = max(abs(result["sigma_p"] - record.sigma_p), abs(result["a"] - record.a))
            max_difference = max(max_difference, difference)
            inner = pd.Series(result["q"], index=names).reindex(codes, fill_value=0).tolist()
            weights = pd.Series(result["w"], index=names).reindex(codes, fill_value=0).tolist()
            exposure = min(1., sum(weights))
            weights.append(1. - exposure)
            assert abs(sum(weights) - 1.) < 1e-10
            row[strategy] = dict(nav=float(record.V), exec=record.exec.date().isoformat(),
                                 stockVol=result["sigma_p"], exposure=exposure, weights=weights,
                                 vols=[float(sigma[sid]) if sid in stocks.index else None for sid in codes], inner=inner)
        data.append(row)
    print(f"Matched {len(data)} weeks; max historical exposure/volatility difference {max_difference:.3g}")
    if max_difference > 1e-6:
        raise ValueError("Computed allocations differ from the historical run.")
    return codes, data, rules["sigma_target"]
