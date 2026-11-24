"""Freeze V7, register as-of daily observations, and report paper trading."""

import argparse
from datetime import datetime, time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import CFG, PROC, ROOT, VTAB
from model2_execution import execute_plan
from model2_v7 import allocate, window_correlation
from models import FLOOR, load, make_model, split

TAIPEI = timezone(timedelta(hours=8))
STRATEGIES = {"A": ("rv20", False), "B": ("ml", False),
              "C": ("rv20", True), "D": ("ml", True)}
COST = {"slip": 0.0005, "comm": 0.001425, "tax": 0.003}
PIPELINE = ("01_download.py", "02_build_dataset.py", "03_features_labels.py", "03b_features_v2.py",
            "features_30.py", "universe.py", "03d_garch_oo.py", "garch_open_returns.py",
            "garch.py", "features.py", "features_v2.py", "common.py", "models.py", "model2_v7.py",
            "model2_execution.py", "17_paper_v7.py")
REFRESH_STEPS = ("02_build_dataset.py", "03_features_labels.py", "03b_features_v2.py",
                 "features_30.py", "universe.py", "03d_garch_oo.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)


def fingerprint():
    paths = [ROOT / "setup.yaml", *(ROOT / "src" / name for name in PIPELINE)]
    return {str(path.relative_to(ROOT).as_posix()): sha256(path.read_bytes()).hexdigest() for path in paths}


def freeze(journal, now):
    path = journal / "freeze.json"
    if path.exists():
        raise SystemExit("The journal is already frozen.")
    rules = read_json(VTAB / "model2" / "frozen_rules.json")
    config = rules["primary_model"]
    if not config.startswith("lasso_alpha_frac"):
        raise SystemExit(f"Unsupported frozen Model 1: {config}")
    features = CFG["features"]["v7"]
    train, _ = split(load(), "2024-12-31", "2025-01-01", "2025-01-01")
    train = train[train.garch_oo5.notna()]
    model = make_model("lasso", {"alpha_frac": float(config.split("frac")[1])}).fit(
        train[features], train.y
    )
    fit = {key: getattr(model, key).tolist() for key in ("med", "lo", "hi", "mu", "sd")}
    fit.update(coefficients=model.m.coef_.tolist(), intercept=float(model.m.intercept_))
    journal.mkdir(parents=True, exist_ok=True)
    write_new(path, dict(frozen_at=now.isoformat(), train_end="2024-12-31", training_rows=len(train),
                         features=features, fit=fit, rules=rules, code=fingerprint(), capital=300_000_000,
                         costs=COST, rebalance="next trading session open"))
    print(f"Frozen protocol: {path}")


def targets(day, panel, frozen):
    frame = load()
    stock = frame[frame.date.eq(day) & frame.tradable.fillna(False) & frame.garch_oo5.notna() &
                  frame.rv20.notna()].sort_values("sid").copy()
    if stock.empty or not stock.is_decision_wk.all():
        raise SystemExit("No eligible weekly decision rows for this session.")
    features, fit, rules = frozen["features"], frozen["fit"], frozen["rules"]
    x = stock[features].to_numpy()
    x = np.where(np.isnan(x), fit["med"], x)
    standardized = (np.clip(x, fit["lo"], fit["hi"]) - fit["mu"]) / fit["sd"]
    stock["ml"] = np.maximum(standardized @ fit["coefficients"] + fit["intercept"], FLOOR)
    sids = stock.sid.tolist()
    valid = panel[panel.valid]
    cal = valid.loc[valid.sid.eq("0050"), "date"].sort_values()
    opens = valid.pivot(index="date", columns="sid", values="aopen").reindex(cal)
    daily = np.log(opens / opens.shift()).loc[:, sids].tail(rules["window"])
    R, lam, _, _, fallback = window_correlation(daily, rules["window"])
    return {
        strategy: dict(sids=sids, lam=lam, est_fail=fallback, **allocate(
            stock[column].to_numpy(), R, rules["alpha"] if dynamic else 0, rules["sigma_target"]
        ))
        for strategy, (column, dynamic) in STRATEGIES.items()
    }


def serializable(plan):
    return {name: {k: (v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v)
                   for k, v in allocation.items()} for name, allocation in plan.items()}


def capture(journal, frozen, now, cash_rate, decision, short_week):
    day = pd.Timestamp(now.date())
    if day.date() <= datetime.fromisoformat(frozen["frozen_at"]).date() or now.time() < time(13, 30):
        raise SystemExit("Capture after the close on a session strictly after the freeze date.")
    if decision and day.dayofweek != 4 and not short_week:
        raise SystemExit("For a holiday-shortened week, confirm the last session with --short-week.")
    panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])
    calendar = pd.DatetimeIndex(panel.loc[panel.sid.eq("0050") & panel.valid, "date"])
    if calendar.empty or calendar[-1] != day or panel.date.max() != day:
        raise SystemExit("Processed data must end at today's completed market session.")
    manifest = read_json(ROOT / "data" / "raw" / "manifest.json")
    retrieved = datetime.fromisoformat(manifest["downloaded_at"]).astimezone(TAIPEI)
    if retrieved.date() != day.date() or not time(13, 30) <= retrieved.time() or retrieved > now:
        raise SystemExit("Today's market snapshot must have been downloaded after the close.")
    daily = journal / "sessions"
    daily.mkdir(exist_ok=True)
    previous = sorted(daily.glob("*.json"))
    if previous and (calendar[(calendar > pd.Timestamp(previous[-1].stem)) & (calendar <= day)].tolist() != [day]):
        raise SystemExit("Missing or duplicate session in the prospective journal.")
    if decision and any(pd.Timestamp(path.stem).to_period("W-SUN") == day.to_period("W-SUN") and
                        read_json(path)["targets"] for path in previous[-5:]):
        raise SystemExit("This week already has a registered decision.")
    # Normalize each price series to its first observation. Later corporate
    # actions then cancel from the scale of already-captured observations.
    anchors = panel.groupby("sid").adj.first()
    today = panel[panel.date.eq(day) & panel.valid].set_index("sid")
    quotes = today[["aopen", "aclose"]].div(anchors, axis=0).dropna()
    orders = serializable(targets(day, panel, frozen)) if decision else {}
    adv = (panel.assign(amount=panel.amount.where(panel.valid)).groupby("sid").amount
           .transform(lambda values: values.rolling(20, min_periods=15).mean()))
    adv_today = panel.assign(adv=adv).loc[panel.date.eq(day)].set_index("sid").adv.dropna().to_dict()
    file = daily / f"{day.date()}.json"
    write_new(file, dict(captured_at=now.isoformat(), date=str(day.date()), cash_rate=cash_rate,
                         quotes=quotes.to_dict("index"), targets=orders, adv=adv_today,
                         data_snapshot=manifest["downloaded_at"]))
    print(f"Recorded session: {file}")


def report(journal, frozen):
    observations = [read_json(path) for path in sorted((journal / "sessions").glob("*.json"))]
    if len(observations) < 2:
        print("No independently observed holding period yet.")
        return
    cal = pd.DatetimeIndex([day["date"] for day in observations])
    opens = pd.DataFrame([{s: v["aopen"] for s, v in day["quotes"].items()} for day in observations], index=cal)
    closes = pd.DataFrame([{s: v["aclose"] for s, v in day["quotes"].items()} for day in observations], index=cal)
    rates = dict(zip(cal, [day["cash_rate"] for day in observations]))
    adv = {(date, sid): value for date, day in zip(cal, observations) for sid, value in day["adv"].items()}
    results = []
    for strategy in STRATEGIES:
        plan = [dict(day["targets"][strategy], date=date) for date, day in zip(cal, observations) if day["targets"]]
        for order in plan:
            order["rc"], order["b"] = np.array(order["rc"]), np.array(order["b"])
        executed = execute_plan(plan, cal, opens, closes, adv, cal[0], cal[-1], capital=frozen["capital"],
                                deposit_rate=rates.__getitem__, **frozen["costs"])
        if executed is None:
            continue
        nav, trades, weekly = executed
        nav.to_csv(journal / f"nav_{strategy}.csv")
        trades.to_csv(journal / f"trades_{strategy}.csv", index=False)
        weekly.to_csv(journal / f"weekly_{strategy}.csv", index=False)
        daily_returns = nav.nav.pct_change().dropna()
        gaps = nav.index.to_series().diff().dt.days.reindex(daily_returns.index)
        excess = daily_returns - gaps * pd.Series(rates).reindex(gaps.index) / 365
        sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if len(excess) >= 20 and excess.std() > 0 else np.nan
        results.append(dict(strategy=strategy, observed_sessions=len(nav), rebalances=len(weekly),
                            net_return=nav.nav.iloc[-1] / frozen["capital"] - 1,
                            max_drawdown=(nav.nav / nav.nav.cummax() - 1).min(), annualized_sharpe=sharpe))
    if results:
        pd.DataFrame(results).to_csv(journal / "performance.csv", index=False)
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("No registered decision has reached its next session open yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=ROOT / "paper_v7")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("refresh", help="Download today's data in this working copy and rebuild features.")
    mark = commands.add_parser("capture")
    mark.add_argument("--cash-rate", type=float, required=True, help="Annual Taiwan Bank rate as a decimal.")
    mark.add_argument("--decision", action="store_true")
    mark.add_argument("--short-week", action="store_true")
    commands.add_parser("report")
    args = parser.parse_args()
    now = datetime.now(TAIPEI)
    if args.command == "freeze":
        freeze(args.journal, now)
    else:
        frozen = read_json(args.journal / "freeze.json")
        if fingerprint() != frozen["code"]:
            raise SystemExit("Code/configuration differs from the frozen protocol.")
        if args.command == "refresh":
            if now.time() < time(13, 30):
                raise SystemExit("Refresh after the Taiwan market closes.")
            env = dict(os.environ, PYTHONUTF8="1")
            subprocess.run([sys.executable, "src/01_download.py", "--end", now.date().isoformat()],
                           cwd=ROOT, env=env, check=True)
            for step in REFRESH_STEPS:
                subprocess.run([sys.executable, f"src/{step}"], cwd=ROOT, env=env, check=True)
        elif args.command == "capture":
            if not 0 <= args.cash_rate < 1:
                parser.error("--cash-rate must be an annual decimal between 0 and 1")
            capture(args.journal, frozen, now, args.cash_rate, args.decision, args.short_week)
        else:
            report(args.journal, frozen)
