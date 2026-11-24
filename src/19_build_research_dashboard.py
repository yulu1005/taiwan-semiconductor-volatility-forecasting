"""Build the four-section, offline V7 research dashboard."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from common import CFG
from models import load, make_model, split
from research_allocation_data import historical_data

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "outputs/v7/dashboard"
TABLES = OUT.parent / "tables"
M2 = TABLES / "model2"
PROC = ROOT / "data/processed"


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def research_data():
    source_codes, overview, target = historical_data()
    data = dict(overview=overview, target=target, codes=CFG["universe"]["stocks"])
    names = pd.read_csv(ROOT / "data/raw/listing_dates.csv", dtype={"sid": str})
    data["names"] = names.set_index("sid")["name"].to_dict()
    model = json.loads((M2 / "frozen_rules.json").read_text())["primary_model"]
    for week in data["overview"]:
        for strategy in ("A", "B"):
            for field, missing in (("weights", 0), ("inner", 0), ("vols", None)):
                row = week[strategy]
                values = dict(zip(source_codes, row[field]))
                row[field] = [values.get(sid, missing) for sid in data["codes"]] + (
                    [row[field][-1]] if field == "weights" else [])
            assert abs(sum(week[strategy]["weights"]) - 1) < 1e-9
    def metrics(filename):
        tab = pd.read_csv(TABLES / filename).set_index("config").loc[[model, "hv"]]
        return records(tab.assign(model=["Lasso", "HV20"]))
    data["m1"] = {"val": metrics("model1_table_2_8.csv"), "post": metrics("model1_posthoc_2025_2026.csv")}
    pred = pd.read_parquet(PROC / "oos_predictions_v7.parquet")
    data["byStock"] = {}
    for key, test in (("val", pred.fold.str.match(r"fold[123]$")), ("post", pred.date.ge("2025-01-01"))):
        wide = pred[test & pred.config.isin([model, "hv"])].pivot(
            index=["date", "sid", "y"], columns="config", values="p").dropna().reset_index()
        output = []
        for sid, rows in wide.groupby("sid"):
            for name, col in (("Lasso", model), ("HV20", "hv")):
                y, yhat = rows.y, rows[col]
                ratio = np.square(y / yhat)
                output.append(dict(sid=sid, model=name, n=len(rows), mae=(y-yhat).abs().mean(),
                                   rmse=np.sqrt(np.square(y-yhat).mean()), qlike=(ratio-np.log(ratio)-1).mean(),
                                   spearman=y.corr(yhat, method="spearman")))
        data["byStock"][key] = records(pd.DataFrame(output))
    features = CFG["features"]["v7"]
    train, _ = split(load(), "2024-12-31", "2025-01-01", "2025-01-01")
    train = train[train.garch_oo5.notna()]
    reg = make_model("lasso", {"alpha_frac": float(model.split("frac")[-1])}).fit(train[features], train.y)
    data["coefficients"] = dict(zip(features, reg.m.coef_.tolist()))
    data["featureScreen"] = records(pd.read_csv(TABLES / "feature_selection/v7/candidates.csv")[
        ["feature", "decision", "bh_q", "inc_p"]])
    data["baseline"] = records(pd.read_csv(TABLES / "model1_vs_baselines.csv"))
    data["crash"] = records(pd.read_csv(M2 / "window_crash_weeks.csv"))
    data["postWindows"] = records(pd.read_csv(M2 / "window_posthoc.csv"))
    bt = pd.read_csv(M2 / "backtest_A_to_D.csv")
    nav = pd.read_csv(M2 / "nav_curves.csv", index_col=0, parse_dates=True)
    data["m2"] = {}
    for key, period, year in (("val", "2022–2024 驗證期", "2022"), ("post", "2025–2026 事後評估", "2025")):
        rows, risk, series_nav = [], {}, {}
        for strategy in "ABCD":
            row = bt[(bt.期間 == period) & (bt.策略 == strategy)].iloc[0].to_dict()
            wealth = nav[f"{period}|{strategy}"].dropna()
            series_nav[strategy] = records(wealth.rename("nav").reset_index())
            weekly = pd.read_csv(M2 / f"weekly_{strategy}_{year}.csv", parse_dates=["exec"])
            row.update(final=wealth.iloc[-1], cumulative=wealth.iloc[-1]/300_000_000-1,
                       total_cost=weekly.cost.sum())
            rows.append(row)
            series = np.log(wealth / wealth.shift()).pow(2).rolling(5).mean().mul(252).pow(.5)
            values = weekly.assign(target=data["target"], after=weekly.a * weekly.sigma_p,
                                   realized=series.reindex(weekly.exec).to_numpy())
            risk[strategy] = records(values[["date", "a", "target", "sigma_p", "after", "realized"]])
        data["m2"][key] = dict(rows=records(pd.DataFrame(rows)), risk=risk, nav=series_nav)
    return data


def build():
    html = (OUT / "dashboard.html").read_text(encoding="utf-8")
    layout, script = html.split("<script>", 1)
    headings = ('<section class="kpis" id="kpis"', '<div class="section-title"><span class="tag">MODEL 1',
                '<div class="section-title"><span class="tag">MODEL 2',
                '<div class="section-title"><span class="tag">驗證')
    first, model1, model2, validation = [layout.index(marker) for marker in headings]
    footer = layout.index("<footer>", validation)
    sections = {
        "model1": layout[model1:model2],
        "model2": layout[first:model1] + layout[model2:validation],
        "validation": layout[validation:footer],
    }
    sections["model2"] = re.sub(r'<div class="legend"><span><i class="sw"[^\n]+</div>\n(?=    <div class="chart" id="ch-nav")',
                                "", sections["model2"], count=1)
    addition = Path(__file__).with_name("research_dashboard_preview_template.html").read_text(encoding="utf-8")
    css, body = addition.split("<!-- PANEL -->", 1)
    body, js = body.split("<!-- SCRIPT -->", 1)
    for key, content in sections.items():
        body = body.replace(f"<!-- {key.upper()} -->", content)
    js = js.replace("/*__RESEARCH__*/null", json.dumps(research_data(), ensure_ascii=False, allow_nan=False).replace("</", r"<\/"))
    header = layout[:first].replace("<head>", "<head>\n<title>V7 研究儀表板</title>\n" + css)
    page = header + body + layout[footer:] + "<script>" + script.replace("</body>", js + "\n</body>")
    destination = OUT / "research_dashboard_preview.html"
    destination.write_text(page, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    build()
