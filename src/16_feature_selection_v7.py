"""Reconstruct the V7 feature screen using 2016–2020 only."""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from common import CFG, PROC, VTAB, nw_mean_ci
from features_30 import F30
from models import load


def hac(values):
    mean, _, _, se = nw_mean_ci(values.dropna().to_numpy())
    if not np.isfinite(se) or se <= 0:
        return mean, np.nan, np.nan
    t = mean / se
    return mean, t, 2 * norm.sf(abs(t))


def residual(rank, control):
    centered = rank - rank.mean()
    denominator = control @ control
    return centered - control * (centered @ control / denominator) if denominator else centered


def screen(train, original, configured, out):
    correlation = train[original + F30].corr(method="spearman")
    candidates, duplicates = list(original), []
    for feature in F30:
        matches = correlation.loc[feature, candidates]
        match = matches[matches > 0.999]
        if match.empty:
            candidates.append(feature)
        else:
            duplicates.append({"feature": feature, "duplicate_of": match.index[0], "rho": match.iloc[0]})

    weeks = list(train.groupby("date"))
    rows = []
    for feature in candidates:
        columns = list(dict.fromkeys([feature, "y", "rv20"]))
        record = {"feature": feature, "source": "17" if feature in original else "30"}
        if train.groupby("date")[feature].nunique().max() <= 1:
            weekly = train.groupby("date")[columns].mean().dropna()
            design = sm.add_constant(weekly[[feature, "rv20"]])
            fit = sm.OLS(weekly.y, design).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            rows.append(dict(record, level="market", inc=fit.params[feature],
                             inc_t=fit.tvalues[feature], inc_p=fit.pvalues[feature]))
            continue
        ics, partial = {}, {}
        for day, group in weeks:
            ranks = group[columns].dropna().rank()
            if len(ranks) < 5 or ranks[feature].nunique() < 2 or ranks.y.nunique() < 2:
                continue
            ics[day] = ranks[feature].corr(ranks.y)
            if feature != "rv20":
                control = ranks.rv20 - ranks.rv20.mean()
                a, b = residual(ranks[feature], control), residual(ranks.y, control)
                if a.std() > 1e-12 and b.std() > 1e-12:
                    partial[day] = a.corr(b)
        ic = pd.Series(ics, dtype=float)
        mean, _, p = hac(ic)
        inc, t, inc_p = hac(pd.Series(partial, dtype=float)) if partial else (np.nan,) * 3
        yearly = ic.groupby(ic.index.year).mean().reindex(range(2017, 2021))
        rows.append(dict(record, level="stock", weeks=len(ic), ic=mean, p=p, inc=inc, inc_t=t, inc_p=inc_p,
                         direction_ok=bool((np.sign(yearly) == np.sign(mean)).all()),
                         **{f"ic_{year}": v for year, v in yearly.items()}))

    table = pd.DataFrame(rows).set_index("feature")
    stock = table.level.eq("stock")
    table.loc[stock, "bh_q"] = multipletests(table.loc[stock, "p"].fillna(1), method="fdr_bh")[1]
    base = (stock & table.bh_q.le(.05) & table.direction_ok.eq(True)) | (~stock & table.inc_p.lt(.05))
    table["passes_screen"] = base
    table["incremental_significant"] = table.inc_p.lt(.05)
    table["configured_v7"] = table.index.isin(configured)
    order = table.inc_t.abs().sort_values(ascending=False, kind="stable").index
    selected = ["rv20"]
    table["decision"] = np.where(base, "eligible", "failed_screen")
    table.loc["rv20", "decision"] = "tier_1_anchor"
    for tier, eligible, limit in ((1, base & table.incremental_significant, None),
                                  (2, base & ~table.incremental_significant, 8)):
        added = 0
        for feature in order[eligible.reindex(order)]:
            if feature in selected:
                continue
            matches = correlation.loc[feature, selected].abs()
            if matches.max() > .9:
                table.loc[feature, "decision"] = "redundant_with_" + matches.idxmax()
            elif limit is None or added < limit:
                selected.append(feature)
                added += 1
                table.loc[feature, "decision"] = f"tier_{tier}"

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "candidates.csv", encoding="utf-8-sig")
    pd.DataFrame(duplicates).to_csv(out / "deduplication.csv", index=False, encoding="utf-8-sig")
    table.loc[selected].to_csv(out / "selected.csv", encoding="utf-8-sig")
    result = dict(input_count=len(original) + len(F30), deduplicated_count=len(candidates),
                  bh_family_size=int(stock.sum()), tier_1_count=sum(table.loc[selected, "decision"].str.startswith("tier_1")),
                  tier_2_count=sum(table.loc[selected, "decision"].eq("tier_2")),
                  configured_only=sorted(set(configured) - set(selected)),
                  reproduced_only=sorted(set(selected) - set(configured)))
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out.name, result)


if __name__ == "__main__":
    frame = load()
    market_cols = ["date", "sid"] + [c for c in pd.read_parquet(PROC / "features_v2.parquet").columns
                                       if c not in frame.columns]
    frame = frame.merge(pd.read_parquet(PROC / "features_v2.parquet", columns=market_cols),
                        on=["date", "sid"], how="left")
    train = frame[frame.is_decision_wk & frame.y.notna() &
                  frame.date.ge(CFG["data"]["study_start"]) & frame.label_end.lt("2021-01-01")]
    original = CFG["features"]["v3"]
    corrected = ["garch_oo5" if name == "garch_fc5" else name for name in original]
    screen(train, original, CFG["features"]["v4"], VTAB / "feature_selection" / "legacy")
    screen(train, corrected, CFG["features"]["v7"], VTAB / "feature_selection" / "v7")
