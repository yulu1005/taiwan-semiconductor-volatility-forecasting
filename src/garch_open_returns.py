"""Opening returns adjusted only for actions within each return interval."""

import numpy as np
import pandas as pd


def opening_returns(panel, actions):
    """Return log returns in percent, retaining the previous valid open per stock."""
    valid = panel.loc[panel.valid & panel.open.notna()].sort_values(["sid", "date"])
    actions_by_stock = dict(iter(actions.groupby("sid", sort=False)))
    result = []
    for sid, bars in valid.groupby("sid", sort=False):
        dates = pd.DatetimeIndex(bars.date)
        opens = bars.open.to_numpy(dtype=float)
        adjustment = np.ones(len(bars))
        events = actions_by_stock.get(sid)
        if events is not None:
            # Include events after the previous valid open through this open.
            indices = dates.searchsorted(events.date, side="left")
            inside = (indices > 0) & (indices < len(dates))
            np.multiply.at(adjustment, indices[inside], events.factor.to_numpy()[inside])
        returns = 100 * np.log(opens[1:] / (opens[:-1] * adjustment[1:]))
        result.append(pd.DataFrame({"sid": sid, "date": dates[1:], "r": returns}))
    if not result:
        return pd.DataFrame(columns=["sid", "date", "r"])
    return pd.concat(result, ignore_index=True)


def read_actions(root):
    path = root / "outputs" / "tables" / "corporate_actions.csv"
    return pd.read_csv(path, dtype={"sid": str}, parse_dates=["date"], float_precision="round_trip")
