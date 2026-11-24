"""Cash rates used by the V7 historical simulation."""

import pandas as pd


def deposit_rate(date):
    steps = [("1900-01-01", 0.00815), ("2022-03-21", 0.01065),
             ("2022-06-20", 0.01190), ("2022-09-26", 0.01315),
             ("2022-12-21", 0.01440), ("2023-03-27", 0.01565),
             ("2024-03-25", 0.01690)]
    rate = steps[0][1]
    for start, value in steps:
        if date >= pd.Timestamp(start):
            rate = value
    return rate
