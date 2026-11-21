"""Simulate replenishment policies to trade service level against inventory.

Each FC reviews stock weekly and orders up to a base-stock level
S = (lead time + 1) x mean weekly demand + z x sd of demand over that period,
with a two-week replenishment lead time. Demand is drawn from the forecast
distribution with extra week-to-week noise (negative binomial). Raising the
safety factor z buys fill rate with inventory; the simulation finds the z that
minimizes holding cost plus lost margin, the point a planner would choose.
"""

import json
import os

import numpy as np
import pandas as pd

LEAD = 2
WEEKS = 52
REPS = 200
Z_GRID = np.round(np.arange(0.0, 3.01, 0.25), 2)


def simulate(mu, z, hold, margin, rng, dispersion=8.0):
    """mu: mean weekly demand per item (vector). Returns fill rate, avg on hand, cost per week."""
    n = len(mu)
    var = mu + mu ** 2 / dispersion                    # negative binomial variance
    S = np.ceil((LEAD + 1) * mu + z * np.sqrt((LEAD + 1) * var))
    on_hand = S.copy()
    pipeline = np.zeros((LEAD, n))
    p = dispersion / (dispersion + mu)
    tot_d = tot_served = tot_onhand = 0.0
    lost_margin = 0.0
    for t in range(WEEKS):
        on_hand += pipeline[0]
        pipeline = np.vstack([pipeline[1:], np.zeros(n)])
        d = rng.negative_binomial(dispersion, p)
        served = np.minimum(d, on_hand)
        on_hand -= served
        tot_d += d.sum()
        tot_served += served.sum()
        lost_margin += ((d - served) * margin).sum()
        tot_onhand += on_hand.sum()
        position = on_hand + pipeline.sum(axis=0)
        pipeline[-1] += np.maximum(S - position, 0)
    avg_onhand = tot_onhand / WEEKS
    return tot_served / tot_d, avg_onhand, (hold * tot_onhand + lost_margin) / WEEKS


def main(data="data", out="results"):
    net = json.load(open(os.path.join(data, "network.json")))
    fc = pd.read_csv(os.path.join(data, "forecast_next_week.csv"))
    styles = pd.read_csv(os.path.join(data, "styles.csv")).set_index("style_id")
    cost = np.array(net["ship_cost"])
    pref = dict(zip(net["regions"], np.array(net["fcs"])[cost.argmin(axis=1)]))
    fc["fc"] = fc.region.map(pref)
    by_fc = fc.groupby(["fc", "style_id"]).forecast.sum().reset_index()
    by_fc["margin"] = by_fc.style_id.map(styles.unit_margin)
    hold = net["holding_cost_per_unit_week"]
    rng = np.random.default_rng(3)
    rows = []
    for z in Z_GRID:
        res = np.array([simulate(by_fc.forecast.values, z, hold, by_fc.margin.values, rng) for _ in range(REPS)])
        rows.append({"safety_factor_z": z, "fill_rate": round(res[:, 0].mean(), 4),
                     "avg_units_on_hand": round(res[:, 1].mean(), 0),
                     "weekly_cost_holding_plus_lost_margin": round(res[:, 2].mean(), 0)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tables", "reorder_policy_tradeoff.csv"), index=False)
    best = df.loc[df.weekly_cost_holding_plus_lost_margin.idxmin()]
    print(df.to_string(index=False))
    print("lowest-cost z:", best.safety_factor_z)


if __name__ == "__main__":
    main()
