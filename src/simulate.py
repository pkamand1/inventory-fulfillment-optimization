"""Simulated network and demand for a fictional apparel retailer.

Four fulfillment centers (FCs) serve twelve client regions. Weekly demand for
60 styles in each region follows a Poisson model whose true coefficients are
known: a style-level base rate, a regional multiplier, annual seasonality that
differs by category, a promotion lift, and a price elasticity. Forecasting
models can therefore be checked against the truth, and policies can be
evaluated on fresh demand drawn from the same process.

Writes data/history.csv (34 weeks), data/network.json, data/truth.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 7
N_STYLES = 60
N_WEEKS = 34
FCS = ["FC-East", "FC-Central", "FC-South", "FC-West"]
REGIONS = ["Northeast", "Mid-Atlantic", "Southeast", "Florida", "Great Lakes", "Plains",
           "Texas", "Mountain", "Southwest", "California", "Pacific NW", "Alaska-Hawaii"]
# Approximate location of each region and FC on a stylized map (for shipping cost).
REGION_XY = np.array([[9, 8], [8.5, 6.5], [7.5, 4], [8, 1.5], [6.5, 7], [4.5, 6], [4.5, 2.5],
                      [2.5, 6], [2, 3], [0.5, 4], [0.5, 8], [0, 0]])
FC_XY = np.array([[8.5, 7], [5.5, 6], [6, 3], [1, 5]])
REGION_SIZE = np.array([1.2, 1.1, 1.0, 0.9, 1.0, 0.6, 1.1, 0.6, 0.8, 1.4, 0.7, 0.2])
CATEGORIES = ["outerwear", "dresses", "denim", "knitwear", "activewear"]
TRUE = {
    "promo_lift_log": 0.35,          # promotions raise demand by exp(0.35) = 1.42
    "price_elasticity": -1.2,        # log demand per log price relative to list
    "season_amp": {"outerwear": 0.55, "dresses": -0.45, "denim": 0.05, "knitwear": 0.40, "activewear": -0.15},
}


def demand_mean(style_base, region_mult, cat_amp, week, promo, price_ratio):
    season = cat_amp * np.cos(2 * np.pi * (week - 2) / 52)   # peaks in early January for positive amplitude
    return np.exp(np.log(style_base) + np.log(region_mult) + season
                  + TRUE["promo_lift_log"] * promo + TRUE["price_elasticity"] * np.log(price_ratio))


def main(out="data"):
    rng = np.random.default_rng(SEED)
    os.makedirs(out, exist_ok=True)
    styles = pd.DataFrame({
        "style_id": [f"ST{i:03d}" for i in range(1, N_STYLES + 1)],
        "category": rng.choice(CATEGORIES, N_STYLES),
        "base_rate": rng.lognormal(np.log(3.0), 0.5, N_STYLES).round(3),
        "unit_margin": rng.uniform(18, 45, N_STYLES).round(2),
    })
    rows = []
    for w in range(N_WEEKS):
        week_of_year = (36 + w) % 52          # history starts in early September
        for _, s in styles.iterrows():
            promo_week = rng.random() < 0.12
            price_ratio = 0.8 if promo_week else (1.0 if rng.random() > 0.1 else 0.9)
            for r, reg in enumerate(REGIONS):
                mu = demand_mean(s.base_rate, REGION_SIZE[r], TRUE["season_amp"][s.category],
                                 week_of_year, int(promo_week), price_ratio)
                rows.append({"week": w, "week_of_year": week_of_year, "style_id": s.style_id,
                             "category": s.category, "region": reg, "promo": int(promo_week),
                             "price_ratio": price_ratio, "units": int(rng.poisson(mu)), "true_mean": round(mu, 4)})
    hist = pd.DataFrame(rows)
    hist.to_csv(os.path.join(out, "history.csv"), index=False)
    styles.to_csv(os.path.join(out, "styles.csv"), index=False)

    dist = np.linalg.norm(REGION_XY[:, None, :] - FC_XY[None, :, :], axis=2)
    ship_cost = (4.0 + 0.9 * dist).round(2)          # dollars per unit, region x FC
    ship_cost[11] += 6.0                              # Alaska-Hawaii premium
    network = {
        "fcs": FCS, "regions": REGIONS,
        "ship_cost": ship_cost.tolist(),              # [region][fc]
        "fc_capacity_units": [560, 480, 420, 560],    # units each FC can hold for the planning week
        "split_handling_cost": 2.5,                   # extra cost when an order ships from a non-preferred FC
        "holding_cost_per_unit_week": 0.35,
    }
    json.dump(network, open(os.path.join(out, "network.json"), "w"), indent=2)
    truth = {"seed": SEED, **TRUE, "region_size": dict(zip(REGIONS, REGION_SIZE.tolist()))}
    json.dump(truth, open(os.path.join(out, "truth.json"), "w"), indent=2)
    print(f"{len(hist):,} rows of weekly demand history; total units {hist.units.sum():,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
