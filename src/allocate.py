"""Allocate next week's inventory across fulfillment centers, three ways, and
evaluate each allocation by simulation on demand it has not seen.

The decision: how many units of each style to place in each FC before the
week starts, given limited supply per style and limited FC capacity. During
the week each order is filled from the cheapest FC that still has the style;
if none has it, the sale is lost.

Policies
  1. Heuristic: split each style's supply by the share of its historical demand
     in regions nearest each FC, then scale down to fit capacity.
  2. Deterministic optimization: a linear program on the point forecast.
  3. Stochastic optimization: a two-stage linear program over 60 demand
     scenarios drawn from the forecast distribution (sample average
     approximation), which hedges against uncertainty in where demand lands.

Evaluation draws 1,000 weeks of demand from the true simulated process, so
the policies are judged on outcomes rather than on their own objective.
"""

import json
import os

import numpy as np
import pandas as pd
import pulp

N_SCENARIOS = 60
N_EVAL = 1000
SEED = 11


def load(data):
    net = json.load(open(os.path.join(data, "network.json")))
    fc = pd.read_csv(os.path.join(data, "forecast_next_week.csv"))
    styles = pd.read_csv(os.path.join(data, "styles.csv")).set_index("style_id")
    hist = pd.read_csv(os.path.join(data, "history.csv"))
    return net, fc, styles, hist


def solve_lp(mean_or_scenarios, supply, margin, net, label):
    """mean_or_scenarios: array [k, S, R] of demand scenarios (k = 1 for the point forecast)."""
    D = mean_or_scenarios
    K, S, R = D.shape
    F = len(net["fcs"])
    cost = np.array(net["ship_cost"])          # R x F
    pref = cost.argmin(axis=1)                 # preferred FC per region
    split = net["split_handling_cost"]
    hold = net["holding_cost_per_unit_week"]
    cap = net["fc_capacity_units"]
    prob = pulp.LpProblem(label, pulp.LpMaximize)
    x = {(s, f): pulp.LpVariable(f"x_{s}_{f}", lowBound=0) for s in range(S) for f in range(F)}
    y = {(k, s, f, r): pulp.LpVariable(f"y_{k}_{s}_{f}_{r}", lowBound=0)
         for k in range(K) for s in range(S) for f in range(F) for r in range(R)}
    unit_cost = lambda f, r: cost[r, f] + (split if f != pref[r] else 0)
    prob += (pulp.lpSum((margin[s] - unit_cost(f, r) + hold) * y[k, s, f, r] / K
                        for k in range(K) for s in range(S) for f in range(F) for r in range(R))
             - pulp.lpSum(hold * x[s, f] for s in range(S) for f in range(F)))
    for s in range(S):
        prob += pulp.lpSum(x[s, f] for f in range(F)) <= supply[s]
    for f in range(F):
        prob += pulp.lpSum(x[s, f] for s in range(S)) <= cap[f]
    for k in range(K):
        for s in range(S):
            for f in range(F):
                prob += pulp.lpSum(y[k, s, f, r] for r in range(R)) <= x[s, f]
            for r in range(R):
                prob += pulp.lpSum(y[k, s, f, r] for f in range(F)) <= D[k, s, r]
    prob.solve(pulp.HiGHS(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal", pulp.LpStatus[prob.status]
    X = np.array([[x[s, f].value() for f in range(F)] for s in range(S)])
    return np.floor(X + 1e-6)                  # whole units


def heuristic(hist, styles_idx, regions, supply, net):
    cost = np.array(net["ship_cost"])
    pref = cost.argmin(axis=1)
    F = len(net["fcs"])
    share = (hist[hist.week >= hist.week.max() - 7].groupby(["style_id", "region"]).units.sum()
             .unstack().reindex(index=styles_idx, columns=regions).fillna(0).values)
    to_fc = np.zeros((len(styles_idx), F))
    for r in range(len(regions)):
        to_fc[:, pref[r]] += share[:, r]
    to_fc = to_fc / to_fc.sum(axis=1, keepdims=True)
    cap = np.array(net["fc_capacity_units"], float)
    X = to_fc * supply[:, None]
    X = np.floor(X * np.minimum(1, cap / X.sum(axis=0)))
    # Spill each style's unplaced supply into FCs that still have room, in
    # proportion to their spare capacity, as a planner would.
    spare = cap - X.sum(axis=0)
    left = supply - X.sum(axis=1)
    if spare.sum() > 0:
        X += np.floor(np.minimum(left[:, None] * spare[None, :] / spare.sum(),
                                 left[:, None] * spare[None, :] / max(left.sum(), 1)))
    return X


def fulfill(X, D, cost, pref, split, margin, hold, rng):
    """Greedy order-by-order fulfillment: each order ships from the cheapest FC with stock."""
    stock = X.copy()
    S, R = D.shape
    order_cost = cost[None, :, :] + split * (np.arange(cost.shape[1])[None, None, :] != pref[None, :, None])
    revenue = shipping = served = 0.0
    for s in range(S):
        orders = np.repeat(np.arange(R), D[s])
        rng.shuffle(orders)
        for r in orders:
            avail = np.flatnonzero(stock[s] > 0)
            if avail.size == 0:
                continue
            f = avail[np.argmin(order_cost[0, r, avail])]
            stock[s, f] -= 1
            served += 1
            revenue += margin[s]
            shipping += order_cost[0, r, f]
    leftover = stock.sum()
    profit = revenue - shipping - hold * leftover
    return profit, served, shipping, leftover


def main(data="data", out="results"):
    net, fc, styles, hist = load(data)
    rng = np.random.default_rng(SEED)
    regions = net["regions"]
    styles_idx = sorted(fc.style_id.unique())
    mean = fc.pivot(index="style_id", columns="region", values="forecast").reindex(
        index=styles_idx, columns=regions).values                          # S x R
    margin = styles.loc[styles_idx, "unit_margin"].values
    supply = np.round(1.05 * mean.sum(axis=1))

    X_h = heuristic(hist, styles_idx, regions, supply, net)
    X_d = solve_lp(mean[None], supply, margin, net, "deterministic")
    scen = rng.poisson(mean[None].repeat(N_SCENARIOS, axis=0))
    X_s = solve_lp(scen.astype(float), supply, margin, net, "stochastic")

    # Evaluate on fresh demand from the true process for the planning week.
    truth = json.load(open(os.path.join(data, "truth.json")))
    from simulate import demand_mean, REGION_SIZE  # noqa: E402
    cat = styles.loc[styles_idx, "category"].values
    base = styles.loc[styles_idx, "base_rate"].values
    woy = (36 + 34) % 52
    true_mu = np.array([[demand_mean(base[s], REGION_SIZE[r], truth["season_amp"][cat[s]], woy, 0, 1.0)
                         for r in range(len(regions))] for s in range(len(styles_idx))])
    cost = np.array(net["ship_cost"])
    pref = cost.argmin(axis=1)
    hold = net["holding_cost_per_unit_week"]
    policies = {"1. heuristic split by historical share": X_h,
                "2. deterministic LP on point forecast": X_d,
                "3. stochastic LP over 60 scenarios": X_s}
    results = {k: [] for k in policies}
    eval_rng = np.random.default_rng(SEED + 1)
    for i in range(N_EVAL):
        D = eval_rng.poisson(true_mu)
        for k, X in policies.items():
            p, served, ship, left = fulfill(X, D, cost, pref, net["split_handling_cost"], margin, hold,
                                            np.random.default_rng(i))
            results[k].append((p, served / D.sum(), ship / max(served, 1), left))
    rows = []
    for k, v in results.items():
        v = np.array(v)
        rows.append({"policy": k, "units_allocated": int(policies[k].sum()),
                     "mean_weekly_profit": round(v[:, 0].mean(), 0),
                     "profit_5th_percentile": round(np.percentile(v[:, 0], 5), 0),
                     "fill_rate": round(v[:, 1].mean(), 4),
                     "shipping_cost_per_unit": round(v[:, 2].mean(), 2),
                     "units_left_over": round(v[:, 3].mean(), 1)})
    res = pd.DataFrame(rows)
    base_profit = res.mean_weekly_profit.iloc[0]
    res["profit_vs_heuristic"] = (res.mean_weekly_profit / base_profit - 1).round(4)
    res.to_csv(os.path.join(out, "tables", "allocation_policies.csv"), index=False)
    comp = []
    stoch = np.array(results["3. stochastic LP over 60 scenarios"])[:, 0]
    for other in ("1. heuristic split by historical share", "2. deterministic LP on point forecast"):
        paired = stoch - np.array(results[other])[:, 0]   # same demand draws, so differences are paired
        se = paired.std(ddof=1) / np.sqrt(len(paired))
        comp.append({"comparison": f"stochastic LP minus {other[3:]}", "mean_weekly_profit_gain": round(paired.mean(), 1),
                     "ci95_low": round(paired.mean() - 1.96 * se, 1), "ci95_high": round(paired.mean() + 1.96 * se, 1),
                     "share_of_weeks_better": round((paired > 0).mean(), 3)})
    pd.DataFrame(comp).to_csv(os.path.join(out, "tables", "paired_policy_comparison.csv"), index=False)
    np.save(os.path.join(data, "policy_profits.npy"), np.array([np.array(v)[:, 0] for v in results.values()]))
    alloc = pd.DataFrame({"fc": net["fcs"], "capacity": net["fc_capacity_units"],
                          "heuristic": X_h.sum(0), "deterministic_lp": X_d.sum(0), "stochastic_lp": X_s.sum(0)})
    alloc.to_csv(os.path.join(out, "tables", "units_by_fc.csv"), index=False)
    np.save(os.path.join(data, "allocations.npy"), np.stack([X_h, X_d, X_s]))
    print(res.to_string(index=False))
    print(alloc.to_string(index=False))
    print(pd.DataFrame(comp).to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    main()
