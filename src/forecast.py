"""Explainable demand forecasting with a Poisson generalized linear model.

log E[units] = style + region + category-specific annual seasonality
               + promotion + price elasticity x log(price ratio)

Every coefficient has a business meaning (a promotion lift, an elasticity, a
seasonal amplitude by category), so the forecast can be explained to
merchandising and finance partners, and here it can be checked against the
coefficients used to simulate the data.

The model is trained on weeks 0 to 25 and evaluated on weeks 26 to 33 against
two baselines that planners commonly use, then refit on all 34 weeks to
forecast the planning week.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FORMULA = ("units ~ C(style_id) + C(region) + C(category):cos_w + C(category):sin_w "
           "+ promo + log_price")


def features(df):
    df = df.copy()
    df["cos_w"] = np.cos(2 * np.pi * (df.week_of_year - 2) / 52)
    df["sin_w"] = np.sin(2 * np.pi * (df.week_of_year - 2) / 52)
    df["log_price"] = np.log(df.price_ratio)
    return df


def fit(df):
    return smf.glm(FORMULA, data=features(df), family=sm.families.Poisson()).fit()


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.maximum(np.asarray(yhat, float), 0.05)  # floor so deviance is finite
    wape = np.abs(y - yhat).sum() / y.sum()
    bias = (yhat.sum() - y.sum()) / y.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.where(y > 0, y * np.log(y / yhat), 0) - 2 * (y - yhat)
    return {"WAPE": round(wape, 4), "bias": round(bias, 4), "mean_poisson_deviance": round(dev.mean(), 4)}


def main(data="data", out="results"):
    hist = pd.read_csv(os.path.join(data, "history.csv"))
    truth = json.load(open(os.path.join(data, "truth.json")))
    train, test = hist[hist.week <= 25], hist[hist.week >= 26].copy()

    m = fit(train)
    test["glm"] = m.predict(features(test))
    # Baselines: mean of the last 4 weeks and of the last 8 weeks of training data per style and region.
    for k in (4, 8):
        base = train[train.week > 25 - k].groupby(["style_id", "region"]).units.mean().rename(f"ma{k}")
        test = test.merge(base, on=["style_id", "region"])
    rows = []
    for name, col in [("Poisson GLM (explainable)", "glm"), ("moving average, 4 weeks", "ma4"),
                      ("moving average, 8 weeks", "ma8"), ("oracle: true simulated mean", "true_mean")]:
        rows.append({"model": name, **metrics(test.units, test[col])})
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(out, "tables", "forecast_accuracy.csv"), index=False)

    # Coefficient recovery
    p, ci = m.params, m.conf_int()
    rec = [{"parameter": "promotion lift (log)", "true": truth["promo_lift_log"],
            "estimate": p["promo"], "ci_low": ci.loc["promo", 0], "ci_high": ci.loc["promo", 1]},
           {"parameter": "price elasticity", "true": truth["price_elasticity"],
            "estimate": p["log_price"], "ci_low": ci.loc["log_price", 0], "ci_high": ci.loc["log_price", 1]}]
    for cat, amp in truth["season_amp"].items():
        key = f"C(category)[{cat}]:cos_w"
        rec.append({"parameter": f"seasonal amplitude, {cat}", "true": amp, "estimate": p[key],
                    "ci_low": ci.loc[key, 0], "ci_high": ci.loc[key, 1]})
    rec = pd.DataFrame(rec)
    rec[["estimate", "ci_low", "ci_high"]] = rec[["estimate", "ci_low", "ci_high"]].round(3)
    rec["true_inside_95ci"] = (rec.ci_low <= rec.true) & (rec.true <= rec.ci_high)
    rec.to_csv(os.path.join(out, "tables", "forecast_coefficients.csv"), index=False)

    # Forecast for the planning week (week 34) from a model refit on all history:
    # no promotion, list price.
    full = fit(hist)
    nxt = hist[hist.week == hist.week.max()][["style_id", "category", "region"]].copy()
    nxt["week"] = hist.week.max() + 1
    nxt["week_of_year"] = (36 + nxt.week) % 52
    nxt["promo"], nxt["price_ratio"] = 0, 1.0
    nxt["forecast"] = full.predict(features(nxt))
    nxt.to_csv(os.path.join(data, "forecast_next_week.csv"), index=False)
    print(acc.to_string(index=False))
    print(rec.to_string(index=False))
    print(f"planning week forecast: {nxt.forecast.sum():.0f} units")


if __name__ == "__main__":
    main()
