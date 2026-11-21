"""Figures for the README."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(data="data", out="results"):
    f = os.path.join(out, "figures")
    os.makedirs(f, exist_ok=True)
    t = os.path.join(out, "tables")

    rec = pd.read_csv(os.path.join(t, "forecast_coefficients.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(rec))[::-1]
    ax.errorbar(rec.estimate, y, xerr=[rec.estimate - rec.ci_low, rec.ci_high - rec.estimate],
                fmt="o", color="#1f77b4", label="estimate, 95% CI")
    ax.scatter(rec.true, y, marker="x", color="#d62728", zorder=3, label="true value")
    ax.set_yticks(y, rec.parameter)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title("Forecast model recovers the true demand drivers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(f, "forecast_coefficients.png"), dpi=150)
    plt.close(fig)

    prof = np.load(os.path.join(data, "policy_profits.npy"))
    labels = ["heuristic", "deterministic LP", "stochastic LP"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, lab, c in zip(prof, labels, ["#7f7f7f", "#1f77b4", "#2ca02c"]):
        ax.hist(p / 1000, bins=40, alpha=0.55, label=f"{lab} (mean ${p.mean() / 1000:.1f}k)", color=c)
    ax.set_xlabel("Weekly profit after shipping and holding ($ thousands)")
    ax.set_ylabel("Simulated weeks (of 1,000)")
    ax.set_title("Allocation policies evaluated on unseen demand")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(f, "allocation_policies.png"), dpi=150)
    plt.close(fig)

    ro = pd.read_csv(os.path.join(t, "reorder_policy_tradeoff.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ro.avg_units_on_hand, ro.fill_rate, "o-")
    for _, r in ro.iloc[::2].iterrows():
        ax[0].annotate(f"z={r.safety_factor_z:g}", (r.avg_units_on_hand, r.fill_rate), fontsize=7,
                       xytext=(4, -10), textcoords="offset points")
    ax[0].set_xlabel("Average units on hand")
    ax[0].set_ylabel("Fill rate")
    ax[0].set_title("Service level bought with inventory")
    best = ro.loc[ro.weekly_cost_holding_plus_lost_margin.idxmin()]
    ax[1].plot(ro.safety_factor_z, ro.weekly_cost_holding_plus_lost_margin, "o-", color="#d62728")
    ax[1].axvline(best.safety_factor_z, ls=":", color="k")
    ax[1].set_xlabel("Safety factor z")
    ax[1].set_ylabel("Holding cost + lost margin per week ($)")
    ax[1].set_title(f"Lowest total cost at z = {best.safety_factor_z:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(f, "reorder_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
