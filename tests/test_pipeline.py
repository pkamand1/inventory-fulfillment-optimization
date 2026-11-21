"""Checks on the outputs of run_all.sh: constraints hold, the forecast
recovers the true drivers, and fulfillment conserves units."""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from allocate import fulfill  # noqa: E402

DATA, TAB = os.path.join(ROOT, "data"), os.path.join(ROOT, "results", "tables")


def test_allocations_respect_capacity_and_supply():
    net = json.load(open(os.path.join(DATA, "network.json")))
    fc = pd.read_csv(os.path.join(DATA, "forecast_next_week.csv"))
    supply = np.round(1.05 * fc.groupby("style_id").forecast.sum().sort_index().values)
    for X in np.load(os.path.join(DATA, "allocations.npy")):
        assert (X >= 0).all() and np.allclose(X, np.round(X))
        assert (X.sum(axis=0) <= np.array(net["fc_capacity_units"]) + 1e-9).all()
        assert (X.sum(axis=1) <= supply + 1e-9).all()


def test_forecast_recovers_true_coefficients():
    rec = pd.read_csv(os.path.join(TAB, "forecast_coefficients.csv"))
    assert rec.true_inside_95ci.all()


def test_glm_beats_moving_averages():
    acc = pd.read_csv(os.path.join(TAB, "forecast_accuracy.csv")).set_index("model")
    assert acc.loc["Poisson GLM (explainable)", "WAPE"] < acc.loc["moving average, 4 weeks", "WAPE"]
    assert acc.loc["Poisson GLM (explainable)", "WAPE"] < acc.loc["moving average, 8 weeks", "WAPE"]


def test_fulfillment_conserves_units():
    rng = np.random.default_rng(0)
    X = np.array([[3, 0], [0, 2]], float)
    D = np.array([[2, 4], [1, 0]])
    cost = np.array([[5.0, 9.0], [9.0, 5.0]])
    pref = np.array([0, 1])
    profit, served, ship, left = fulfill(X, D, cost, pref, 2.5, np.array([20.0, 20.0]), 0.0, rng)
    assert served == 4                      # style 0: 3 units for 6 orders; style 1: 1 order filled from FC 1
    assert left == X.sum() - served
    assert ship > 0 and profit == 20 * served - ship
