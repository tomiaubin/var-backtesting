"""Tests for var.py. Run with:  pytest"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from var import historical, parametric, monte_carlo, kupiec_test, basel_zone, backtest


def test_historical_var_on_known_data():
    # Returns -1%, -2%, ..., -100% shifted so the worst 1% is easy to know
    returns = -np.arange(1, 101) / 100          # losses are 0.01 ... 1.00
    var, es = historical(returns, 0.95)
    assert abs(var - 0.9505) < 1e-9             # 95th percentile of the losses
    assert es >= var


def test_parametric_matches_normal_formulas():
    np.random.seed(0)
    returns = np.random.normal(0.0, 0.01, 200_000)
    var, es = parametric(returns, 0.99)
    assert abs(var - 0.01 * norm.ppf(0.99)) < 2e-4
    assert es > var


def test_three_methods_agree_on_normal_data():
    np.random.seed(1)
    cov = [[1e-4, 0.5e-4], [0.5e-4, 1e-4]]
    assets = pd.DataFrame(np.random.multivariate_normal([0, 0], cov, 50_000))
    w = np.array([0.5, 0.5])
    port = assets.values @ w
    v_h, _ = historical(port, 0.99)
    v_p, _ = parametric(port, 0.99)
    v_m, _ = monte_carlo(assets, w, 0.99)
    assert abs(v_h - v_p) < 5e-4 and abs(v_m - v_p) < 5e-4


def test_kupiec_accepts_expected_number_of_exceptions():
    # 99% VaR, 1000 days: 10 exceptions is exactly what we expect
    assert kupiec_test(10, 1000, 0.99) > 0.99
    # 30 exceptions is far too many: model rejected
    assert kupiec_test(30, 1000, 0.99) < 0.001
    # zero exceptions must not crash
    assert 0 < kupiec_test(0, 250, 0.99) < 1


def test_basel_zones():
    assert basel_zone(4) == "green"
    assert basel_zone(5) == "yellow"
    assert basel_zone(9) == "yellow"
    assert basel_zone(10) == "red"


def test_backtest_exception_rate_on_normal_data():
    # If returns really are normal, the 99% parametric VaR should fail about 1% of the days
    np.random.seed(2)
    assets = pd.DataFrame(np.random.normal(0, 0.01, (3000, 1)),
                          index=pd.date_range("2010-01-01", periods=3000, freq="B"))
    var, loss = backtest(assets, np.array([1.0]), 0.99, "parametric")
    rate = (loss > var).mean()
    assert 0.005 < rate < 0.02
