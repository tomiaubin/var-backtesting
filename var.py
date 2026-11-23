"""Value at Risk (VaR) and Expected Shortfall (ES) of a stock portfolio, with backtesting.

Three ways of computing VaR/ES are compared (historical, parametric, Monte Carlo) and
then tested on 10 years of real data: how many times did the loss exceed the VaR?

Losses are always reported as positive numbers (a VaR of 0.02 means a 2% loss).
Run `python var.py` to reproduce the figures and the numbers in the README.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
WINDOW = 250  # trading days (about 1 year) used to estimate the risk


# ---------------------------------------------------------------- data
def load_returns(csv="data/prices.csv"):
    """Daily returns of each stock. Prices are downloaded once and saved to a CSV."""
    if not os.path.exists(csv):
        import yfinance as yf
        prices = yf.download(TICKERS, start="2015-01-01", end="2025-12-31",
                             auto_adjust=True, progress=False)["Close"][TICKERS].dropna()
        prices.to_csv(csv)
    prices = pd.read_csv(csv, index_col=0, parse_dates=True)
    return prices.pct_change().dropna()


# ---------------------------------------------------------------- risk measures
# `returns` is a 1D array with the portfolio's daily returns.
# `alpha` is the confidence level (0.99 = "in 99% of days the loss is below the VaR").

def historical(returns, alpha):
    """Use the past returns directly, with no assumption about their distribution."""
    losses = -np.asarray(returns)
    var = np.percentile(losses, alpha * 100)
    es = losses[losses >= var].mean()      # average of the losses beyond the VaR
    return var, es


def parametric(returns, alpha):
    """Assume returns are normal: only the mean and standard deviation are needed."""
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    z = norm.ppf(alpha)
    var = -mu + sigma * z
    es = -mu + sigma * norm.pdf(z) / (1 - alpha)   # closed formula for the normal case
    return var, es


def monte_carlo(asset_returns, weights, alpha, n=100_000):
    """Simulate many possible days using the mean and covariance of the assets,
    then take the VaR/ES of the simulated portfolio returns."""
    mu = asset_returns.mean().values
    cov = asset_returns.cov().values
    simulated = np.random.multivariate_normal(mu, cov, size=n) @ weights
    return historical(simulated, alpha)


# ---------------------------------------------------------------- backtesting
def backtest(asset_returns, weights, alpha, method):
    """For each day, estimate VaR with the previous WINDOW days and compare it with
    the loss that actually happened that day. Returns the VaR series and the losses."""
    port = asset_returns.values @ weights
    var_list, loss_list, dates = [], [], []
    for t in range(WINDOW, len(port)):
        past = port[t - WINDOW:t]
        if method == "historical":
            var, _ = historical(past, alpha)
        elif method == "parametric":
            var, _ = parametric(past, alpha)
        else:
            var, _ = monte_carlo(asset_returns.iloc[t - WINDOW:t], weights, alpha, n=5000)
        var_list.append(var)
        loss_list.append(-port[t])
        dates.append(asset_returns.index[t])
    return pd.Series(var_list, index=dates), pd.Series(loss_list, index=dates)


def kupiec_test(n_exceptions, n_days, alpha):
    """Kupiec proportion-of-failures test.
    H0: the true exception rate equals 1 - alpha. Returns the p-value
    (a small p-value, e.g. < 0.05, means the VaR model is rejected)."""
    p = 1 - alpha
    x, T = n_exceptions, n_days
    # Log-likelihood of the data if the exception rate is p (H0) ...
    log_lik_null = (T - x) * np.log(1 - p) + x * np.log(p)
    # ... and if it is the observed rate x/T (the best possible fit)
    log_lik_obs = (T - x) * np.log(1 - x / T)
    if x > 0:
        log_lik_obs += x * np.log(x / T)
    lr = -2 * (log_lik_null - log_lik_obs)   # approximately chi-square with 1 d.o.f. under H0
    return 1 - chi2.cdf(lr, df=1)


def basel_zone(n_exceptions_250):
    """Basel 'traffic light' for 99% VaR: exceptions in the last 250 days."""
    if n_exceptions_250 <= 4:
        return "green"
    if n_exceptions_250 <= 9:
        return "yellow"
    return "red"


# ---------------------------------------------------------------- main
def main():
    np.random.seed(0)
    asset_returns = load_returns()
    weights = np.ones(len(TICKERS)) / len(TICKERS)       # equal-weighted portfolio
    port = asset_returns.values @ weights
    print(f"{len(port)} daily returns, {asset_returns.index[0].date()} to {asset_returns.index[-1].date()}")

    # 1) One-day VaR and ES with all the data, three methods
    print("\nOne-day VaR / ES using the full sample")
    print(f"{'method':<12}{'alpha':>6}{'VaR':>9}{'ES':>9}")
    for alpha in (0.95, 0.99):
        for name, (var, es) in {"historical": historical(port, alpha),
                                "parametric": parametric(port, alpha),
                                "monte carlo": monte_carlo(asset_returns, weights, alpha)}.items():
            print(f"{name:<12}{alpha:>6}{var:>9.2%}{es:>9.2%}")

    # Histogram of returns with the 99% VaR / ES
    var99, es99 = historical(port, 0.99)
    plt.figure(figsize=(8, 4.5))
    plt.hist(port, bins=100, color="lightgray", density=True)
    x = np.linspace(port.min(), port.max(), 300)
    plt.plot(x, norm.pdf(x, port.mean(), port.std()), "k", lw=1, label="normal fit")
    plt.axvline(-var99, color="C1", label=f"99% VaR = {var99:.2%}")
    plt.axvline(-es99, color="C3", label=f"99% ES = {es99:.2%}")
    plt.xlabel("daily portfolio return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/returns_var.png", dpi=150)

    # 2) Backtest: rolling VaR vs real losses
    print("\nBacktest (rolling window of 250 days)")
    print(f"{'method':<12}{'alpha':>6}{'days':>7}{'exceptions':>12}{'expected':>10}{'Kupiec p':>10}")
    series = {}
    for alpha in (0.95, 0.99):
        for method in ("historical", "parametric", "monte carlo"):
            var, loss = backtest(asset_returns, weights, alpha, method)
            exceptions = loss > var
            n_exc, n_days = int(exceptions.sum()), len(loss)
            p_value = kupiec_test(n_exc, n_days, alpha)
            print(f"{method:<12}{alpha:>6}{n_days:>7}{n_exc:>12}{(1 - alpha) * n_days:>10.1f}{p_value:>10.3f}")
            series[(method, alpha)] = (var, loss, exceptions)

    # Basel traffic light on the last 250 days (99% VaR)
    print("\nBasel traffic light (last 250 days, 99% VaR)")
    for method in ("historical", "parametric", "monte carlo"):
        n_exc = int(series[(method, 0.99)][2].iloc[-250:].sum())
        print(f"{method:<12}{n_exc:>3} exceptions -> {basel_zone(n_exc)}")

    # The same traffic light during the COVID crash (250 days from 2020-01-01)
    print("\nBasel traffic light during 2020 (250 days from 2020-01-01)")
    for method in ("historical", "parametric", "monte carlo"):
        exc = series[(method, 0.99)][2]
        n_exc = int(exc.loc["2020-01-01":].iloc[:250].sum())
        print(f"{method:<12}{n_exc:>3} exceptions -> {basel_zone(n_exc)}")

    # Figure: losses, VaR and exceptions for the parametric method at 99%
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, method in zip(axes, ("parametric", "historical")):
        var, loss, exc = series[(method, 0.99)]
        ax.plot(loss.index, loss, color="lightgray", lw=0.7, label="daily loss")
        ax.plot(var.index, var, color="C0", lw=1, label="99% VaR")
        ax.scatter(loss.index[exc], loss[exc], color="C3", s=12, zorder=3,
                   label=f"exceptions ({exc.sum()})")
        ax.set_title(method)
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("imgs/backtest_99.png", dpi=150)


if __name__ == "__main__":
    main()
