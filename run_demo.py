"""Run the full pipeline and save charts to docs/.

    python run_demo.py                         # offline, synthetic data
    python run_demo.py --source yahoo          # real prices + Fama-French factors
    python run_demo.py --source yahoo --tickers AAPL MSFT JPM XOM JNJ
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from factormodel import (FactorRiskModel, efficient_frontier, estimate_exposures,
                         max_sharpe, min_variance, performance_summary)
from factormodel.attribution import factor_attribution
from factormodel.backtest import walk_forward
from factormodel import data

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "CVX", "PFE",
                   "BAC", "INTC", "CSCO", "MRK", "HD", "DIS", "MCD", "IBM", "CAT", "MMM"]


def load(source: str, tickers: list[str], start: str):
    if source == "synthetic":
        returns, factors, _, rf = data.synthetic_market()
        return returns, factors, rf
    prices = data.load_prices(tickers, start)
    returns = data.prices_to_returns(prices)
    ff = data.load_fama_french_3()
    common = returns.index.intersection(ff.index)
    returns, ff = returns.loc[common].dropna(axis=1), ff.loc[common]
    excess = returns.sub(ff["RF"], axis=0)
    return excess, ff[["MKT_RF", "SMB", "HML"]], ff["RF"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yahoo"], default="synthetic")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--max-weight", type=float, default=0.15)
    args = ap.parse_args()
    out = Path("docs")
    out.mkdir(exist_ok=True)

    excess, factors, rf = load(args.source, args.tickers, args.start)
    label = "synthetic data" if args.source == "synthetic" else "US stocks, Fama-French 3"
    print(f"Data: {excess.shape[1]} assets x {excess.shape[0]} months ({args.source})\n")

    # 1. Factor exposures
    expo = estimate_exposures(excess, factors)
    print("Factor exposures (first 5 assets):")
    print(expo.head().round(3), "\n")

    # 2. Risk model and optimal portfolios
    betas = expo[factors.columns]
    model = FactorRiskModel(betas, factors, expo["resid_var"])
    cov = model.covariance()
    mu = betas @ factors.mean()
    w_mv = min_variance(cov, args.max_weight)
    w_ms = max_sharpe(mu, cov, max_weight=args.max_weight)
    print("Max-Sharpe weights (>1%):")
    print(w_ms[w_ms > 0.01].sort_values(ascending=False).round(3), "\n")
    risk = model.risk_decomposition(w_ms)
    print(f"Max-Sharpe monthly vol {risk['total_vol']:.4f}; "
          f"factor share of variance {risk['factor_var'] / risk['total_vol']**2:.0%}\n")

    # 3. Efficient frontier chart
    frontier = efficient_frontier(mu, cov, max_weight=args.max_weight)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frontier["vol"] * 12 ** 0.5, frontier["ret"] * 12, label="Efficient frontier")
    for w, name in [(w_mv, "Min variance"), (w_ms, "Max Sharpe")]:
        v = (w @ cov @ w) ** 0.5 * 12 ** 0.5
        ax.scatter(v, (w @ mu) * 12, zorder=3, label=name)
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected excess return")
    ax.set_title(f"Long-only efficient frontier, max weight {args.max_weight:.0%} ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "efficient_frontier.png", dpi=150)

    # 4. Walk-forward backtest vs equal weight
    strat = lambda m, c: max_sharpe(m, c, max_weight=args.max_weight)
    port, _ = walk_forward(excess, factors, strat, window=60)
    ew = excess.loc[port.index].mean(axis=1).rename("equal_weight")
    rf_bt = rf.loc[port.index] if isinstance(rf, pd.Series) else 0.0
    summary = pd.DataFrame({
        "max_sharpe": performance_summary(port + rf_bt, ew + rf_bt, rf_bt),
        "equal_weight": performance_summary(ew + rf_bt, rf=rf_bt),
    })
    print("Walk-forward backtest (out of sample):")
    print(summary.round(3), "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    (1 + port).cumprod().plot(ax=ax, label="Max Sharpe (factor model)")
    (1 + ew).cumprod().plot(ax=ax, label="Equal weight")
    ax.set_title(f"Out-of-sample growth of 1, monthly rebalance ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "backtest.png", dpi=150)

    # 5. Factor attribution of the backtest (using full-sample betas, for illustration)
    avg_w = w_ms
    attr = factor_attribution(avg_w, betas, factors.loc[port.index], port)
    print("Average monthly return attribution:")
    print(attr.mean().round(4))
    summary.round(4).to_csv(out / "backtest_summary.csv")
    print(f"\nCharts saved to {out}/")


if __name__ == "__main__":
    main()
