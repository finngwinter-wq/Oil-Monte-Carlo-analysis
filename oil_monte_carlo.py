"""
Monte Carlo analysis of Brent crude oil price.

Calibration is driven by news available on 2026-04-20:
    - Brent spot ~ $95.42/bbl (Fortune, Trading Economics)
    - EIA April 2026 STEO: 2026 Brent projection ~ $96/bbl
    - IEA April 2026 OMR: demand growth revised down to 0.6 mb/d (from 1.2)
    - Realised daily moves of 10-11% around Strait of Hormuz headlines
    - Analyst 2026 range roughly $74 - $161 (Goldman, LiteFinance)
    - Primary risk: US-Iran tensions / Hormuz chokepoint (~20% of seaborne oil)

Model: Merton jump-diffusion (GBM + Poisson jumps).

    dS/S = (mu - 0.5*sigma^2) dt + sigma dW + (e^J - 1) dN
    J ~ Normal(mu_J, sigma_J), N ~ Poisson(lambda)

Jumps capture the Hormuz-driven fat tails that a pure lognormal misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    S0: float = 95.42           # Brent spot on 2026-04-20 (USD/bbl)
    horizon_days: int = 252     # 1 trading year
    dt: float = 1.0 / 252
    mu_annual: float = 0.02     # mild drift: EIA implies ~flat into YE 2026
    sigma_annual: float = 0.55  # elevated realised vol post-Hormuz
    jump_lambda: float = 6.0    # ~6 shock events/yr (Iran/OPEC/Hormuz headlines)
    jump_mu: float = -0.01      # slight negative bias (supply returns > new cuts)
    jump_sigma: float = 0.12    # ~12% std per jump -> reproduces 10% daily moves
    n_paths: int = 20_000
    seed: int = 20260420


def simulate(p: Params) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    n_steps = p.horizon_days

    # Risk-neutral-style drift correction so E[S_t] stays consistent with mu.
    kappa = np.exp(p.jump_mu + 0.5 * p.jump_sigma ** 2) - 1.0
    drift = (p.mu_annual - 0.5 * p.sigma_annual ** 2 - p.jump_lambda * kappa) * p.dt

    z = rng.standard_normal((n_steps, p.n_paths))
    diffusion = p.sigma_annual * np.sqrt(p.dt) * z

    n_jumps = rng.poisson(p.jump_lambda * p.dt, size=(n_steps, p.n_paths))
    # Sum of n_jumps iid normals == Normal(n*mu_J, n*sigma_J^2).
    jump_mean = n_jumps * p.jump_mu
    jump_std = np.sqrt(n_jumps) * p.jump_sigma
    jumps = jump_mean + jump_std * rng.standard_normal(n_jumps.shape)

    log_returns = drift + diffusion + jumps
    log_paths = np.concatenate(
        [np.zeros((1, p.n_paths)), np.cumsum(log_returns, axis=0)], axis=0
    )
    return p.S0 * np.exp(log_paths)


def summarise(paths: np.ndarray, p: Params) -> pd.DataFrame:
    horizons = {
        "1M (21d)": 21,
        "3M (63d)": 63,
        "6M (126d)": 126,
        "12M (252d)": 252,
    }
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    rows = []
    for label, h in horizons.items():
        s = paths[h]
        row = {"horizon": label, "mean": s.mean(), "std": s.std()}
        for q in quantiles:
            row[f"p{int(q * 100)}"] = np.quantile(s, q)
        row["P(S < 70)"] = float((s < 70).mean())
        row["P(S > 120)"] = float((s > 120).mean())
        row["P(S > 150)"] = float((s > 150).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fan(paths: np.ndarray, p: Params, out: Path) -> None:
    q = np.quantile(paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1)
    t = np.arange(paths.shape[0]) / 252.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(t, q[0], q[4], alpha=0.15, label="5-95%")
    ax.fill_between(t, q[1], q[3], alpha=0.3, label="25-75%")
    ax.plot(t, q[2], linewidth=2, label="median")
    ax.axhline(p.S0, linestyle="--", linewidth=1, label=f"spot ${p.S0:.2f}")
    ax.set_xlabel("Years from 2026-04-20")
    ax.set_ylabel("Brent (USD/bbl)")
    ax.set_title("Brent crude Monte Carlo - jump-diffusion, 20k paths")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_terminal_hist(paths: np.ndarray, p: Params, out: Path) -> None:
    terminal = paths[-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(terminal, bins=80, alpha=0.8)
    for q, style in [(0.05, ":"), (0.5, "-"), (0.95, ":")]:
        v = np.quantile(terminal, q)
        ax.axvline(v, linestyle=style, linewidth=1.2, label=f"p{int(q*100)} = ${v:.1f}")
    ax.axvline(p.S0, linestyle="--", color="black", linewidth=1, label=f"spot ${p.S0:.2f}")
    ax.set_xlabel("Brent 12M terminal price (USD/bbl)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of 12-month terminal Brent")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = Params()
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    paths = simulate(p)
    summary = summarise(paths, p)
    summary.to_csv(out_dir / "summary.csv", index=False)

    pd.DataFrame(
        {
            "day": np.arange(paths.shape[0]),
            "p05": np.quantile(paths, 0.05, axis=1),
            "p25": np.quantile(paths, 0.25, axis=1),
            "p50": np.quantile(paths, 0.50, axis=1),
            "p75": np.quantile(paths, 0.75, axis=1),
            "p95": np.quantile(paths, 0.95, axis=1),
            "mean": paths.mean(axis=1),
        }
    ).to_csv(out_dir / "daily_quantiles.csv", index=False)

    plot_fan(paths, p, out_dir / "fan_chart.png")
    plot_terminal_hist(paths, p, out_dir / "terminal_hist.png")

    print("Monte Carlo summary (USD/bbl):")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
