"""
100 Brent scenarios from 2026-04-20 to 2030-12-31 under geometric Brownian motion.

Pure GBM (no jumps), per request:

    dS/S = mu dt + sigma dW
    S_t = S_0 * exp((mu - 0.5 sigma^2) t + sigma W_t)

Calibration (same news backdrop as oil_monte_carlo.py):
    S0    = $95.42/bbl  (Brent spot, 2026-04-20)
    mu    = 2% / yr     (EIA 2026 STEO trajectory ~flat into YE)
    sigma = 45% / yr    (elevated but below jump-diffusion sigma since
                         pure GBM has no jump component)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S0 = 95.42
MU = 0.02
SIGMA = 0.45
N_PATHS = 100
SEED = 20260420

START = pd.Timestamp("2026-04-20")
END = pd.Timestamp("2030-12-31")


def business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def simulate_gbm(
    s0: float, mu: float, sigma: float, n_paths: int, n_steps: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    drift = (mu - 0.5 * sigma ** 2) * dt
    shock = sigma * np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
    log_returns = drift + shock
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_returns, axis=0)])
    return s0 * np.exp(log_paths)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    dates = business_days(START, END)
    n_steps = len(dates) - 1
    paths = simulate_gbm(S0, MU, SIGMA, N_PATHS, n_steps, SEED)

    df = pd.DataFrame(
        paths,
        index=dates,
        columns=[f"scenario_{i+1:03d}" for i in range(N_PATHS)],
    )
    df.index.name = "date"
    df.to_csv(out_dir / "brent_gbm_100_scenarios_2030.csv", float_format="%.4f")

    terminal = paths[-1]
    term_stats = {
        "mean": terminal.mean(),
        "std": terminal.std(),
        "min": terminal.min(),
        "p5": np.quantile(terminal, 0.05),
        "p50": np.quantile(terminal, 0.50),
        "p95": np.quantile(terminal, 0.95),
        "max": terminal.max(),
    }
    pd.Series(term_stats).to_csv(out_dir / "brent_gbm_2030_terminal_stats.csv")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dates, paths, linewidth=0.7, alpha=0.55)
    ax.plot(dates, np.median(paths, axis=1), color="black", linewidth=2, label="median")
    ax.axhline(S0, linestyle="--", color="grey", linewidth=1, label=f"spot ${S0:.2f}")
    ax.set_title("Brent crude - 100 GBM scenarios, 2026-04-20 to 2030-12-31")
    ax.set_xlabel("Date")
    ax.set_ylabel("Brent (USD/bbl)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "brent_gbm_100_scenarios_2030.png", dpi=150)
    plt.close(fig)

    print(f"Simulated {N_PATHS} paths, {n_steps + 1} business days to {END.date()}.")
    print("Terminal (2030-12-31) stats:")
    for k, v in term_stats.items():
        print(f"  {k:>4}: ${v:,.2f}")


if __name__ == "__main__":
    main()
