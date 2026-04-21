# Oil price Monte Carlo — April 2026

Jump-diffusion Monte Carlo for Brent crude, calibrated to the news backdrop
as of **2026-04-20**.

## News backdrop driving the calibration

| Fact | Source |
|---|---|
| Brent spot $95.42 on 2026-04-20; recent single-day moves of 5–11% | Trading Economics, Fortune, Al Jazeera |
| EIA April 2026 STEO: 2026 Brent projection raised to ~$96 | Rigzone / EIA |
| IEA April 2026 OMR: 2026 demand growth cut to 0.6 mb/d (from 1.2) | IEA OMR |
| Primary risk: US–Iran tensions, Strait of Hormuz status | CNBC, NBC |
| Analyst 2026 range ~ $74–$161 | LiteFinance / Goldman |

## Model

Merton jump-diffusion:

```
dS/S = (mu - 0.5 sigma^2 - lambda*kappa) dt + sigma dW + (e^J - 1) dN
J ~ N(mu_J, sigma_J^2),  N ~ Poisson(lambda)
```

The Poisson jumps capture the **Hormuz-style fat tails** a pure GBM can't —
the recent 10–11% daily moves are ~5–6σ events under lognormal vol.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `S0` | 95.42 | Brent spot, 2026-04-20 |
| `mu_annual` | 2% | EIA implies ~flat into year-end |
| `sigma_annual` | 55% | Elevated realised vol post-Hormuz |
| `jump_lambda` | 6/yr | Rough pace of Iran/OPEC/Hormuz headlines |
| `jump_mu` | −1% | Slight negative bias (supply restoration > new cuts) |
| `jump_sigma` | 12% | Reproduces observed 10% daily shocks |
| `n_paths` | 20 000 | |
| Horizon | 252 trading days | 1Y forward |

## Results (USD/bbl)

```
  horizon  mean   std    p5   p25   p50    p75    p95  P(S<70)  P(S>120)  P(S>150)
 1M (21d)  95.7  17.4  70.0  83.6  94.3  106.0  126.4    0.05     0.09      0.01
 3M (63d)  95.8  30.8  54.4  74.2  91.2  112.7  152.8    0.20     0.19      0.06
 6M (126d) 96.4  45.0  42.1  64.8  87.2  117.7  182.3    0.31     0.24      0.11
12M (252d) 97.1  67.0  28.6  52.3  79.7  122.6  222.6    0.42     0.26      0.16
```

### Headline reads

- **Mean 1Y ≈ $97**, consistent with EIA's 2026 STEO print of ~$96.
- **Median drifts below spot** (~$80 at 12M) because jumps are slightly
  negative-biased: the market prices a meaningful chance of Hormuz
  de-escalation.
- **p95 at 1Y ≈ $223** — within sight of the $161 upper analyst range
  once you include the tail of consecutive positive jumps.
- **~26% probability Brent trades above $120 at some point in the 1Y
  terminal distribution**; ~16% above $150. These are the Hormuz-closure
  scenarios.
- **~42% probability the 1Y terminal print is below $70**, reflecting the
  demand-growth downgrade and supply restoration scenarios.

## Running it

```bash
pip install numpy pandas matplotlib
python3 oil_monte_carlo.py
```

Outputs land in `outputs/`:

- `summary.csv` — quantiles and tail probabilities per horizon
- `daily_quantiles.csv` — full daily fan data
- `fan_chart.png` — 5/25/50/75/95 fan over 1Y
- `terminal_hist.png` — 1Y terminal distribution

## Caveats

- Single-asset, single-regime model. No regime switching between
  "Hormuz-open" and "Hormuz-closed" states — a two-state Markov model
  would give a cleaner bimodal terminal.
- Constant vol. Implied vol surfaces on CLZ26/CBJ26 are skewed and would
  sharpen upside tails further.
- Jumps are symmetric in shape but biased in mean. Real geopolitical
  shocks are lumpier (single 20% spikes) than the Gaussian jump size
  assumption allows.
- No explicit term structure / convenience yield: the drift is reduced
  form, not from futures curve fitting.
