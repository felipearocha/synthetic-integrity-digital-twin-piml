# Synthetic Integrity Digital Twin (PIML + PINN Framework)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172483.svg)](https://doi.org/10.5281/zenodo.20172483)

## Overview

This project generates a fully synthetic refinery-style digital twin including:

- 3D equipment visualization (tanks, vessels, columns, pumps, exchangers)
- Routed piping topology
- Corrosion and chemical attributes
- RBI-style integrity metadata
- 100 synthetic SCC alternative scenarios
- Optional Physics-Informed Machine Learning (PIML / PINN) demo

All data is synthetic and anonymized.

This repository is intended for:

- Research
- Digital twin prototyping
- Integrity analytics architecture design
- AI experimentation for corrosion and cracking

It is NOT a certified engineering calculator and does not replace API 579 or ASME FFS assessments.

## Engineering Context

Conceptually aligned with:

- API 570 (Piping Inspection)
- API 571 (Damage Mechanisms)
- API 580 / 581 (RBI)
- API 579-1 / ASME FFS-1 (Fitness-for-Service)
- ASME B31.3 (Process Piping)

These references are conceptual only.

## Features

### 1. Synthetic Equipment Model

- Tagged equipment
- Plot position and simple geometry (height, radius)

### 2. Piping Attributes

- Service type
- Design and operating conditions
- Material specification
- Insulation and coating metadata
- Chlorides
- pH
- H2S
- Corrosion rate (MPY)
- SCC susceptibility score (0–1)

### 3. 3D Interactive Visualization

Generates `synthetic_refinery_unit_3d.html`. Open it in a browser to rotate, zoom and inspect.

### 4. PIML / PINN Demo

Optional physics-informed neural network to:

- Fit an SCC susceptibility score
- Roll out a Paris-type crack growth law, with soft penalties on its parameters
- Demonstrate hybrid physics + ML architecture

## PINN Demo Results

Printed at the end of each run. 100 synthetic scenarios, 80 for training, 20 held out. The baseline is a constant equal to the training mean. Measured with PyTorch 2.15 on CPU.

| Held-out RMSE | Model | Constant baseline |
|---|---|---|
| SCC susceptibility (0–1) | 0.271 | 0.352 |
| Crack depth at 5 years (mm) | 15.67 | 4.26 |

The susceptibility head beats the baseline. The crack depth rollout does not: it fits the training rows (RMSE 0.26 mm) but fails on held-out rows, and 2 of the 20 reach the 50 mm cap. Over 5 different splits, susceptibility beat the baseline every time (0.168–0.291 against 0.288–0.377, one of them by a small margin) and depth lost every time (5.42–15.67 mm against 3.33–4.94 mm).

## Model Equations

Surrogate stress term and stress intensity range:

```math
\sigma = 12\,p\,\mathrm{clip}\left(\frac{d}{0.2},\,0.4,\,2.5\right), \qquad \Delta K = \sigma\sqrt{\frac{a}{10}}
```

with $p$ the operating pressure (MPa), $d$ the nominal size (m) and $a$ the crack depth (mm). The scaling is arbitrary, so $\sigma$ and $\Delta K$ are surrogates and are not in MPa or MPa√m.

Paris-type growth rate with a relative residual:

```math
\frac{da}{dN} = \max\left(0,\ e^{\ln C}\,\Delta K^{m}\left[1 + 0.05\,r\left(s - \frac{1}{2}\right)\right]\right)
```

The network outputs $\ln C$, $m$, $r \ge 0$ and the susceptibility $s$ for each row.

Explicit rollout over 5 years in 49 steps, at an assumed 200,000 cycles per year:

```math
a_{k+1} = \min\left(a_k + \frac{da}{dN}\,\Delta N,\ 50\right), \qquad a_0 = 0.5, \qquad \Delta N = \frac{5 \times 200000}{49} \approx 20408
```

Training loss:

```math
L = L_s + 0.35\,L_a + 0.05\,L_m + 0.02\,L_C + 0.01\,\mathrm{mean}(r^2)
```

where $L_s$ and $L_a$ are the mean squared errors on susceptibility and on depth at 5 years. The constraints are soft penalties, not hard constraints:

```math
L_m = \mathrm{mean}\left[\max(0,\,2-m)^2 + \max(0,\,m-6)^2\right]
```

```math
L_C = \mathrm{mean}\left[\max(0,\,-22-\ln C)^2 + \max(0,\,\ln C+14)^2\right]
```

The band for $\ln C$, with $C$ in mm/cycle, is an assumed scale sized for mm-scale growth. It is not a published Paris constant.

## Installation

```bash
pip install -r requirements.txt
```

Optional (for ML):

```bash
pip install torch
```

## Run

```bash
python synthetic_unit_piml_pinn.py
```

Without PyTorch the script skips the PINN demo. With PyTorch installed, add `--no-ml` to skip it.

Tests need PyTorch and pytest:

```bash
pip install pytest
pytest
```

## Outputs

Written next to the script:

- `unit_topology_anonymized.tsv`
- `equipment_anonymized.tsv`
- `piping_attributes_anonymized.tsv`
- `scc_alternatives_100.tsv`
- `synthetic_refinery_unit_3d.html`
- `pinn_checkpoint.pt` (PINN demo only)

## Limitations

- The PINN demo is an architecture demo, not a predictor. Its crack depth rollout loses to a constant baseline on held-out rows.
- The synthetic depth target does not depend on pressure or size, but $\Delta K$ does. The network has to cancel that dependence through $\ln C$, which is the likely cause of the held-out failure.
- $\ln C$ and $m$ are free for each row, so the Paris law fixes the shape of the growth curve, not its rate.
- The soft penalty does not hold $m$ in its band: 0 of 20 held-out rows have $m$ in $[2, 6]$ (at most 15% over 5 splits).
- Training and test data are synthetic. No result here is validated against inspection data.

## Future Extensions

- Real P&ID parser integration
- Crack growth time simulation
- Remaining life forecasting
- Monte Carlo reliability modeling
- Integration with Power BI dashboards
- Risk prioritization engine

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [integrity-code-series-week3-f1-lap-simulation](https://github.com/felipearocha/integrity-code-series-week3-f1-lap-simulation) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity-code-series-week7-h2-lferw](https://github.com/felipearocha/integrity-code-series-week7-h2-lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| Week 10 | [integrity-code-series-week10-nnph-scc](https://github.com/felipearocha/integrity-code-series-week10-nnph-scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

---

## How to Cite

If this software contributes to your work, please cite both the software (this repository) and the underlying methods it implements.

**Software (archived release):**

> Rocha, F. (2026). *Synthetic Integrity Digital Twin (PIML + PINN Framework) - Integrity Code Series Bonus* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20172483

**BibTeX:**

```bibtex
@software{rocha_2026_piml,
  author       = {Rocha, Felipe},
  title        = {{Synthetic Integrity Digital Twin (PIML + PINN Framework) - Integrity Code Series Bonus}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.20172483},
  url          = {https://doi.org/10.5281/zenodo.20172483}
}
```

The two DOIs Zenodo provides are:

| DOI                                  | What it points to                                                  |
|--------------------------------------|--------------------------------------------------------------------|
| `10.5281/zenodo.20172483` (concept)   | Always resolves to the latest version - use this for citation.     |
| `10.5281/zenodo.20172484` (version)   | Pinned to v1.0.1 specifically - use when reproducibility matters.  |

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff) - GitHub will display a "Cite this repository" widget at the top right of the repo page that exports BibTeX / APA / RIS automatically.

