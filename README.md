# NinaPro DB2 — cross-subject sEMG gesture recognition

An honest classical baseline for hand-gesture classification from surface EMG, built to answer a
question I keep running into in wearable and implantable biosignal work: **how much of the
published performance survives when the model meets a user it has never seen?**

Mohamad Mahayni · report in [`REPORT.md`](REPORT.md) · code in [`src/`](src)

## Headline

On **NinaPro DB2 Exercise 1** with a **subject-disjoint** split (train S1–S10, test S11–S15) and a
classical pipeline (200 ms windows, 6 time-domain features × 12 channels, shrinkage-LDA):

| Setting | Classes | macro-F1 (bootstrap 95 % CI) |
|---|---|---|
| With rest | 18 | **0.232** [0.227, 0.237] |
| Without rest | 17 | **0.212** [0.207, 0.217] |

Per-subject test macro-F1 spans **0.11 (S11) to 0.34 (S15)** — a 3× spread on one fixed model.
**Calibration to the individual user matters more than the choice of model.** That result, not the
absolute number, is the point of the repository.

## Why it is built this way

Peak accuracy is not the goal. Three things are:

1. **Every choice is stated and defensible** — window length, feature set, label source, and why a
   shrinkage-LDA is the right yardstick before anything deeper is allowed to claim value.
2. **Leakage is designed out** — the subject split happens *before* windowing, so no test-subject
   window ever reaches `fit()`, and windows with non-constant labels are dropped.
3. **The numbers are connected to system design** — electrode count, latency budget, compute cost
   on a microcontroller-class target, calibration time, and drift monitoring.

## Ablations

Two design questions I set myself before trusting any model, answered with numbers rather than opinion:

- **How many electrodes are needed?** 8 spaced electrodes retain **92 %** of full performance at
  two thirds of the channel count; layout beats anatomical targeting at low channel counts
  (spaced-4 > targeted-4 by +0.04 F1); below 4 channels it collapses.
- **Which features can be dropped?** With only five test subjects, **none reliably** — every
  leave-one-block-out drop lands inside the stability band. The honest answer is "not enough
  subjects to prune yet", and the repository says so instead of picking a winner.

## Layout

```
REPORT.md          the analysis (dataset critique, method, experiments, system implications)
notebook.ipynb     light walkthrough; calls into src/
src/baseline.py    end-to-end pipeline: load -> filter -> window -> features -> LDA
src/ablation.py    electrode-count and feature-block ablations
artifacts/         metrics and the confusion-matrix figure (feature caches are gitignored)
systemc/           SystemC models of the 18 x 72 inference block: float, fixed point, 1-8 parallel MAC units
data/              NinaPro DB2 .mat files — not redistributed, see data/README.md
```

## Reproduce

Python 3.12. Download the data first (see [`data/README.md`](data/README.md)), then:

```bash
pip install -r requirements.txt
python src/baseline.py
python src/ablation.py
```

First run extracts features for every subject (≈5–8 min on a laptop CPU) and caches them; later
runs finish in seconds.

## Hardware view

`systemc/` models the classifier as a dedicated block: 1,296 cycles per decision at one
multiply-accumulate per cycle, 324 with four fixed-point MAC units in parallel. Weight
fetches stay at 1,296 in every variant, so parallelism buys latency, not energy. A Cortex-M4
spends roughly the one-unit cycle count in software, so the case for a block is energy per
operation and core sleep time, not cycles — and the classifier is the cheapest stage of the
chain; filtering and features run at the sample rate. See [`systemc/README.md`](systemc/README.md).

## Limits I am not hiding

- Five test subjects. The confidence intervals are window-level, so they are a stability check, not
  a significance test.
- Single train/test fit, not cross-validated.
- DB2 subjects are intact-limbed, recorded in one cued session by an operator who placed the
  electrodes. Published DB2 numbers are an **upper bound** on a real first fit by a real user.

## Data and licence

NinaPro DB2 (Atzori et al., *Scientific Data*, 2014) is **not redistributed here** — see
[`data/README.md`](data/README.md) for how to obtain it under its own terms. The code in this
repository is MIT-licensed.
