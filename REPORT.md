# NinaPro DB2 — Cross-Subject sEMG Gesture Recognition

**Mohamad Mahayni** · code: `src/baseline.py`, `src/ablation.py`, `notebook.ipynb`

> **Headline.** On NinaPro DB2 Exercise 1 with a subject-disjoint split (train S1–S10, test S11–S15) and a classical pipeline (200 ms windows, 6 time-domain features × 12 channels, shrinkage-LDA), I report **macro-F1 = 0.232 (bootstrap-95 % CI [0.227, 0.237]) over 18 classes with rest** and **macro-F1 = 0.212 [0.207, 0.217] over 17 classes without rest**, single train/test fit. Per-subject test macro-F1 spreads from **0.11 (S11) to 0.34 (S15)** — calibration to the user matters more than the model. Performance is *not* the primary deliverable; the goal is to make every choice honest, surface limitations, and connect numbers to system design.

## 1. Dataset assessment

DB2: 12-channel sEMG @ 2 kHz (Delsys Trigno), 36-ch accel, 22-ch CyberGlove kinematics, force (E3); 40 intact subjects, 6 reps × 49 movements. Labels: `stimulus`/`repetition` (cue) and `restimulus`/`rerepetition` (Gijsberts 2014 onset-relabelled — what I use). E1 = 17 movements + rest = 18 classes.

**Suitability — partial.** Right *kind* of data (wearable EMG, multi-class, multi-subject), good *algorithm* benchmark; **not** a deployment validation set. Subjects are intact (users will be amputee), recording is short (no sweat/impedance drift), cadence is artificial (5 s on / 3 s rest, cued), operator places electrodes (a user does not). Treat published DB2 numbers as **upper bounds** on a real first-fit. Demographic skew: 33/40 male, ages 23–45, neuro-typical. **Class imbalance:** rest = 50.9 % of train windows; the rarest movement class is ~1 800 windows, the most common ~3 600 — a 2× spread *before* you add rest.

**Missing for deployment.** Amputee cohort; within- and across-day sessions; explicit electrode-shift trials; user-controlled donning; continuous un-cued transitions; ADL background activity (typing, holding a cup); device-grade hardware.

**Protocol improvements I would push for.** (i) **Re-don block** per session — directly measures calibration burden, single highest-value addition. (ii) **Continuous un-cued "free movement"** block — required to even pose Q3 below. (iii) **Second session ≥24 h later** for drift. (iv) Per-subject placement photos so the "operator placed exactly" assumption is testable. (v) Even five amputee subjects would change every cross-cohort conclusion.

## 2. Data capture and preprocessing

**Acquisition.** 12 Trigno electrodes: 8 spaced around the radio-humeral joint (array prior), 2 on flex/ext digitorum, 2 on biceps/triceps. EMG @ 2 kHz. **Synchronization:** DB2 ships EMG, acc, glove, labels on the same sample grid; I trust the acquisition software's alignment. **Segmentation:** cued stimulus boundaries are coarse; `restimulus` is post-hoc onset/offset relabelled — what I use.

**Pipeline.** (1) Load `.mat`; assert 12 channels at `sr=2000`; lengths and label ranges checked across all 15 available subjects. (2) Zero-phase Butterworth 20–450 Hz band-pass + 50 Hz notch per channel. (3) **200 ms windows / 100 ms hop** (Englehart & Hudgins) — shorter windows reduce latency but variance of RMS/MAV grows ∝ 1/N, longer windows stabilise features at a perceived lag; 200 ms is the prosthetic-control sweet spot. (4) **Drop impure windows** whose `restimulus`/`rerepetition` is not constant — a leakage trap. (5) Features: RMS, MAV, WL, ZC, SSC, log-var per channel → 72-dim. (6) `StandardScaler` fit on **train subjects only**.

**Implicit assumptions.** (a) cue ≈ action (`restimulus` only partially fixes onset variance); (b) electrodes placed identically across subjects; (c) labels are clean; (d) inter-movement transitions are uninformative (3 s rest); (e) all gestures equally important — they are not.

**Real-wearable improvements (pipeline side).** Per-session RMS-threshold idle detector; MNF/MDF spectral pair for fatigue-induced compression TD misses; impedance check at session start to abort on a dry lead. **Leakage:** subject split is applied *before* windowing — no test-subject window reaches `fit()`.

## 3. Methods

**3A — Classical baseline (implemented).** Shrinkage-LDA (`solver='lsqr', shrinkage='auto'`) on the 72-dim TD vector. **Why LDA:** closed-form (no seed lottery), interpretable per-channel, microsecond inference, the de-facto EMG benchmark for a decade. If a complex model cannot beat shrinkage-LDA on this exact pipeline, the complexity is not earning its keep.

**3B — Modern alternative (proposed).** Small **1-D CNN over raw windows** (3 conv blocks, kernel 5–9, 32→64→128 ch, GAP + linear head, ~50–100 k params, INT8 deployable). TD features discard cross-channel phase; a CNN learns spatial-temporal templates directly. **Why CNN over LSTM/transformer:** at 200 ms windows context is short, so an LSTM's recurrence buys little while costing serial latency on edge silicon; a transformer is over-parameterised for ~400 samples and worse at INT8 latency than a stride-conv stack. **Trade-offs:** *perf* +Δ within-subject, marginal cross-subject without calibration; *compute* +1–3 ms on Cortex-M @ INT8, +~200 KB; *robustness* unchanged across subjects without calibration or domain-adversarial training (Ganin 2015). The 0.11–0.34 per-subject spread below shows calibration is a bigger lever than architecture.

## 4. Experiments and product decisions

| Setting | N | Train wins | Test wins | macro-F1 (95 % CI) | weighted-F1 | acc |
|---|---|---|---|---|---|---|
| E1, **with rest** | 18 | 85 973 | 43 016 | **0.232** [0.227, 0.237] | 0.441 | 0.506 |
| E1, **without rest** | 17 | 42 238 | 23 172 | **0.212** [0.207, 0.217] | 0.205 | 0.219 |

Per-test-subject macro-F1: S11 0.112, S12 0.261, S13 0.191, S14 0.299, S15 0.343 — **a 3× spread on the same model**. The deployment-relevant quantity is the *worst-case user*, not the population mean. Macro-F1 stays roughly flat when rest is removed but accuracy drops 0.51 → 0.22 — single-number reporting hides this.

**Reformulating the three product questions into testable ones** *(ablation results are real)*:

**Q1. "How many electrodes are required?"** → *"Smallest electrode set keeping subject-disjoint macro-F1 within Δ of the 12-channel baseline; does layout (spaced array vs anatomical targeting) matter at fixed N?"* Variables: N ∈ {12, 8, 4, 2}; layout ∈ {spaced, targeted}. Same split as headline.

| Config | N_ch | macro-F1 | Δ vs 12 | retained |
|---|---|---|---|---|
| All 12 (baseline) | 12 | 0.232 | — | 100 % |
| Spaced 8 (radio-humeral array) | 8 | 0.214 | −0.018 | **92 %** |
| Spaced 4 (every-other array) | 4 | 0.159 | −0.073 | 69 % |
| Targeted 4 (digitorum + bi/triceps) | 4 | 0.118 | −0.114 | 51 % |
| Targeted 2 (flex/ext digitorum) | 2 | 0.078 | −0.154 | 34 % |

**Answer:** 8 spaced electrodes retain **92 %** of full performance at 67 % of the BOM and donning time; **layout dominates targeting at low N** (spaced-4 beats targeted-4 by +0.04 F1). Below 4 collapses.

**Q2. "Which features are most relevant?"** → leave-one-feature-block-out ablation on cached features.

| Drop | macro-F1 | Δ |
|---|---|---|
| (none, full 6) | 0.232 | — |
| WL | 0.220 | −0.012 |
| ZC | 0.223 | −0.009 |
| logVar | 0.224 | −0.008 |
| RMS | 0.230 | −0.002 |
| MAV | 0.230 | −0.002 |
| SSC | 0.231 | −0.001 |

**Answer (revised on N_test = 5):** **No single TD block looks reliably redundant** — every drop hurts, and the smallest drops (RMS, MAV, SSC) are within the window-bootstrap stability band of the baseline (±0.006). The CI is window-level (overlapping windows on the same five subjects), not subject-level — a stability check, not a formal significance test, but enough to show these deltas are not robust. *Honest conclusion:* with 5 test subjects we cannot recommend a 4-feature reduction (combo {drop SSC, MAV} = 0.228, Δ −0.003, inside noise); a defensible pruning needs ≥10 test subjects with subject-level CV.

**Q3. "How should continuous gestures be handled?"** → *"On un-cued reach-grasp-release sequences, what is the trade-off between window/hop, abstain-on-low-confidence, and post-processing on **gesture-level edit distance** vs latency-to-detection?"* Variables: window ∈ {100, 200, 400 ms}, hop ∈ {50, 100, 200 ms}, decision ∈ {argmax, argmax-with-abstain (posterior < τ)}, post-proc ∈ {none, majority-vote-K, HMM, CTC}.

**DB2 cannot answer this directly**, but a useful **proxy is feasible today:** concatenate the same subject's 5-s windows for movement-A → B → C (with 3 s rest unchanged), apply the live decoder to the synthetic stream, measure (a) **transition-detection latency** (frames between true onset and first correct prediction in the new class), (b) **edit distance** between predicted and ground-truth label sequence, (c) **spurious-switch rate** during sustained holds. This isolates the post-processing/stability question (which DB2 *can* test) from un-cued-onset (which it cannot). The §1 protocol additions are still needed for the real answer.

**System implications.** *Latency:* end-to-end ≈250 ms (200 ms window + filter + features <1 ms + LDA µs) — fits the prosthetic-control budget. *Compute:* shrinkage-LDA is one (18 × 72) matmul — trivial on Cortex-M. Q1 ⇒ 8 spaced electrodes cuts BOM ~⅓ for 8 % F1 cost; Q2 ⇒ **don't prune features yet — collect more subjects first**. *Calibration:* 0.23 is first-fit on a new user with a 0.11–0.34 spread; closing to within-subject SOA needs 1–5 min of cued calibration — the question is *how long is acceptable*, not whether. *Drift:* monitor rest-period RMS distribution; re-fit the LDA head when it trips. *Failure modes:* wrong-direction predictions mid-grasp are dangerous → abstain-on-low-confidence + per-class confusion review are cheap safeguards.

## Reproducibility

Python 3.12.10; pinned `requirements.txt`; explicit subject lists (train S1–S10, test S11–S15, no shuffling); `restimulus`/`rerepetition` are the labels; bootstrap CI uses 1 000 resamples, seed 0. Run: `pip install -r requirements.txt && python src/baseline.py && python src/ablation.py` → `artifacts/{metrics,ablation}.json` + `confusion_matrix*.npz`; the notebook renders `confusion_matrix.png`. ≈12 s baseline + 4 s ablations on a laptop CPU after one-time feature extraction; per-subject features cached under `artifacts/feat_s*.npz`.

*Refs:* Atzori 2014 Sci. Data; Gijsberts 2014; Englehart & Hudgins 2003; Atzori 2016; Geng 2016; Ganin 2015.
