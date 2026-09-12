"""
Supporting experiments: electrode-count ablation and feature-block ablation.
Reuses cached per-subject features from src/baseline.py (artifacts/feat_s*.npz).

Electrode ablation answers design question Q1 ("how many electrodes?") with concrete numbers
under the same subject-disjoint split.

Feature ablation answers design question Q2 ("which features are most relevant?") by removing
each of the 6 time-domain feature blocks and measuring the macro-F1 drop.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts"

TRAIN_SUBJ = list(range(1, 11))
TEST_SUBJ = [11, 12, 13, 14, 15]
N_CH = 12

FEATURE_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC", "logVar"]


def _columns_drop_blocks(drop_blocks: list[int]) -> np.ndarray:
    """Drop one or more feature blocks (each block = N_CH columns)."""
    keep = [b for b in range(6) if b not in drop_blocks]
    cols = []
    for b in keep:
        for c in range(N_CH):
            cols.append(b * N_CH + c)
    return np.array(cols, dtype=np.int64)


def _load(subj: int):
    z = np.load(ARTIFACT_DIR / f"feat_s{subj}.npz")
    return z["X"], z["y"], z["rep"]


def _stack(subjs):
    Xs, ys = [], []
    for s in subjs:
        X, y, _ = _load(s)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Xtr_s, ytr)
    yp = clf.predict(Xte_s)
    return float(f1_score(yte, yp, average="macro")), float((yp == yte).mean())


def _feature_columns_for_channels(channels: list[int]) -> np.ndarray:
    """Feature vector layout (per features_for_window): [RMS_ch1..12, MAV_ch1..12, ...]
    For 6 feature blocks * 12 channels = 72 dims. Pick column indices for given chs."""
    cols = []
    for block in range(6):
        for c in channels:
            cols.append(block * N_CH + c)
    return np.array(cols, dtype=np.int64)


def _columns_drop_feature(drop_idx: int) -> np.ndarray:
    """Drop one of the 6 feature blocks (60 dims kept)."""
    return _columns_drop_blocks([drop_idx])


def combo_ablation():
    """Confirm the leave-one-out finding: drop {SSC, MAV} jointly should still beat baseline."""
    Xtr, ytr = _stack(TRAIN_SUBJ)
    Xte, yte = _stack(TEST_SUBJ)
    combos = [
        ("baseline (all 6)", []),
        ("drop {SSC}", [4]),
        ("drop {MAV}", [1]),
        ("drop {SSC, MAV} (lean 4-feat)", [4, 1]),
    ]
    rows = []
    for name, blocks in combos:
        cols = _columns_drop_blocks(blocks) if blocks else np.arange(6 * N_CH)
        macro, _ = _eval(Xtr[:, cols], ytr, Xte[:, cols], yte)
        rows.append({
            "config": name,
            "dropped": [FEATURE_NAMES[b] for b in blocks],
            "n_features": int(len(cols)),
            "macro_f1": round(macro, 4),
        })
        print(f"  {name:35s}  n={len(cols):3d}  macroF1={macro:.4f}", flush=True)
    return rows


def electrode_ablation():
    Xtr, ytr = _stack(TRAIN_SUBJ)
    Xte, yte = _stack(TEST_SUBJ)
    configs = [
        ("all 12", list(range(12))),
        ("array8 (ch 1-8 spaced)", list(range(0, 8))),
        ("targeted4 (ext/flex/biceps/triceps = ch 9-12)", [8, 9, 10, 11]),
        ("array4 (ch 1,3,5,7 spaced)", [0, 2, 4, 6]),
        ("targeted2 (flex/ext digitorum = ch 9,10)", [8, 9]),
    ]
    rows = []
    for name, chs in configs:
        cols = _feature_columns_for_channels(chs)
        macro, acc = _eval(Xtr[:, cols], ytr, Xte[:, cols], yte)
        rows.append({"config": name, "n_electrodes": len(chs), "n_features": len(cols), "macro_f1": round(macro, 4), "accuracy": round(acc, 4)})
        print(f"  {name:50s}  n_ch={len(chs):2d}  macroF1={macro:.3f}  acc={acc:.3f}", flush=True)
    return rows


def feature_ablation():
    Xtr, ytr = _stack(TRAIN_SUBJ)
    Xte, yte = _stack(TEST_SUBJ)
    macro_full, _ = _eval(Xtr, ytr, Xte, yte)
    print(f"  full (6 features)                               macroF1={macro_full:.3f}", flush=True)
    rows = [{"dropped": "(none)", "n_features": 72, "macro_f1": round(macro_full, 4)}]
    for i, name in enumerate(FEATURE_NAMES):
        cols = _columns_drop_feature(i)
        macro, _ = _eval(Xtr[:, cols], ytr, Xte[:, cols], yte)
        delta = macro - macro_full
        rows.append({"dropped": name, "n_features": int(len(cols)), "macro_f1": round(macro, 4), "delta_vs_full": round(delta, 4)})
        print(f"  drop {name:8s}                                       macroF1={macro:.3f}  d={delta:+.4f}", flush=True)
    return rows


def main():
    t0 = time.time()
    print("[ablation] electrode ablation (subject-disjoint, with rest)", flush=True)
    e_rows = electrode_ablation()
    print("[ablation] feature ablation (drop one TD feature block at a time)", flush=True)
    f_rows = feature_ablation()
    print("[ablation] combo ablation (drop {SSC,MAV} jointly)", flush=True)
    c_rows = combo_ablation()
    out = {"electrode_ablation": e_rows, "feature_ablation": f_rows, "combo_ablation": c_rows, "elapsed_sec": time.time() - t0}
    (ARTIFACT_DIR / "ablation.json").write_text(json.dumps(out, indent=2))
    print(f"[ablation] wrote {ARTIFACT_DIR / 'ablation.json'} in {out['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
