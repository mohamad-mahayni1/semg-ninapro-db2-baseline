"""
Baseline: per-subject sEMG window features + cross-subject LDA on NinaPro DB2 Exercise 1.

Pipeline (matches REPORT.md sections 2-3):
  load .mat -> filter (4th-order Butterworth band-pass 20-450 Hz, notch 50 Hz)
            -> 200 ms windows, 100 ms hop -> per-window features (RMS, MAV, WL, ZC, SSC, log-var)
            -> per-channel z-score fit on TRAIN windows only -> LDA(lsqr, shrinkage='auto').

Splits:
  Train: subjects 1-10 (S1..S10)
  Test : subjects 11-15 (S11..S15)
  Subject-disjoint by construction. Restimulus is the label (Gijsberts 2014 relabelling).

Run: python src/baseline.py
Outputs: artifacts/metrics.json, artifacts/confusion_matrix.npz, artifacts/feat_s*.npz (cached)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

SR = 2000
WIN_MS = 200
HOP_MS = 100
WIN = SR * WIN_MS // 1000
HOP = SR * HOP_MS // 1000
TRAIN_SUBJ = list(range(1, 11))
TEST_SUBJ = [11, 12, 13, 14, 15]
RANDOM_SEED = 0


def _build_filters():
    nyq = SR / 2
    bp_b, bp_a = butter(4, [20 / nyq, 450 / nyq], btype="band")
    bp_sos = tf2sos(bp_b, bp_a)
    nt_b, nt_a = iirnotch(w0=50 / nyq, Q=30)
    nt_sos = tf2sos(nt_b, nt_a)
    return bp_sos, nt_sos


def load_subject(subj):
    p = DATA_DIR / f"DB2_s{subj}" / f"S{subj}_E1_A1.mat"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    m = sio.loadmat(str(p))
    emg = np.asarray(m["emg"], dtype=np.float32)
    rest = np.asarray(m["restimulus"], dtype=np.int8).ravel()
    rerep = np.asarray(m["rerepetition"], dtype=np.int8).ravel()
    assert emg.shape[1] == 12, f"expected 12 ch, got {emg.shape[1]}"
    return emg, rest, rerep


def filter_emg(emg, bp_sos, nt_sos):
    out = np.empty_like(emg)
    for c in range(emg.shape[1]):
        x = sosfiltfilt(bp_sos, emg[:, c])
        x = sosfiltfilt(nt_sos, x)
        out[:, c] = x.astype(np.float32)
    return out


def _zc(w, thr=1e-5):
    s = np.sign(w)
    s[np.abs(w) < thr] = 0
    return (np.diff(s, axis=0) != 0).sum(axis=0).astype(np.float32)


def _ssc(w, thr=1e-5):
    d = np.diff(w, axis=0)
    s = np.sign(d)
    s[np.abs(d) < thr] = 0
    return (np.diff(s, axis=0) != 0).sum(axis=0).astype(np.float32)


def features_for_window(w):
    rms = np.sqrt(np.mean(w * w, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    zc = _zc(w)
    ssc = _ssc(w)
    logvar = np.log(np.var(w, axis=0) + 1e-12)
    return np.concatenate([rms, mav, wl, zc, ssc, logvar]).astype(np.float32)


def window_subject(emg, rest, rerep):
    n = emg.shape[0]
    starts = np.arange(0, n - WIN + 1, HOP)
    feats = np.empty((len(starts), 72), dtype=np.float32)
    ys = np.empty(len(starts), dtype=np.int8)
    reps = np.empty(len(starts), dtype=np.int8)
    keep = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        e = s + WIN
        wlab = rest[s:e]
        wrep = rerep[s:e]
        if (wlab[0] != wlab).any() or (wrep[0] != wrep).any():
            continue
        keep[i] = True
        feats[i] = features_for_window(emg[s:e])
        ys[i] = wlab[0]
        reps[i] = wrep[0]
    return feats[keep], ys[keep], reps[keep]


def process_subject(subj, bp_sos, nt_sos):
    cache = ARTIFACT_DIR / f"feat_s{subj}.npz"
    if cache.exists():
        z = np.load(cache)
        return {"X": z["X"], "y": z["y"], "rep": z["rep"], "subj": subj}
    t0 = time.time()
    emg, rest, rerep = load_subject(subj)
    emg = filter_emg(emg, bp_sos, nt_sos)
    X, y, rep = window_subject(emg, rest, rerep)
    np.savez_compressed(cache, X=X, y=y, rep=rep)
    print(f"  S{subj}: {X.shape[0]} windows  ({time.time()-t0:.1f}s)", flush=True)
    return {"X": X, "y": y, "rep": rep, "subj": subj}


def main(train_subj=TRAIN_SUBJ, test_subj=TEST_SUBJ, drop_rest=False):
    t_start = time.time()
    bp_sos, nt_sos = _build_filters()

    print(f"[baseline] loading train subjects {train_subj}", flush=True)
    train_parts = [process_subject(s, bp_sos, nt_sos) for s in train_subj]
    print(f"[baseline] loading test subjects {test_subj}", flush=True)
    test_parts = [process_subject(s, bp_sos, nt_sos) for s in test_subj]

    Xtr = np.vstack([p["X"] for p in train_parts])
    ytr = np.concatenate([p["y"] for p in train_parts])
    Xte = np.vstack([p["X"] for p in test_parts])
    yte = np.concatenate([p["y"] for p in test_parts])

    if drop_rest:
        m = ytr != 0
        Xtr, ytr = Xtr[m], ytr[m]
        m = yte != 0
        Xte, yte = Xte[m], yte[m]

    print(f"[baseline] train X={Xtr.shape}, test X={Xte.shape}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    try:
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(Xtr_s, ytr)
        model_name = "LDA(lsqr, shrinkage=auto)"
    except Exception as e:
        print(f"[baseline] LDA failed ({e}); falling back to LogisticRegression", flush=True)
        clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_SEED)
        clf.fit(Xtr_s, ytr)
        model_name = "LogisticRegression(max_iter=1000)"

    yp = clf.predict(Xte_s)
    macro = f1_score(yte, yp, average="macro")
    weighted = f1_score(yte, yp, average="weighted")
    acc = float((yp == yte).mean())
    classes = sorted(set(ytr.tolist()))
    n_classes = len(classes)
    cm = confusion_matrix(yte, yp, labels=classes)

    # Per-subject test macro-F1 (variance across held-out users)
    per_subject = {}
    offset = 0
    for p in test_parts:
        n = p["X"].shape[0]
        if drop_rest:
            mask = p["y"] != 0
            n_kept = int(mask.sum())
            sl_y = p["y"][mask]
            sl_p = yp[offset : offset + n_kept]
            offset += n_kept
        else:
            sl_y = p["y"]
            sl_p = yp[offset : offset + n]
            offset += n
        if len(sl_y) > 0:
            per_subject[f"S{p['subj']}"] = {
                "n_windows": int(len(sl_y)),
                "macro_f1": round(float(f1_score(sl_y, sl_p, average="macro")), 4),
                "accuracy": round(float((sl_p == sl_y).mean()), 4),
            }

    # Bootstrap 95% CI on macro-F1 across test windows (1000 resamples)
    rng = np.random.default_rng(RANDOM_SEED)
    n_test = len(yte)
    boot = np.empty(1000, dtype=np.float64)
    for i in range(1000):
        idx = rng.integers(0, n_test, size=n_test)
        boot[i] = f1_score(yte[idx], yp[idx], average="macro")
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    # Per-class window counts (class-imbalance evidence for the dataset assessment)
    train_class_counts = {int(c): int((ytr == c).sum()) for c in classes}
    test_class_counts = {int(c): int((yte == c).sum()) for c in classes}

    summary = {
        "task": "DB2 Exercise 1 cross-subject classification",
        "model": model_name,
        "train_subjects": train_subj,
        "test_subjects": test_subj,
        "drop_rest": drop_rest,
        "n_classes": n_classes,
        "n_train_windows": int(Xtr.shape[0]),
        "n_test_windows": int(Xte.shape[0]),
        "feature_dim": int(Xtr.shape[1]),
        "macro_f1": float(macro),
        "macro_f1_bootstrap_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "weighted_f1": float(weighted),
        "accuracy": acc,
        "per_subject_test": per_subject,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "elapsed_sec": time.time() - t_start,
        "headline": (
            f"NinaPro DB2 Exercise 1, train S{train_subj[0]}-S{train_subj[-1]} "
            f"/ test S{test_subj[0]}-S{test_subj[-1]} (subject-disjoint), "
            f"N_classes={n_classes}, drop_rest={drop_rest}, "
            f"macro-F1={macro:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}]). "
            "Single-fit cross-subject baseline; performance is NOT the primary criterion."
        ),
    }
    suffix = "_norest" if drop_rest else "_withrest"
    (ARTIFACT_DIR / f"metrics{suffix}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(ARTIFACT_DIR / f"confusion_matrix{suffix}.npz", cm=cm, labels=np.array(classes))
    print(f"\n[baseline] {summary['headline']}", flush=True)
    print(f"[baseline] elapsed {summary['elapsed_sec']:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    s_with = main(drop_rest=False)
    s_no = main(drop_rest=True)
    combined = {"with_rest": s_with, "without_rest": s_no}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(combined, indent=2))
