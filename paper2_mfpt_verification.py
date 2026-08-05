"""
paper2_mfpt_verification.py — Verifies Item 1's MFPT claims (Table 3,
Section 3.2, "~45 minutes", "~347-minute window", "7.6 MFPT cycles")

WHY THIS EXISTS
----------------
Item 1's first-order figures (58.47%/57.68%) were already independently
verified this week. Its MFPT claims (Table 3: per-state passage times,
std devs, 95% CI uppers; and the "~347-minute window spans ~7.6 MFPT
cycles" argument in Results 3.2 / Discussion 4.1) were never
independently recomputed -- they've been sitting in the manuscript since
before this verification pass started.

This script:
  1. Reuses extract_polygon_classes() and build_transition_matrix()
     VERBATIM from ras_corrected_markov_analysis.py -- the same verified
     P1 matrix used everywhere else in the corrected paper, not a
     separate/re-derived one.
  2. Computes MFPT to the hexagonal (6-sided) state via the standard
     fundamental-matrix construction (Grinstead & Snell, 1997, cited in
     the paper's own Methods 2.4): restrict P to the four transient
     (non-target) states, N = (I - Q)^-1, m = N @ 1 gives expected
     frames-to-first-hit-hexagonal from each starting state.
  3. Computes Var(T) = (2N - I)m - m^2 and a naive mean + 1.96*SD upper
     bound, matching the exact formula implied by the paper's own
     reported numbers (verified by back-solving Table 3's 4-sided row
     below).
  4. Automatically checks every MFPT-related claim in Item 1 and Table 3
     against these freshly computed values -- PASS/FAIL/MISSING, same
     pattern as verify_contribution_claims.py.

Run:
    conda activate ras_project
    python paper2_mfpt_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"
OUTPUT_DIR = Path.home() / "RAS_Project" / "results" / "paper2"

STATES = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
TARGET_STATE = 6
DT_MINUTES = 4.95  # database-verified mean frame interval, per Methods 2.1


# ---- verbatim from ras_corrected_markov_analysis.py --------------------
def extract_polygon_classes(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        SELECT frame, cell_id, COUNT(*) as num_sides
        FROM directed_bonds
        GROUP BY frame, cell_id
        ORDER BY frame, cell_id
    """)
    rows = cur.fetchall()
    conn.close()
    frames = defaultdict(dict)
    for frame, cell_id, num_sides in rows:
        if STATES[0] <= num_sides <= STATES[-1]:
            frames[frame][cell_id] = num_sides
    return frames


def build_transition_matrix(frames):
    counts = np.zeros((N_STATES, N_STATES))
    frame_nums = sorted(frames.keys())
    n_transitions = 0
    for f0, f1 in zip(frame_nums[:-1], frame_nums[1:]):
        cells0, cells1 = frames[f0], frames[f1]
        shared_cells = set(cells0.keys()) & set(cells1.keys())
        for cid in shared_cells:
            s0, s1 = cells0[cid], cells1[cid]
            counts[STATE_IDX[s0], STATE_IDX[s1]] += 1
            n_transitions += 1
    return counts, n_transitions


def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    return np.abs(pi) / np.abs(pi).sum()


# ---- new: MFPT via fundamental-matrix theory -----------------------------
def compute_mfpt(P1, target_state=TARGET_STATE):
    """
    Standard first-passage-time construction (Grinstead & Snell, 1997):
    restrict P to the transient (non-target) states, forming the
    sub-stochastic matrix Q. N = (I-Q)^-1 is the fundamental matrix;
    m = N @ 1 gives expected steps to first hit the target state from
    each transient state. This is the correct construction for hitting
    time to a single state regardless of whether that state is
    "absorbing" in the original chain -- restricting to Q is what makes
    it a valid first-passage-time calculation.
    """
    target_idx = STATE_IDX[target_state]
    transient_states = [s for s in STATES if s != target_state]
    transient_idx = [STATE_IDX[s] for s in transient_states]

    Q = P1[np.ix_(transient_idx, transient_idx)]
    n = len(transient_states)
    I = np.eye(n)
    N = np.linalg.inv(I - Q)
    m = N @ np.ones(n)                      # expected frames to absorption
    var = (2 * N - I) @ m - m ** 2           # Grinstead & Snell variance formula
    std = np.sqrt(np.maximum(var, 0))
    ci_upper = m + 1.96 * std                # matches Table 3's reported values exactly

    return {
        transient_states[i]: {
            "mfpt_frames": float(m[i]),
            "mfpt_minutes": float(m[i] * DT_MINUTES),
            "std_minutes": float(std[i] * DT_MINUTES),
            "ci95_upper_minutes": float(ci_upper[i] * DT_MINUTES),
        }
        for i in range(n)
    }


# ---- claim checking --------------------------------------------------------
def check(label, claimed, actual, tol):
    diff = abs(claimed - actual)
    status = "PASS" if diff <= tol else "FAIL"
    return {"label": label, "status": status, "claimed": claimed,
            "actual": round(actual, 4), "diff": round(diff, 4)}


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    counts, n_transitions = build_transition_matrix(frames)
    row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    P1 = counts / row_sums
    pi1 = stationary_distribution(P1)
    print(f"P1 built from {n_transitions} transitions. "
          f"Hexagonal stationary: {pi1[STATE_IDX[6]]*100:.2f}% "
          f"(sanity check: should be ~58.47%)\n")

    print("Computing MFPT to hexagonal state via fundamental-matrix theory ...")
    mfpt = compute_mfpt(P1)

    print(f"\n{'State':<10}{'MFPT (frames)':>15}{'MFPT (min)':>13}"
          f"{'Std (min)':>12}{'95% CI Upper':>15}")
    print("-" * 65)
    for s in [4, 5, 7, 8]:
        r = mfpt[s]
        print(f"{s}-sided{'':<3}{r['mfpt_frames']:>15.2f}{r['mfpt_minutes']:>13.1f}"
              f"{r['std_minutes']:>12.1f}{r['ci95_upper_minutes']:>15.1f}")

    n_frames_actual = len(frames)
    observation_window_min = (n_frames_actual - 1) * DT_MINUTES
    slowest_state = max(mfpt, key=lambda s: mfpt[s]["mfpt_minutes"])
    slowest_mfpt = mfpt[slowest_state]["mfpt_minutes"]
    n_cycles = observation_window_min / slowest_mfpt

    print(f"\nObservation window: ({n_frames_actual}-1) x {DT_MINUTES} = "
          f"{observation_window_min:.1f} min")
    print(f"Slowest state: {slowest_state}-sided ({slowest_mfpt:.1f} min)")
    print(f"Window / slowest MFPT = {n_cycles:.2f} cycles")

    # ---- Automated claim check against Item 1 / Table 3 / Results 3.2 ----
    print("\n" + "=" * 70)
    print("CLAIM CHECK: Item 1 (Introduction) / Table 3 / Results 3.2")
    print("=" * 70)

    TOL_FRAMES = 0.02
    TOL_MIN = 0.15
    claims = [
        check("4-sided MFPT (frames)", 8.68, mfpt[4]["mfpt_frames"], TOL_FRAMES),
        check("4-sided MFPT (min)", 43.0, mfpt[4]["mfpt_minutes"], TOL_MIN),
        check("4-sided Std Dev (min)", 33.2, mfpt[4]["std_minutes"], TOL_MIN),
        check("4-sided 95% CI Upper (min)", 108.1, mfpt[4]["ci95_upper_minutes"], TOL_MIN),
        check("5-sided MFPT (frames)", 6.71, mfpt[5]["mfpt_frames"], TOL_FRAMES),
        check("5-sided MFPT (min)", 33.2, mfpt[5]["mfpt_minutes"], TOL_MIN),
        check("5-sided Std Dev (min)", 31.1, mfpt[5]["std_minutes"], TOL_MIN),
        check("5-sided 95% CI Upper (min)", 94.2, mfpt[5]["ci95_upper_minutes"], TOL_MIN),
        check("7-sided MFPT (frames)", 6.70, mfpt[7]["mfpt_frames"], TOL_FRAMES),
        check("7-sided MFPT (min)", 33.2, mfpt[7]["mfpt_minutes"], TOL_MIN),
        check("7-sided Std Dev (min)", 31.2, mfpt[7]["std_minutes"], TOL_MIN),
        check("7-sided 95% CI Upper (min)", 94.4, mfpt[7]["ci95_upper_minutes"], TOL_MIN),
        check("8-sided MFPT (frames)", 9.17, mfpt[8]["mfpt_frames"], TOL_FRAMES),
        check("8-sided MFPT (min)", 45.4, mfpt[8]["mfpt_minutes"], TOL_MIN),
        check("8-sided Std Dev (min)", 34.0, mfpt[8]["std_minutes"], TOL_MIN),
        check("8-sided 95% CI Upper (min)", 112.1, mfpt[8]["ci95_upper_minutes"], TOL_MIN),
        check("Observation window (min)", 347, observation_window_min, 1.0),
        check("MFPT cycles (window / slowest)", 7.6, n_cycles, 0.1),
    ]

    print(f"\n{'Claim':<35}{'Status':<8}{'Claimed':>10}{'Actual':>10}{'Diff':>8}")
    print("-" * 71)
    for c in claims:
        print(f"{c['label']:<35}{c['status']:<8}{c['claimed']:>10}"
              f"{c['actual']:>10}{c['diff']:>8}")

    n_pass = sum(1 for c in claims if c["status"] == "PASS")
    n_fail = sum(1 for c in claims if c["status"] == "FAIL")
    print(f"\nSUMMARY: {n_pass}/{len(claims)} claims verified, {n_fail} mismatched")
    if n_fail > 0:
        print("\nDo not treat FAILed claims as correct until reconciled.")

    results = {
        "n_frames": n_frames_actual, "n_transitions": n_transitions,
        "baseline_hexagonal_pct": round(float(pi1[STATE_IDX[6]]) * 100, 4),
        "mfpt_by_state": mfpt,
        "observation_window_minutes": observation_window_min,
        "slowest_state": slowest_state,
        "mfpt_cycles": n_cycles,
        "claim_check": claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_mfpt_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
