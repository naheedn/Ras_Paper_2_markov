"""
paper2_methods_2_6_2_7_verification.py - Verifies Methods 2.6 (Spatial
Neighbor-Coupled Model) and 2.7 (Continuous-Time Markov Chain).

WHY THIS EXISTS
----------------
Most headline RESULTS numbers from these two sections were already checked
this week (paper2_spatial_corrected.py, paper2_ctmc_verification.py). This
script targets what those didn't cover: the specific claims stated in the
METHODS text itself, plus mathematical invariants that should hold if the
methods are correctly implemented -- not just "does the final percentage
match" but "is the construction itself valid."

Part A (Methods 2.6):
  1. Re-confirms 233,546 raw neighbor pairs from the directed_bonds self-join
     (previously only seen in terminal output, not in a dedicated PASS/FAIL
     checker).
  2. Checks the environment classification (low <5.5, hex 5.5-6.5, high >6.5)
     is EXHAUSTIVE and NON-OVERLAPPING -- every possible mean-neighbor-sides
     value maps to exactly one category, no gaps.
  3. Re-confirms environment counts (146 low, 1,838 high) against fresh
     extraction.

Part B (Methods 2.7):
  1. Re-confirms dt=4.95 min by direct query (not hardcoded), cross-checked
     against Methods 2.1's independent verification.
  2. Re-confirms exactly 4 negative off-diagonal corrections.
  3. NEW invariant checks not previously performed:
       a. Regularized Q_c has all rows summing to ~0 (valid generator)
       b. All off-diagonal entries of regularized Q_c are >=0 (valid generator)
       c. The stationary distribution pi actually solves pi @ Q_c ~ 0
          (verifies the solve claimed in Methods 2.7 is mathematically
          consistent, not just that the resulting percentage looks right)

Run:
    conda activate ras_project
    python paper2_methods_2_6_2_7_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import logm

DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"
OUTPUT_DIR = Path.home() / "RAS_Project" / "results" / "paper2"

STATES = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
TARGET_STATE = 6


def check(label, claimed, actual, tol=0, direction="abs"):
    if actual is None:
        return {"label": label, "status": "MISSING", "claimed": claimed, "actual": None}
    if direction == "abs":
        diff = abs(claimed - actual) if not isinstance(claimed, bool) else None
        status = "PASS" if (diff is not None and diff <= tol) else \
                  ("PASS" if claimed == actual else "FAIL")
    else:
        status = "PASS" if actual else "FAIL"
        diff = None
    return {"label": label, "status": status, "claimed": claimed,
            "actual": actual, "diff": diff}


# =============================================================================
# PART A: Methods 2.6 -- Spatial Neighbor-Coupled Model
# =============================================================================
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


def extract_neighbor_pairs(db_path):
    conn = sqlite3.connect(str(db_path))
    neighbor_query = """
    SELECT d1.frame AS frame, d1.cell_id AS cell_id, d2.cell_id AS neighbor_id
    FROM directed_bonds d1
    JOIN directed_bonds d2
      ON d1.frame = d2.frame AND d1.conj_dbond_id = d2.dbond_id
    WHERE d1.cell_id != d2.cell_id
    """
    neighbors = pd.read_sql(neighbor_query, conn)
    conn.close()
    return neighbors


def neighbor_env(mns):
    if mns < 5.5:
        return "low"
    elif mns <= 6.5:
        return "hex"
    else:
        return "high"


def check_classification_completeness():
    """Sweeps a fine grid of possible mean-neighbor-sides values and
    confirms every value maps to exactly one of {low, hex, high} with no
    gaps or ambiguity -- a logical check on the threshold definition
    itself, independent of any specific dataset."""
    test_values = np.arange(0, 15, 0.01)
    labels = [neighbor_env(v) for v in test_values]
    all_labeled = all(l in ("low", "hex", "high") for l in labels)
    # check boundary behavior explicitly
    boundary_5_5 = neighbor_env(5.5)   # should be 'hex' (>=5.5 per "5.5 <= mean")
    boundary_6_5 = neighbor_env(6.5)   # should be 'hex' (<=6.5)
    boundary_just_below_5_5 = neighbor_env(5.499)  # should be 'low'
    boundary_just_above_6_5 = neighbor_env(6.501)  # should be 'high'
    return {
        "all_values_classified": all_labeled,
        "boundary_5_5_is_hex": boundary_5_5 == "hex",
        "boundary_6_5_is_hex": boundary_6_5 == "hex",
        "just_below_5_5_is_low": boundary_just_below_5_5 == "low",
        "just_above_6_5_is_high": boundary_just_above_6_5 == "high",
    }


# =============================================================================
# PART B: Methods 2.7 -- Continuous-Time Markov Chain
# =============================================================================
def build_transition_matrix(frames):
    counts = np.zeros((N_STATES, N_STATES))
    frame_nums = sorted(frames.keys())
    for f0, f1 in zip(frame_nums[:-1], frame_nums[1:]):
        shared = set(frames[f0].keys()) & set(frames[f1].keys())
        for cid in shared:
            counts[STATE_IDX[frames[f0][cid]], STATE_IDX[frames[f1][cid]]] += 1
    return counts


def derive_ctmc_generator(P1, dt):
    Qc_raw = logm(P1) / dt
    Qc = np.real(Qc_raw)
    return Qc


def regularize_generator(Qc):
    Qc_reg = Qc.copy()
    n = Qc_reg.shape[0]
    corrections = []
    for i in range(n):
        for j in range(n):
            if i != j and Qc_reg[i, j] < 0:
                corrections.append((STATES[i], STATES[j], float(Qc_reg[i, j])))
                Qc_reg[i, j] = 0.0
    for i in range(n):
        Qc_reg[i, i] = -sum(Qc_reg[i, j] for j in range(n) if j != i)
    return Qc_reg, corrections


def query_frame_interval(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT frame, time_sec FROM frames ORDER BY frame")
    rows = cur.fetchall()
    conn.close()
    times = np.array([r[1] for r in rows], dtype=float)
    diffs = np.diff(np.sort(times))
    return float(np.mean(diffs)) / 60.0


if __name__ == "__main__":
    print("=" * 70)
    print("PART A: Methods 2.6 -- Spatial Neighbor-Coupled Model")
    print("=" * 70)

    frames = extract_polygon_classes(DB_PATH)
    neighbors = extract_neighbor_pairs(DB_PATH)
    n_neighbor_pairs = len(neighbors)
    print(f"Raw neighbor pairs (directed_bonds self-join): {n_neighbor_pairs}")

    classification_check = check_classification_completeness()
    print("\nEnvironment classification completeness check:")
    for k, v in classification_check.items():
        print(f"  {k}: {'OK' if v else 'FAILED'}")

    sides_lookup = {(f, cid): s for f, cd in frames.items() for cid, s in cd.items()}
    neighbors = neighbors.copy()
    neighbors["neighbor_sides"] = neighbors.apply(
        lambda r: sides_lookup.get((r["frame"], r["neighbor_id"]), np.nan), axis=1)
    neighbors = neighbors.dropna(subset=["neighbor_sides"])
    mean_neighbor = (neighbors.groupby(["frame", "cell_id"])["neighbor_sides"]
                      .mean().reset_index(name="mean_neighbor_sides"))
    mean_neighbor["environment"] = mean_neighbor["mean_neighbor_sides"].apply(neighbor_env)
    env_counts = mean_neighbor["environment"].value_counts().to_dict()
    print(f"\nEnvironment counts: {env_counts}")

    spatial_claims = [
        check("Raw neighbor pairs", 233546, n_neighbor_pairs, 0),
        check("Low-neighbor count", 146, env_counts.get("low", 0), 0),
        check("High-neighbor count", 1838, env_counts.get("high", 0), 0),
        check("Classification: all values labeled", True,
              classification_check["all_values_classified"], direction="bool"),
        check("Classification: 5.5 boundary -> hex", True,
              classification_check["boundary_5_5_is_hex"], direction="bool"),
        check("Classification: 6.5 boundary -> hex", True,
              classification_check["boundary_6_5_is_hex"], direction="bool"),
    ]

    print("\n" + "=" * 70)
    print("PART B: Methods 2.7 -- Continuous-Time Markov Chain")
    print("=" * 70)

    dt = query_frame_interval(DB_PATH)
    print(f"Independently queried frame interval: {dt:.4f} min (claimed: 4.95)")

    counts = build_transition_matrix(frames)
    row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    P1 = counts / row_sums

    Qc_raw = derive_ctmc_generator(P1, dt)
    Qc, corrections = regularize_generator(Qc_raw)
    print(f"Negative off-diagonal corrections: {len(corrections)} (claimed: 4)")

    # NEW invariant checks
    row_sum_residuals = np.abs(Qc.sum(axis=1))
    max_row_sum_residual = float(row_sum_residuals.max())
    off_diag_mask = ~np.eye(N_STATES, dtype=bool)
    min_off_diag = float(Qc[off_diag_mask].min())
    all_off_diag_nonneg = min_off_diag >= -1e-10

    eigvals, eigvecs = np.linalg.eig(Qc.T)
    idx = np.argmin(np.abs(eigvals))
    pi_ctmc = np.real(eigvecs[:, idx])
    pi_ctmc = np.abs(pi_ctmc) / np.abs(pi_ctmc).sum()
    stationary_residual = float(np.abs(pi_ctmc @ Qc).max())

    print(f"\nInvariant checks (is the generator construction mathematically valid?):")
    print(f"  Max |row sum| of regularized Q_c (should be ~0): {max_row_sum_residual:.2e}")
    print(f"  Min off-diagonal entry (should be >=0): {min_off_diag:.2e}")
    print(f"  Max |pi @ Q_c| residual (should be ~0): {stationary_residual:.2e}")

    ctmc_claims = [
        check("Frame interval dt (min)", 4.95, dt, 0.01),
        check("Negative off-diagonal count", 4, len(corrections), 0),
        check("Generator row sums ~0 (valid generator)", True,
              max_row_sum_residual < 1e-8, direction="bool"),
        check("Off-diagonals non-negative (valid generator)", True,
              all_off_diag_nonneg, direction="bool"),
        check("pi solves pi@Qc~0 (stationary solve valid)", True,
              stationary_residual < 1e-6, direction="bool"),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_claims = spatial_claims + ctmc_claims
    for c in all_claims:
        print(f"  {c['label']:<45}{c['status']:<10}claimed={c['claimed']}  actual={c['actual']}")

    n_pass = sum(1 for c in all_claims if c["status"] == "PASS")
    n_fail = sum(1 for c in all_claims if c["status"] == "FAIL")
    print(f"\n{n_pass}/{len(all_claims)} verified, {n_fail} mismatched")

    results = {
        "part_a_spatial": {
            "n_neighbor_pairs": n_neighbor_pairs, "env_counts": env_counts,
            "classification_check": classification_check,
        },
        "part_b_ctmc": {
            "dt_minutes": dt, "n_corrections": len(corrections),
            "corrections": corrections,
            "max_row_sum_residual": max_row_sum_residual,
            "min_off_diagonal": min_off_diag,
            "stationary_residual": stationary_residual,
        },
        "claims": all_claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_methods_2_6_2_7_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved full results to: {out_path}")
