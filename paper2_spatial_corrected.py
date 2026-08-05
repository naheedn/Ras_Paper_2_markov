"""
paper2_spatial_corrected.py — Spatial neighbor-coupled model, corrected
and formally tested

WHY THIS EXISTS
----------------
paper2_spatial_markov.py has two problems, not one:

  1. It derives cell polygon classes from cellshapes.RData, not
     directed_bonds -- inconsistent with the directed_bonds-based pipeline
     used everywhere else in the corrected paper (Methods 2.2).
  2. Its first-order comparison baseline is HARDCODED:
         pi1 = np.array([0.0946, 0.1537, 0.2353, 0.2457, 0.2707])
     0.2353 is the PRE-CORRECTION, buggy hexagonal stationary value
     (23.53%), not the corrected 58.47%. Every "deviation from baseline"
     figure that script prints is computed against the wrong number.

This script fixes both, reusing verified code where it already exists:

  - extract_polygon_classes(), build_transition_matrix()  -- VERBATIM from
    ras_corrected_markov_analysis.py (the source of the corrected 58.47%
    first-order figure used throughout the rest of the paper)
  - The directed_bonds neighbor-pair self-join query -- VERBATIM from
    paper2_spatial_markov.py Section 2 (this part was never the problem;
    it's already directed_bonds-based)

Then it adds what was missing entirely: a formal test of whether neighbor
environment predicts transition behavior beyond the cell's own current
state, using the same Anderson-Goodman order-test machinery already
applied to the second-order model, plus a calibration check to confirm
the test doesn't spuriously reject when environment truly carries no
information (mirroring the second-order negative control).

Run:
    conda activate ras_project
    python paper2_spatial_corrected.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"
OUTPUT_DIR = Path.home() / "RAS_Project" / "results" / "paper2"

STATES = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
ENVS = ["low", "hex", "high"]


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


# ---- verbatim from paper2_spatial_markov.py Section 2 -------------------
def extract_neighbor_pairs(db_path):
    conn = sqlite3.connect(str(db_path))
    neighbor_query = """
    SELECT
        d1.frame        AS frame,
        d1.cell_id      AS cell_id,
        d2.cell_id      AS neighbor_id
    FROM directed_bonds d1
    JOIN directed_bonds d2
      ON d1.frame = d2.frame
     AND d1.conj_dbond_id = d2.dbond_id
    WHERE d1.cell_id != d2.cell_id
    """
    neighbors = pd.read_sql(neighbor_query, conn)
    conn.close()
    neighbors["frame"] = neighbors["frame"].astype(int)
    neighbors["cell_id"] = neighbors["cell_id"].astype(int)
    neighbors["neighbor_id"] = neighbors["neighbor_id"].astype(int)
    return neighbors


def neighbor_env(mns):
    if mns < 5.5:
        return "low"
    elif mns <= 6.5:
        return "hex"
    else:
        return "high"


# ---- new: fully directed_bonds-consistent environment assignment --------
def assign_environments(frames, neighbors):
    """
    Attaches nb_sides to neighbors using the SAME directed_bonds-derived
    `frames` dict used for P1, computes mean neighbor sides per
    (frame, cell_id), and classifies into low/hex/high. Unlike the
    original script, there is no separate RData-derived side lookup --
    one consistent state source throughout.
    """
    sides_lookup = {}
    for frame, cell_dict in frames.items():
        for cid, sides in cell_dict.items():
            sides_lookup[(frame, cid)] = sides

    neighbors = neighbors.copy()
    neighbors["neighbor_sides"] = neighbors.apply(
        lambda r: sides_lookup.get((r["frame"], r["neighbor_id"]), np.nan), axis=1)
    neighbors = neighbors.dropna(subset=["neighbor_sides"])

    mean_neighbor = (neighbors.groupby(["frame", "cell_id"])["neighbor_sides"]
                      .mean().reset_index(name="mean_neighbor_sides"))
    mean_neighbor["environment"] = mean_neighbor["mean_neighbor_sides"].apply(neighbor_env)

    env_lookup = {(int(r.frame), int(r.cell_id)): r.environment
                  for r in mean_neighbor.itertuples()}
    return env_lookup, mean_neighbor


def build_env_conditioned_matrices(frames, env_lookup):
    """Frame-to-frame transitions, split by the CURRENT cell's neighbor
    environment at the current frame (matches the original script's
    convention: environment is evaluated at the 'current' timepoint)."""
    counts = {env: np.zeros((N_STATES, N_STATES)) for env in ENVS}
    n_trans = {env: 0 for env in ENVS}
    triplet_records = {env: [] for env in ENVS}  # for the formal test: (curr, next) pairs

    frame_nums = sorted(frames.keys())
    for f0, f1 in zip(frame_nums[:-1], frame_nums[1:]):
        cells0, cells1 = frames[f0], frames[f1]
        shared = set(cells0.keys()) & set(cells1.keys())
        for cid in shared:
            env = env_lookup.get((f0, cid))
            if env is None:
                continue
            curr, nxt = cells0[cid], cells1[cid]
            counts[env][STATE_IDX[curr], STATE_IDX[nxt]] += 1
            n_trans[env] += 1
            triplet_records[env].append((curr, nxt))

    P_env = {}
    for env in ENVS:
        row_sums = counts[env].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P_env[env] = counts[env] / row_sums
    return P_env, n_trans, triplet_records, counts


# ---- formal test: does environment predict next-state beyond curr state? -
def anderson_goodman_environment_test(triplet_records):
    """
    H0: P(next | curr, environment) = P(next | curr)  -- environment adds
        no information beyond the cell's own current state.
    H1: transition probabilities depend on environment.

    For each current state k, build a contingency table of
    environment x next-state counts (pooling records across all three
    environments for that curr state), G-test against the
    pooled-over-environment expectation, sum G/df across strata.
    """
    G_total, df_total = 0.0, 0
    detail = {}

    for curr in STATES:
        envs_present = [env for env in ENVS if any(c == curr for c, n in triplet_records[env])]
        if len(envs_present) < 2:
            continue

        table = np.zeros((len(envs_present), N_STATES))
        for i, env in enumerate(envs_present):
            for c, n in triplet_records[env]:
                if c == curr:
                    table[i, STATE_IDX[n]] += 1

        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        grand_total = table.sum()
        if grand_total == 0:
            continue
        expected = row_totals @ col_totals / grand_total

        mask = table > 0
        G = 2 * np.sum(table[mask] * np.log(table[mask] / expected[mask]))
        df = (len(envs_present) - 1) * (N_STATES - 1)
        G_total += G
        df_total += df
        detail[curr] = {"G": float(G), "df": int(df), "envs_present": envs_present,
                         "n_total": int(grand_total),
                         "n_by_env": {env: int(table[i].sum()) for i, env in enumerate(envs_present)}}

    if df_total == 0:
        return None, 0, None, detail
    p_value = stats.chi2.sf(G_total, df_total)
    return float(G_total), int(df_total), float(p_value), detail


# ---- calibration check: does the test spuriously reject when environment
#      truly carries no information? ---------------------------------------
def negative_control_calibration(P1, env_proportions, n_cells, n_frames,
                                  n_replicates=200, alpha=0.05, seed=0):
    """
    Simulates n_replicates datasets where cell state trajectories follow
    P1 EXACTLY (no environment dependence by construction) and environment
    labels are assigned independently, at random, in the same marginal
    proportions as observed in the real data. If the test is well
    calibrated, the false-positive rate should sit near alpha (0.05).
    """
    rng = np.random.default_rng(seed)
    eigvals, eigvecs = np.linalg.eig(P1.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi0 = np.real(eigvecs[:, idx]); pi0 = np.abs(pi0) / np.abs(pi0).sum()

    env_names = list(env_proportions.keys())
    env_probs = np.array([env_proportions[e] for e in env_names])
    env_probs = env_probs / env_probs.sum()

    rejections, valid_runs, p_values = 0, 0, []

    for rep in range(n_replicates):
        triplet_records = {env: [] for env in ENVS}
        for cid in range(n_cells):
            state = rng.choice(N_STATES, p=pi0)
            for f in range(n_frames - 1):
                curr = STATES[state]
                env = rng.choice(env_names, p=env_probs)
                next_state = rng.choice(N_STATES, p=P1[state])
                triplet_records[env].append((curr, STATES[next_state]))
                state = next_state

        G, df, p, _ = anderson_goodman_environment_test(triplet_records)
        if p is None:
            continue
        valid_runs += 1
        p_values.append(p)
        if p < alpha:
            rejections += 1

    fpr = rejections / valid_runs if valid_runs else None
    return {
        "n_replicates_requested": n_replicates, "n_valid_runs": valid_runs,
        "false_positive_rate": fpr, "nominal_alpha": alpha,
        "mean_p_value": float(np.mean(p_values)) if p_values else None,
        "calibrated": bool(abs(fpr - alpha) < 0.05) if fpr is not None else None,
    }


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} (directed_bonds-derived) ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    print("Deriving FRESH first-order P1 (replaces the hardcoded 23.53%-baseline array) ...")
    counts, n_transitions = build_transition_matrix(frames)
    row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    P1 = counts / row_sums
    pi1 = stationary_distribution(P1)
    print(f"P1 built from {n_transitions} transitions. "
          f"Hexagonal stationary: {pi1[STATE_IDX[6]]*100:.2f}% "
          f"(sanity check: should be ~58.47%, NOT 23.53%)")

    print("\nExtracting neighbor pairs (directed_bonds self-join) ...")
    neighbors = extract_neighbor_pairs(DB_PATH)
    print(f"Neighbor pairs: {len(neighbors)}")

    print("Assigning environments using directed_bonds-derived states throughout ...")
    env_lookup, mean_neighbor = assign_environments(frames, neighbors)
    env_counts = mean_neighbor["environment"].value_counts()
    total_cells = len(mean_neighbor)
    env_proportions = {env: env_counts.get(env, 0) / total_cells for env in ENVS}
    print("Environment distribution:")
    for env in ENVS:
        print(f"  {env:5s}: {env_counts.get(env,0)} ({env_proportions[env]*100:.1f}%)")

    print("\nBuilding environment-conditioned transition matrices ...")
    P_env, n_trans, triplet_records, raw_counts = build_env_conditioned_matrices(frames, env_lookup)
    pi_env = {env: stationary_distribution(P_env[env]) for env in ENVS}

    print(f"\n{'Environment':<10}{'n transitions':>15}{'Hexagonal stationary %':>25}")
    print("-" * 50)
    print(f"{'baseline (P1)':<10}{n_transitions:>15}{pi1[STATE_IDX[6]]*100:>24.2f}%")
    for env in ENVS:
        print(f"{env:<10}{n_trans[env]:>15}{pi_env[env][STATE_IDX[6]]*100:>24.2f}%")

    print("\nRunning Anderson-Goodman test: does environment predict next-state "
          "beyond current state? ...")
    G, df, p, detail = anderson_goodman_environment_test(triplet_records)
    if G is not None:
        print(f"G={G:.2f}, df={df}, p={p:.3e}")
        print(f"{'SIGNIFICANT' if p < 0.001 else 'not significant'} at p<0.001")
    else:
        print("Insufficient data across environments for a formal test.")

    print("\nRunning negative-control calibration (200 simulated environment-independent "
          "datasets) ...")
    n_cells_approx = len(set(cid for cell_dict in frames.values() for cid in cell_dict))
    calib = negative_control_calibration(
        P1, env_proportions, n_cells=n_cells_approx, n_frames=len(frames), n_replicates=200)
    if calib["false_positive_rate"] is not None:
        print(f"False positive rate: {calib['false_positive_rate']*100:.1f}% "
              f"(nominal 5%), calibrated={calib['calibrated']}")
    else:
        print("Calibration check produced no valid runs -- see output for diagnostics.")

    results = {
        "n_frames": len(frames), "n_transitions": n_transitions,
        "baseline_hexagonal_pct": round(float(pi1[STATE_IDX[6]]) * 100, 4),
        "baseline_stationary_distribution": {s: float(pi1[STATE_IDX[s]]) for s in STATES},
        "environment_proportions": env_proportions,
        "environment_results": {
            env: {
                "n_transitions": n_trans[env],
                "hexagonal_stationary_pct": round(float(pi_env[env][STATE_IDX[6]]) * 100, 4),
                "stationary_distribution": {s: float(pi_env[env][STATE_IDX[s]]) for s in STATES},
            } for env in ENVS
        },
        "anderson_goodman_test": {"G": G, "df": df, "p_value": p, "per_curr_state_detail": detail},
        "negative_control_calibration": calib,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_spatial_corrected_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
