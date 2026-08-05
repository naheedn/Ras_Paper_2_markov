"""
sqlite_markov_analysis.py — Corrected, unified first-order Markov chain analysis

WHY THIS EXISTS
----------------
The original Paper 2 pipeline (drosophila_markov.py) derived polygon class by
running binary_dilation + np.unique() on RGB *visualization* images
(tracked_cells_resized.tif). Those images encode cell identity as distinct RGB
color triplets, not as a monotonic grayscale/integer cell ID — so treating
unique colors as a proxy for neighbor count produced meaningless results.

This script instead uses TissueMiner's own directed_bonds SQLite table, which
is the tool's authoritative structural record of cell-cell adjacency. Polygon
class for a given cell in a given frame = COUNT(*) of directed_bonds rows for
that (frame, cell_id) pair.

Run identically on all four datasets (demo, WT_1, WT_2, WT_3) so the numbers
are directly comparable and methodologically consistent.

BEFORE RUNNING: verify the schema matches what this script expects.
    sqlite3 <db> ".schema directed_bonds"
Expected columns: frame, cell_id (one row per bond/side).
If column names differ, edit extract_polygon_classes() accordingly.

Usage:
    python sqlite_markov_analysis.py
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Dataset paths — adjust if yours differ ──────────────────────────────
DATASETS = {
    "demo": Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite",
    "WT_1": Path.home() / "TissueMiner_WT_Data" / "example_data" / "WT_1" / "WT_1.sqlite",
    "WT_2": Path.home() / "TissueMiner_WT_Data" / "example_data" / "WT_2" / "WT_2.sqlite",
    "WT_3": Path.home() / "TissueMiner_WT_Data" / "example_data" / "WT_3" / "WT_3.sqlite",
}

STATES = [4, 5, 6, 7, 8]          # polygon classes modeled; cells outside this range are dropped
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)


def extract_polygon_classes(db_path):
    """
    Query directed_bonds: polygon class = COUNT(*) grouped by (frame, cell_id).
    Returns a dict: {frame_number: {cell_id: polygon_class}}, restricted to
    STATES (4–8 sides) since cells outside this range are geometrically rare
    edge cases / likely segmentation artifacts.
    """
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
    """
    Build a first-order transition COUNT matrix from consecutive frame pairs.
    A cell contributes a transition (state_t -> state_t+1) only if its
    cell_id is present in both consecutive frames (i.e., it wasn't lost to
    tracking, division, or apoptosis between frames).
    """
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
    """
    Solve pi P = pi via the left eigenvector of P for eigenvalue 1
    (equivalently, right eigenvector of P^T), normalized to sum to 1.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    pi = pi / pi.sum()
    return pi


def run_analysis(name, db_path):
    print(f"\n{'='*60}\nDataset: {name}  ({db_path})\n{'='*60}")

    if not db_path.exists():
        print("  WARNING: file not found, skipping.")
        return None

    frames = extract_polygon_classes(db_path)
    print(f"Frames with polygon data: {len(frames)}")
    if len(frames) < 2:
        print("  WARNING: fewer than 2 frames with data, cannot compute transitions.")
        return None

    # Observed (raw) polygon class distribution across all frames
    all_states = [s for frame_dict in frames.values() for s in frame_dict.values()]
    observed = {s: all_states.count(s) / len(all_states) for s in STATES}
    print("Observed distribution:")
    for s in STATES:
        print(f"  {s}-sided: {observed[s]*100:.2f}%")

    counts, n_transitions = build_transition_matrix(frames)
    print(f"Total cell-to-cell transitions used: {n_transitions}")

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for unvisited states
    P = counts / row_sums

    pi = stationary_distribution(P)
    print("First-order stationary distribution:")
    result = {}
    for s, p in zip(STATES, pi):
        print(f"  {s}-sided: {p*100:.2f}%")
        result[s] = float(p)

    return {
        "n_frames": len(frames),
        "n_transitions": n_transitions,
        "observed_distribution": observed,
        "transition_matrix": P.tolist(),
        "stationary_distribution": result,
        "hexagonal_stationary_pct": round(result[6] * 100, 2),
        "hexagonal_observed_pct": round(observed[6] * 100, 2),
    }


if __name__ == "__main__":
    all_results = {}
    for name, path in DATASETS.items():
        res = run_analysis(name, path)
        if res is not None:
            all_results[name] = res

    out_path = Path.home() / "RAS_Project" / "results" / "corrected_markov_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nSaved full results to: {out_path}")
    print("\nSummary (first-order stationary hexagonal %):")
    for name, res in all_results.items():
        print(f"  {name}: {res['hexagonal_stationary_pct']}% (observed: {res['hexagonal_observed_pct']}%)")
