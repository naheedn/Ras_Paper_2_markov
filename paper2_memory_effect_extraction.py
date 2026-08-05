"""
paper2_memory_effect_extraction.py — Figure 6 data extraction
(per-pair memory-effect deviations, |P2 - P1|)

WHY THIS EXISTS
----------------
Figure 6 needs, for every observed (prev, curr) pair, the maximum absolute
deviation between the second-order transition probabilities
P2[(prev,curr) -> next] and the first-order transition probabilities
P1[curr -> next]. Neither ras_corrected_markov_analysis.py (which only
saves the first-order stationary summary) nor
paper2_second_order_corrected.py (which only saves stationary-distribution
marginals and G-test stats) ever wrote out the full pair-to-next matrix
needed for this. This script computes and saves it, and regenerates
fig6.png directly.

Reuses VERBATIM:
  - extract_polygon_classes()      from ras_corrected_markov_analysis.py
  - build_transition_matrix()      from ras_corrected_markov_analysis.py
    (first-order P1)
  - extract_triplets()             from paper2_second_order_corrected.py
  - build_pair_transition_matrix() from paper2_second_order_corrected.py
    (second-order P2, full model, no threshold)

Run:
    conda activate ras_project
    python paper2_memory_effect_extraction.py

Produces:
    paper2_memory_effect.json   -- per-pair deviations, all next-state probs
    fig6.png                    -- regenerated figure, same style as before
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only the demo dataset is used for Figure 6 in the current manuscript.
DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"

STATES = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
MEMORY_THRESHOLD = 0.05  # matches the paper's stated 0.05 threshold


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


# ---- verbatim from paper2_second_order_corrected.py ---------------------
def extract_triplets(frames):
    triplet_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    frame_nums = sorted(frames.keys())
    n_triplets = 0
    for f0, f1, f2 in zip(frame_nums[:-2], frame_nums[1:-1], frame_nums[2:]):
        cells0, cells1, cells2 = frames[f0], frames[f1], frames[f2]
        shared = set(cells0.keys()) & set(cells1.keys()) & set(cells2.keys())
        for cid in shared:
            prev, curr, nxt = cells0[cid], cells1[cid], cells2[cid]
            triplet_counts[(prev, curr, nxt)] += 1
            pair_counts[(prev, curr)] += 1
            n_triplets += 1
    return triplet_counts, pair_counts, n_triplets


def build_pair_transition_matrix(triplet_counts, pair_counts, min_count=0):
    observed_pairs = sorted(p for p, c in pair_counts.items() if c >= min_count)
    n_obs = len(observed_pairs)
    P2 = np.zeros((n_obs, N_STATES))
    counts_obs = np.zeros(n_obs)
    for i, (prev, curr) in enumerate(observed_pairs):
        total = pair_counts[(prev, curr)]
        counts_obs[i] = total
        for j, nxt in enumerate(STATES):
            P2[i, j] = triplet_counts.get((prev, curr, nxt), 0) / total
    return observed_pairs, P2, counts_obs


# ---- new: memory-effect deviation computation ---------------------------
def compute_memory_deviations(observed_pairs, P2, P1, pair_counts):
    """
    For each (prev, curr) pair, compute max_j |P2[j] - P1[curr, j]| --
    the largest single-next-state deviation between the second-order
    transition probabilities for this specific pair and the first-order
    transition probabilities for its current state alone.
    """
    records = []
    for i, (prev, curr) in enumerate(observed_pairs):
        p1_row = P1[STATE_IDX[curr]]
        deviations = np.abs(P2[i] - p1_row)
        max_dev = float(deviations.max())
        max_dev_state = STATES[int(deviations.argmax())]
        records.append({
            "prev": prev, "curr": curr,
            "pair_label": f"{prev}\u2192{curr}",
            "n": int(pair_counts[(prev, curr)]),
            "max_abs_deviation": max_dev,
            "max_deviation_next_state": max_dev_state,
            "P2_by_next_state": {str(s): float(P2[i, j]) for j, s in enumerate(STATES)},
            "P1_curr_row": {str(s): float(p1_row[j]) for j, s in enumerate(STATES)},
        })
    records.sort(key=lambda r: r["max_abs_deviation"], reverse=True)
    return records


def make_figure(records, out_path, threshold=MEMORY_THRESHOLD):
    labels = [r["pair_label"] for r in records]
    devs = [r["max_abs_deviation"] for r in records]
    colors = ["#e07b39" if d > threshold else "#8899aa" for d in devs]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(records))))
    y_pos = np.arange(len(records))
    ax.barh(y_pos, devs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold ({threshold})")
    ax.set_xlabel(r"$\max_j |P_2[(\mathrm{prev,curr}) \to j] - P_1[\mathrm{curr} \to j]|$")
    ax.set_title("Memory effect: deviation of second-order from first-order transitions")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    print("Building first-order transition matrix (P1) ...")
    counts, n_transitions = build_transition_matrix(frames)
    row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    P1 = counts / row_sums
    print(f"P1 built from {n_transitions} transitions.")

    print("Extracting triplets and building second-order matrix (P2, full model) ...")
    triplet_counts, pair_counts, n_triplets = extract_triplets(frames)
    observed_pairs, P2, counts_obs = build_pair_transition_matrix(
        triplet_counts, pair_counts, min_count=0)
    print(f"P2 built from {n_triplets} triplets across {len(observed_pairs)} observed pairs.")

    print("Computing per-pair memory-effect deviations ...")
    records = compute_memory_deviations(observed_pairs, P2, P1, pair_counts)

    strongest = records[0]
    weakest = records[-1]
    n_above_threshold = sum(1 for r in records if r["max_abs_deviation"] > MEMORY_THRESHOLD)

    print(f"\nStrongest memory: {strongest['pair_label']} "
          f"(deviation {strongest['max_abs_deviation']:.4f}, n={strongest['n']})")
    print(f"Weakest memory:   {weakest['pair_label']} "
          f"(deviation {weakest['max_abs_deviation']:.4f}, n={weakest['n']})")
    print(f"Pairs exceeding {MEMORY_THRESHOLD} threshold: "
          f"{n_above_threshold} of {len(records)}")

    out_json = Path.home() / "RAS_Project" / "results" / "paper2" / "paper2_memory_effect.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "n_frames": len(frames), "n_transitions": n_transitions,
            "n_triplets": n_triplets, "n_pairs_observed": len(observed_pairs),
            "threshold": MEMORY_THRESHOLD,
            "strongest_pair": strongest["pair_label"],
            "strongest_deviation": strongest["max_abs_deviation"],
            "weakest_pair": weakest["pair_label"],
            "weakest_deviation": weakest["max_abs_deviation"],
            "n_pairs_above_threshold": n_above_threshold,
            "per_pair": records,
        }, f, indent=2)
    print(f"\nSaved per-pair data to: {out_json}")

    out_fig = Path.home() / "RAS_Project" / "results" / "paper2" / "fig6.png"
    make_figure(records, out_fig)
    print(f"Saved regenerated figure to: {out_fig}")
    print("\nCopy fig6.png into the paper2_corrected/ folder to replace the old figure,")
    print("and update the Fig 6 caption's strongest/weakest values using the numbers above.")
