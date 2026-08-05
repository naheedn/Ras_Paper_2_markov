"""
paper2_aboav_correlation_verification.py - Verifies the Aboav's law
correlation claimed throughout the paper: r = -0.611, between a cell's
own polygon class and its mean neighbor polygon class.

WHY THIS EXISTS
----------------
This number has been in the manuscript since before this week's
verification pass and was never independently recomputed -- unlike every
other spatial-model number, which was checked via paper2_spatial_
corrected.py. This script computes it directly from the same verified
directed_bonds pipeline.

Reuses VERBATIM:
  - extract_polygon_classes()  from ras_corrected_markov_analysis.py
  - extract_neighbor_pairs()   from paper2_spatial_corrected.py (the
    directed_bonds self-join on conjugate bond indices, already verified
    to produce exactly 233,546 raw neighbor pairs)

Run:
    conda activate ras_project
    python paper2_aboav_correlation_verification.py
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


# ---- verbatim from paper2_spatial_corrected.py --------------------------
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


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    print("Extracting neighbor pairs (directed_bonds self-join) ...")
    neighbors = extract_neighbor_pairs(DB_PATH)
    print(f"Raw neighbor pairs: {len(neighbors)} (sanity check: should be 233,546)")

    # Attach each neighbor's own polygon class (sides), using the same
    # directed_bonds-derived states as everywhere else this week.
    sides_lookup = {}
    for frame, cell_dict in frames.items():
        for cid, sides in cell_dict.items():
            sides_lookup[(frame, cid)] = sides

    neighbors = neighbors.copy()
    neighbors["neighbor_sides"] = neighbors.apply(
        lambda r: sides_lookup.get((r["frame"], r["neighbor_id"]), np.nan), axis=1)
    neighbors = neighbors.dropna(subset=["neighbor_sides"])

    # Mean neighbor polygon class per (frame, cell_id)
    mean_neighbor = (neighbors.groupby(["frame", "cell_id"])["neighbor_sides"]
                      .mean().reset_index(name="mean_neighbor_sides"))

    # Attach the cell's OWN polygon class
    mean_neighbor["own_sides"] = mean_neighbor.apply(
        lambda r: sides_lookup.get((r["frame"], r["cell_id"]), np.nan), axis=1)
    mean_neighbor = mean_neighbor.dropna(subset=["own_sides"])

    n_cells = len(mean_neighbor)
    print(f"\nCells with both own polygon class and mean neighbor class: {n_cells}")

    own = mean_neighbor["own_sides"].values.astype(float)
    mean_nb = mean_neighbor["mean_neighbor_sides"].values.astype(float)

    r, p_value = stats.pearsonr(own, mean_nb)

    print(f"\nPearson correlation (own polygon class vs. mean neighbor class):")
    print(f"  r = {r:.4f}  (claimed: -0.611)")
    print(f"  p = {p_value:.3e}")
    print(f"  n = {n_cells}")

    diff = abs(-0.611 - r)
    status = "PASS" if diff <= 0.01 else "FAIL"
    print(f"\n{'='*60}")
    print(f"CLAIM CHECK: Aboav correlation r=-0.611 -- {status} "
          f"(claimed=-0.611, actual={r:.4f}, diff={diff:.4f})")
    print(f"{'='*60}")

    if status == "FAIL":
        print("\nDo not treat this as correct until reconciled -- check whether the")
        print("original computation used a different neighbor-averaging method,")
        print("different frame subset, or different state source.")

    results = {
        "n_frames": len(frames),
        "n_raw_neighbor_pairs": len(neighbors),
        "n_cells_in_correlation": n_cells,
        "pearson_r": float(r),
        "p_value": float(p_value),
        "claimed_r": -0.611,
        "diff": float(diff),
        "status": status,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_aboav_correlation_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
