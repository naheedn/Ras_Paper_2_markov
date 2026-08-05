"""
paper2_environment_transition_counts_verification.py - Verifies the
Limitations section claim: "the high-neighbor environment contained only
1,838 transitions... the low-neighbor environment contained only 146
transitions."

WHY THIS EXISTS
----------------
1,838 and 146 were already verified this week -- but as CELL-FRAME
CLASSIFICATION counts (how many cell-frame observations fall into each
neighbor environment), not TRANSITION counts (how many of those cells
also have a valid next-frame state, making them usable in the
neighbor-conditioned transition matrices, Fig 14).

A cell is classified into an environment once per frame it appears in,
but only counts as a "transition" if it ALSO has a tracked state in the
following frame. This means the transition count for each environment
should be <= the classification count, and is likely a different,
smaller number.

The Limitations text calls 1,838 and 146 "transitions" -- this script
checks whether that's the correct quantity, or whether the classification
counts were mislabeled.

Reuses VERBATIM:
  - extract_polygon_classes()  from ras_corrected_markov_analysis.py
  - extract_neighbor_pairs()   from paper2_spatial_corrected.py
  - neighbor_env() classification thresholds, also unchanged

Run:
    conda activate ras_project
    python paper2_environment_transition_counts_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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


def neighbor_env(mns):
    if mns < 5.5:
        return "low"
    elif mns <= 6.5:
        return "hex"
    else:
        return "high"


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    print("Extracting neighbor pairs (directed_bonds self-join) ...")
    neighbors = extract_neighbor_pairs(DB_PATH)
    print(f"Raw neighbor pairs: {len(neighbors)} (sanity check: should be 233,546)")

    sides_lookup = {(f, cid): s for f, cd in frames.items() for cid, s in cd.items()}
    neighbors = neighbors.copy()
    neighbors["neighbor_sides"] = neighbors.apply(
        lambda r: sides_lookup.get((r["frame"], r["neighbor_id"]), np.nan), axis=1)
    neighbors = neighbors.dropna(subset=["neighbor_sides"])

    mean_neighbor = (neighbors.groupby(["frame", "cell_id"])["neighbor_sides"]
                      .mean().reset_index(name="mean_neighbor_sides"))
    mean_neighbor["environment"] = mean_neighbor["mean_neighbor_sides"].apply(neighbor_env)

    # --- Classification counts (already verified: should be low=146, high=1838) ---
    classification_counts = mean_neighbor["environment"].value_counts().to_dict()
    print(f"\nClassification counts (cell-frame observations, already verified): "
          f"{classification_counts}")

    # --- Transition counts: does this classified cell ALSO have a valid ---
    # --- state in the NEXT frame? Only those count toward the neighbor- ---
    # --- conditioned transition matrices (Fig 14). ---------------------
    mean_neighbor["has_next_frame_state"] = mean_neighbor.apply(
        lambda r: (r["frame"] + 1, r["cell_id"]) in sides_lookup, axis=1)

    transition_counts = (mean_neighbor[mean_neighbor["has_next_frame_state"]]
                          ["environment"].value_counts().to_dict())
    print(f"Transition counts (has valid next-frame state): {transition_counts}")

    dropped = {env: classification_counts.get(env, 0) - transition_counts.get(env, 0)
               for env in ("low", "hex", "high")}
    print(f"\nCells dropped (classified but no next-frame state): {dropped}")

    print("\n" + "=" * 70)
    print("CLAIM CHECK: Limitations section wording")
    print("=" * 70)

    claims = []
    for env, claimed_label in [("high", 1838), ("low", 146)]:
        actual_class = classification_counts.get(env, 0)
        actual_trans = transition_counts.get(env, 0)
        matches_classification = (claimed_label == actual_class)
        matches_transition = (claimed_label == actual_trans)
        print(f"\n{env.upper()}-neighbor environment, manuscript claims {claimed_label} 'transitions':")
        print(f"  Actual classification count (cell-frame observations): {actual_class} "
              f"{'<-- MATCHES claimed number' if matches_classification else ''}")
        print(f"  Actual transition count (has next-frame state):        {actual_trans} "
              f"{'<-- MATCHES claimed number' if matches_transition else ''}")
        if matches_classification and not matches_transition:
            verdict = ("MISLABELED: the claimed number is the CLASSIFICATION count, "
                       "not a transition count. Either the word 'transitions' should "
                       "be 'cell-frame observations', or the number should be updated "
                       f"to the true transition count ({actual_trans}).")
        elif matches_transition:
            verdict = "CORRECT: claimed number matches the true transition count."
        else:
            verdict = "NEITHER MATCHES: claimed number doesn't match either quantity -- needs investigation."
        print(f"  Verdict: {verdict}")
        claims.append({
            "environment": env, "claimed": claimed_label,
            "actual_classification_count": actual_class,
            "actual_transition_count": actual_trans,
            "matches_classification": matches_classification,
            "matches_transition": matches_transition,
            "verdict": verdict,
        })

    results = {
        "classification_counts": classification_counts,
        "transition_counts": transition_counts,
        "dropped_counts": dropped,
        "claims": claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_environment_transition_counts_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
