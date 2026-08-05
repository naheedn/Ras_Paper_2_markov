"""
paper2_triplet_cells_verification.py - Verifies Methods 2.5's "681 distinct
cells" claim: the number of unique cell_id values that contribute at least
one valid (prev, curr, next) triplet across the sliding three-consecutive-
frame window.

WHY THIS EXISTS
----------------
paper2_second_order_corrected.py's extract_triplets() computes the shared
cell set per (f0,f1,f2) window internally, but never accumulates or prints
the UNION of those sets across all windows -- so "681 distinct cells" has
never actually been computed by any script this week. This one does,
reusing the same verbatim extraction and triplet logic already verified.

Run:
    conda activate ras_project
    python paper2_triplet_cells_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

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


# ---- extended from paper2_second_order_corrected.py's extract_triplets --
def extract_triplets_with_cell_tracking(frames):
    """Same triplet-extraction logic as the verified second-order script,
    but additionally accumulates the UNION of cell_ids that contribute at
    least one triplet across all sliding three-frame windows."""
    triplet_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    contributing_cells = set()
    frame_nums = sorted(frames.keys())

    n_triplets = 0
    for f0, f1, f2 in zip(frame_nums[:-2], frame_nums[1:-1], frame_nums[2:]):
        cells0, cells1, cells2 = frames[f0], frames[f1], frames[f2]
        shared = set(cells0.keys()) & set(cells1.keys()) & set(cells2.keys())
        for cid in shared:
            prev, curr, nxt = cells0[cid], cells1[cid], cells2[cid]
            triplet_counts[(prev, curr, nxt)] += 1
            pair_counts[(prev, curr)] += 1
            contributing_cells.add(cid)
            n_triplets += 1

    return triplet_counts, pair_counts, n_triplets, contributing_cells


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    triplet_counts, pair_counts, n_triplets, contributing_cells = \
        extract_triplets_with_cell_tracking(frames)

    n_distinct_cells = len(contributing_cells)

    print(f"\nTotal triplets: {n_triplets} (sanity check: should be 34,944)")
    print(f"Distinct cells contributing >=1 triplet: {n_distinct_cells} "
          f"(claimed in Methods 2.5: 681)")

    # sanity cross-check against the 710-unique-cells figure from Methods 2.2
    # (that count is over consecutive-PAIR tracking, a looser requirement
    # than triplet tracking, so 681 should be <= 710, never greater)
    all_cell_ids_any_frame = set()
    for cell_dict in frames.values():
        all_cell_ids_any_frame.update(cell_dict.keys())
    print(f"\nCross-check: total distinct cell_ids appearing in ANY frame: "
          f"{len(all_cell_ids_any_frame)}")
    print("(681 triplet-contributing cells should be <= this total, and <= "
          "the 710 pair-tracked figure from Methods 2.2, since triplet "
          "tracking across 3 consecutive frames is a stricter requirement)")

    claimed = 681
    diff = abs(claimed - n_distinct_cells)
    status = "PASS" if diff == 0 else "FAIL"
    print(f"\n{'='*60}")
    print(f"CLAIM CHECK: '681 distinct cells' -- {status} "
          f"(claimed={claimed}, actual={n_distinct_cells}, diff={diff})")
    print(f"{'='*60}")

    results = {
        "n_frames": len(frames),
        "n_triplets": n_triplets,
        "n_distinct_cells_triplet_tracked": n_distinct_cells,
        "n_distinct_cells_any_frame": len(all_cell_ids_any_frame),
        "claimed_681": claimed,
        "status": status,
        "diff": diff,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_triplet_cells_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
