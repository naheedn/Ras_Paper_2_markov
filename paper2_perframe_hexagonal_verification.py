"""
paper2_perframe_hexagonal_verification.py - Verifies the per-frame
hexagonal percentage claims in Section 3.1 / Fig 2 caption:
"ranged from 47.72% to 66.80% (mean 57.90%)"

WHY THIS EXISTS
----------------
Every script this week computed POOLED statistics (all 36,694 cell-frame
observations aggregated together, giving 57.68%). This is a different
quantity: the hexagonal percentage computed SEPARATELY within each of the
71 individual frames, then the min/max/mean taken across those 71 values.
No script has computed this before.

Reuses extract_polygon_classes() VERBATIM from ras_corrected_markov_
analysis.py -- the same verified directed_bonds pipeline as everything
else this week.

Run:
    conda activate ras_project
    python paper2_perframe_hexagonal_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def check(label, claimed, actual, tol):
    diff = abs(claimed - actual)
    status = "PASS" if diff <= tol else "FAIL"
    return {"label": label, "status": status, "claimed": claimed,
            "actual": round(actual, 4), "diff": round(diff, 4)}


if __name__ == "__main__":
    print(f"Loading polygon classes from {DB_PATH} ...")
    frames = extract_polygon_classes(DB_PATH)
    print(f"Frames: {len(frames)}")

    per_frame_pct = {}
    for frame_num, cell_dict in frames.items():
        sides = list(cell_dict.values())
        n_total = len(sides)
        n_hex = sum(1 for s in sides if s == 6)
        pct = (n_hex / n_total * 100) if n_total > 0 else None
        per_frame_pct[frame_num] = {"n_total": n_total, "n_hex": n_hex, "hex_pct": pct}

    pcts = np.array([v["hex_pct"] for v in per_frame_pct.values() if v["hex_pct"] is not None])
    n_frames_with_data = len(pcts)

    min_pct = float(pcts.min())
    max_pct = float(pcts.max())
    mean_pct = float(pcts.mean())

    min_frame = min(per_frame_pct, key=lambda f: per_frame_pct[f]["hex_pct"] or 999)
    max_frame = max(per_frame_pct, key=lambda f: per_frame_pct[f]["hex_pct"] or -1)

    print(f"\nFrames with valid data: {n_frames_with_data} of {len(frames)}")
    print(f"\n{'Statistic':<15}{'Value':>12}")
    print("-" * 27)
    print(f"{'Min':<15}{min_pct:>11.2f}%  (frame {min_frame}, "
          f"n={per_frame_pct[min_frame]['n_total']} cells)")
    print(f"{'Max':<15}{max_pct:>11.2f}%  (frame {max_frame}, "
          f"n={per_frame_pct[max_frame]['n_total']} cells)")
    print(f"{'Mean':<15}{mean_pct:>11.2f}%")

    print("\n" + "=" * 70)
    print("CLAIM CHECK: Section 3.1 / Fig 2 caption")
    print("=" * 70)

    claims = [
        check("Per-frame min hex %", 47.72, min_pct, 0.05),
        check("Per-frame max hex %", 66.80, max_pct, 0.05),
        check("Per-frame mean hex %", 57.90, mean_pct, 0.05),
    ]

    print(f"\n{'Claim':<25}{'Status':<8}{'Claimed':>10}{'Actual':>10}{'Diff':>8}")
    print("-" * 63)
    for c in claims:
        print(f"{c['label']:<25}{c['status']:<8}{c['claimed']:>10}"
              f"{c['actual']:>10}{c['diff']:>8}")

    n_pass = sum(1 for c in claims if c["status"] == "PASS")
    n_fail = sum(1 for c in claims if c["status"] == "FAIL")
    print(f"\nSUMMARY: {n_pass}/{len(claims)} claims verified, {n_fail} mismatched")
    if n_fail > 0:
        print("\nDo not treat FAILed claims as correct until reconciled.")

    # sanity cross-check: pooled mean weighted by frame size should match 57.68%
    total_cells = sum(v["n_total"] for v in per_frame_pct.values())
    total_hex = sum(v["n_hex"] for v in per_frame_pct.values())
    pooled_pct = total_hex / total_cells * 100
    print(f"\nSanity cross-check: pooled (cell-weighted) hexagonal % = {pooled_pct:.2f}% "
          f"(should be 57.68% -- this is a DIFFERENT number from the unweighted "
          f"per-frame mean above, since frames have different cell counts)")

    results = {
        "n_frames": len(frames),
        "n_frames_with_data": n_frames_with_data,
        "per_frame": {str(k): v for k, v in per_frame_pct.items()},
        "min_pct": min_pct, "max_pct": max_pct, "mean_pct": mean_pct,
        "min_frame": min_frame, "max_frame": max_frame,
        "pooled_cell_weighted_pct": pooled_pct,
        "claim_check": claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_perframe_hexagonal_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
