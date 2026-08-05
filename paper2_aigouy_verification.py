"""
paper2_aigouy_verification.py - Verifies Item 5's cross-tissue claims
(Table 7, Section 3.6, Fig 13/14: 33.67%, 3,113 cells, 26.3-38.5% per-image
range) using the REAL extraction from aigouy_hex_check.py.

WHY THIS EXISTS
----------------
Items 1-4 are now independently verified. Item 5 (cross-tissue histoblast
comparison) has never been checked. Reading aigouy_hex_check.py surfaced
two things that need resolving, not just re-running:

  1. The script hardcodes `etournay_result = 57.59` as the wing-disc
     comparison value. The manuscript itself (Table 7, and every figure
     this week) uses 57.68% -- the independently verified directed_bonds
     figure. 57.59 doesn't match anything else. This script flags the
     discrepancy explicitly rather than silently using either number.
  2. The script restricts polygon classes to 4-9 sides
     (`(pc >= 4) & (pc <= 9)`), but the manuscript's Methods states this
     analysis uses "the same biological size range (4-8 sides) used
     throughout this study." This script runs BOTH ranges and reports
     whether any 9-sided cells actually exist in the data -- i.e.
     whether this discrepancy changes anything or is cosmetic.

This reuses extract_polygon_classes() VERBATIM from aigouy_hex_check.py --
not reimplemented, so the extraction logic itself is unchanged. Only the
range filter and comparison value are made explicit and checked, rather
than silently trusted.

Run:
    conda activate ras_project
    python paper2_aigouy_verification.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    os.system(f"{sys.executable} -m pip install pillow --quiet")
    from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    os.system(f"{sys.executable} -m pip install scipy --quiet")
    from scipy import ndimage as ndi

HOME = os.path.expanduser("~")
SEG_DIR = os.path.join(HOME, "RAS_Project", "datasets", "aigouy_data", "dataset_2_segmentation")
OUTPUT_DIR = os.path.join(HOME, "RAS_Project", "results", "aigouy_validation")

# The two comparison values in play -- see docstring. Both are checked,
# neither is assumed correct.
ETOURNAY_SCRIPT_VALUE = 57.59   # hardcoded in aigouy_hex_check.py
ETOURNAY_PAPER_VALUE = 57.68    # used throughout the manuscript, independently verified


# ---- verbatim from aigouy_hex_check.py -----------------------------------
def extract_polygon_classes(mask, boundary_val, min_cell_area=200, margin=20):
    interior = (mask != boundary_val)
    labeled, num_features = ndi.label(interior)

    h, w = mask.shape
    border_labels = set()
    border_labels.update(np.unique(labeled[0:margin, :]))
    border_labels.update(np.unique(labeled[h-margin:h, :]))
    border_labels.update(np.unique(labeled[:, 0:margin]))
    border_labels.update(np.unique(labeled[:, w-margin:w]))
    border_labels.discard(0)

    sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, num_features + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes)
                    if s >= min_cell_area and (i + 1) not in border_labels]

    if len(valid_labels) == 0:
        return []

    polygon_classes = []
    struct = ndi.generate_binary_structure(2, 2)

    for lbl in valid_labels:
        cell_mask = (labeled == lbl)
        dilated = ndi.binary_dilation(cell_mask, structure=struct, iterations=3)
        touched_labels = set(np.unique(labeled[dilated])) - {0, lbl}
        touched_valid = touched_labels & set(valid_labels)
        n_sides = len(touched_valid)
        if n_sides > 0:
            polygon_classes.append(n_sides)

    return polygon_classes


def determine_boundary_polarity(seg_dir):
    sample_path = os.path.join(seg_dir, "0.tif")
    sample = np.array(Image.open(sample_path))
    val0_frac = np.mean(sample == 0)
    val255_frac = np.mean(sample == 255)
    if val0_frac < val255_frac:
        return 0
    return 255


def run_extraction(seg_dir, boundary_val, side_range):
    """Runs extraction across all 10 images, restricted to side_range
    (inclusive), returns pooled classes, per-image results."""
    all_classes = []
    per_image = []
    lo, hi = side_range
    for i in range(10):
        fpath = os.path.join(seg_dir, f"{i}.tif")
        if not os.path.exists(fpath):
            per_image.append({"image": i, "n_cells": 0, "n_hex": 0, "hex_pct": None,
                               "status": "MISSING"})
            continue
        mask = np.array(Image.open(fpath))
        pc = np.array(extract_polygon_classes(mask, boundary_val))
        valid = pc[(pc >= lo) & (pc <= hi)]
        n_cells = len(valid)
        n_hex = int(np.sum(valid == 6))
        hex_pct = (n_hex / n_cells * 100) if n_cells > 0 else None
        per_image.append({"image": i, "n_cells": int(n_cells), "n_hex": n_hex,
                           "hex_pct": hex_pct, "status": "OK"})
        all_classes.extend(valid.tolist())
    return np.array(all_classes), per_image


def check(label, claimed, actual, tol):
    if actual is None:
        return {"label": label, "status": "MISSING", "claimed": claimed, "actual": None}
    diff = abs(claimed - actual)
    status = "PASS" if diff <= tol else "FAIL"
    return {"label": label, "status": status, "claimed": claimed,
            "actual": round(actual, 4), "diff": round(diff, 4)}


if __name__ == "__main__":
    if not os.path.isdir(SEG_DIR):
        print(f"ERROR: segmentation directory not found: {SEG_DIR}")
        sys.exit(1)

    boundary_val = determine_boundary_polarity(SEG_DIR)
    print(f"Boundary value inferred: {boundary_val}\n")

    print("=" * 70)
    print("RANGE SENSITIVITY CHECK: does 4-8 vs 4-9 (script's actual filter) matter?")
    print("=" * 70)

    classes_4_8, per_image_4_8 = run_extraction(SEG_DIR, boundary_val, (4, 8))
    classes_4_9, per_image_4_9 = run_extraction(SEG_DIR, boundary_val, (4, 9))

    n_nine_sided = int(np.sum(classes_4_9 == 9))
    print(f"9-sided cells present in raw data: {n_nine_sided}")
    if n_nine_sided > 0:
        print("FLAG: the script's actual 4-9 filter includes cells the paper's stated")
        print("methods (4-8 range) would exclude. This changes total cell count and")
        print("possibly the hexagonal percentage. Methods text or script needs alignment.")
    else:
        print("No 9-sided cells found -- the 4-8 vs 4-9 discrepancy is cosmetic for this")
        print("dataset and does not change any reported number.")
    print()

    total_4_8 = len(classes_4_8)
    hex_4_8 = int(np.sum(classes_4_8 == 6))
    pct_4_8 = (hex_4_8 / total_4_8 * 100) if total_4_8 > 0 else None

    total_4_9 = len(classes_4_9)
    hex_4_9 = int(np.sum(classes_4_9 == 6))
    pct_4_9 = (hex_4_9 / total_4_9 * 100) if total_4_9 > 0 else None

    print(f"{'Range':<10}{'Total cells':>14}{'Hex cells':>12}{'Hex %':>10}")
    print("-" * 46)
    print(f"{'4-8':<10}{total_4_8:>14}{hex_4_8:>12}{f'{pct_4_8:.2f}%' if pct_4_8 else 'N/A':>10}")
    print(f"{'4-9':<10}{total_4_9:>14}{hex_4_9:>12}{f'{pct_4_9:.2f}%' if pct_4_9 else 'N/A':>10}")

    # ---- Claim check against Item 5 / Table 7 / Results 3.6 / Fig 13-14 ----
    print("\n" + "=" * 70)
    print("CLAIM CHECK: Item 5 (Introduction) / Table 7 / Results 3.6")
    print("=" * 70)
    print("(checked against the 4-8 range, matching the paper's stated methods)\n")

    hex_pcts_4_8 = [r["hex_pct"] for r in per_image_4_8 if r["hex_pct"] is not None]
    min_pct = min(hex_pcts_4_8) if hex_pcts_4_8 else None
    max_pct = max(hex_pcts_4_8) if hex_pcts_4_8 else None

    claims = [
        check("Total valid cells", 3113, total_4_8, 5),
        check("Pooled hexagonal %", 33.67, pct_4_8, 0.1),
        check("Per-image min hex %", 26.3, min_pct, 0.5),
        check("Per-image max hex %", 38.5, max_pct, 0.5),
    ]

    print(f"{'Claim':<25}{'Status':<10}{'Claimed':>10}{'Actual':>10}{'Diff':>8}")
    print("-" * 63)
    for c in claims:
        if c["status"] == "MISSING":
            print(f"{c['label']:<25}{c['status']:<10}{c['claimed']:>10}{'N/A':>10}")
        else:
            print(f"{c['label']:<25}{c['status']:<10}{c['claimed']:>10}"
                  f"{c['actual']:>10}{c['diff']:>8}")

    print("\n" + "-" * 70)
    print("COMPARISON VALUE CHECK (not a PASS/FAIL -- needs your decision)")
    print("-" * 70)
    print(f"Script's hardcoded Etournay comparison value: {ETOURNAY_SCRIPT_VALUE}%")
    print(f"Manuscript's actual, verified Etournay value:  {ETOURNAY_PAPER_VALUE}%")
    print(f"Difference: {abs(ETOURNAY_SCRIPT_VALUE - ETOURNAY_PAPER_VALUE):.2f} points")
    if pct_4_8 is not None:
        print(f"\nDeviation of Aigouy result from script's value:  "
              f"{abs(pct_4_8 - ETOURNAY_SCRIPT_VALUE):.2f} points")
        print(f"Deviation of Aigouy result from paper's value:    "
              f"{abs(pct_4_8 - ETOURNAY_PAPER_VALUE):.2f} points")
    print("\nThe manuscript should use 57.68% consistently (matching Table 7 and")
    print("every other reference this week). aigouy_hex_check.py's internal")
    print("comparison should be updated to match if it's ever rerun for reporting.")

    n_pass = sum(1 for c in claims if c["status"] == "PASS")
    n_fail = sum(1 for c in claims if c["status"] == "FAIL")
    n_missing = sum(1 for c in claims if c["status"] == "MISSING")
    print(f"\nSUMMARY: {n_pass}/{len(claims)} claims verified, {n_fail} mismatched, "
          f"{n_missing} missing")

    results = {
        "boundary_val": boundary_val,
        "n_nine_sided_cells_in_raw_data": n_nine_sided,
        "range_4_8": {"total_cells": total_4_8, "hex_cells": hex_4_8, "hex_pct": pct_4_8},
        "range_4_9": {"total_cells": total_4_9, "hex_cells": hex_4_9, "hex_pct": pct_4_9},
        "per_image_4_8": per_image_4_8,
        "per_image_min_pct": min_pct,
        "per_image_max_pct": max_pct,
        "etournay_script_value": ETOURNAY_SCRIPT_VALUE,
        "etournay_paper_value": ETOURNAY_PAPER_VALUE,
        "claim_check": claims,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "paper2_aigouy_verification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")
