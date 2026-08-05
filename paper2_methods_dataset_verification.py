"""
paper2_methods_dataset_verification.py - Verifies Methods 2.1 (Dataset) and
2.2 (Polygon Class Extraction) claims that were never independently checked
this week:

  1. "database-verified mean interval of 4.95 minutes" -- every script this
     week hardcoded DT_MINUTES=4.95 without ever querying frames.time_sec
     directly. This script does that query.
  2. "restricted to polygon classes 4-8, accounting for >99% of observed
     cells" -- every extraction script pre-filtered to 4-8 before counting,
     so nothing has checked what fraction of the RAW, unfiltered population
     actually falls in that range. This script runs the extraction WITHOUT
     the filter and checks.
  3. "710 unique cells" -- never computed by any script this week (distinct
     from the 36,694 cell-FRAME observations already checked). This script
     counts distinct cell_id values in the filtered dataset.

IMPORTANT: this script does NOT assume the frames table's schema. It
inspects it first and prints what it finds. If the time_sec query fails
because the actual column/table name differs from what Methods 2.1
claims, that is itself a finding to report, not something to silently
work around by guessing a different column name.

Run:
    conda activate ras_project
    python paper2_methods_dataset_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"
OUTPUT_DIR = Path.home() / "RAS_Project" / "results" / "paper2"

STATES_CLAIMED = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES_CLAIMED)}


# ---- Section 1: schema inspection (no assumptions) -----------------------
def inspect_schema(db_path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables found: {tables}\n")

    schema_info = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [row[1] for row in cur.fetchall()]
        schema_info[t] = cols
        if t.lower() in ("frames", "frame"):
            print(f"'{t}' table columns: {cols}")
    conn.close()
    return schema_info


# ---- Section 2: frame interval, queried directly, no assumption ----------
def verify_frame_interval(db_path, schema_info):
    """
    Looks for a frames-like table with a time-related column and computes
    the ACTUAL mean inter-frame interval. Does not assume table/column
    names beyond what Methods 2.1 claims ('frames' table, 'time_sec'
    column) -- but reports clearly if that assumption is wrong rather
    than silently substituting a different column.
    """
    candidate_tables = [t for t in schema_info if t.lower() in ("frames", "frame")]
    if not candidate_tables:
        return {"status": "TABLE_NOT_FOUND", "detail": f"No 'frames' table found. "
                "Tables present: " + str(list(schema_info.keys()))}

    table = candidate_tables[0]
    cols = schema_info[table]
    time_col_candidates = [c for c in cols if "time" in c.lower()]
    if not time_col_candidates:
        return {"status": "TIME_COLUMN_NOT_FOUND",
                "detail": f"'{table}' table has no column containing 'time'. "
                f"Columns present: {cols}"}

    time_col = time_col_candidates[0]
    frame_col_candidates = [c for c in cols if c.lower() in ("frame", "frame_nb", "frame_id")]
    frame_col = frame_col_candidates[0] if frame_col_candidates else None

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    if frame_col:
        cur.execute(f"SELECT {frame_col}, {time_col} FROM {table} ORDER BY {frame_col}")
    else:
        cur.execute(f"SELECT {time_col} FROM {table}")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 2:
        return {"status": "INSUFFICIENT_DATA",
                "detail": f"Only {len(rows)} row(s) in '{table}.{time_col}'."}

    times = np.array([r[-1] for r in rows], dtype=float)
    times_sorted = np.sort(times)
    diffs = np.diff(times_sorted)
    mean_interval_sec = float(np.mean(diffs))
    mean_interval_min = mean_interval_sec / 60.0
    std_interval_min = float(np.std(diffs)) / 60.0

    return {
        "status": "OK", "table": table, "time_column": time_col,
        "frame_column": frame_col, "n_rows": len(rows),
        "n_intervals": len(diffs),
        "mean_interval_minutes": mean_interval_min,
        "std_interval_minutes": std_interval_min,
        "min_interval_minutes": float(diffs.min()) / 60.0,
        "max_interval_minutes": float(diffs.max()) / 60.0,
    }


# ---- Section 3: unfiltered extraction, to check the >99% claim -----------
def extract_polygon_classes_unfiltered(db_path):
    """Same query as the verified pipeline, but WITHOUT restricting to
    STATES_CLAIMED -- returns every observed num_sides value."""
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

    frames_filtered = defaultdict(dict)
    all_num_sides = []
    for frame, cell_id, num_sides in rows:
        all_num_sides.append(num_sides)
        if STATES_CLAIMED[0] <= num_sides <= STATES_CLAIMED[-1]:
            frames_filtered[frame][cell_id] = num_sides

    return frames_filtered, np.array(all_num_sides)


def check(label, claimed, actual, tol, direction="abs"):
    if actual is None:
        return {"label": label, "status": "MISSING", "claimed": claimed, "actual": None}
    if direction == "gte":
        status = "PASS" if actual >= claimed else "FAIL"
        diff = actual - claimed
    else:
        diff = abs(claimed - actual)
        status = "PASS" if diff <= tol else "FAIL"
    return {"label": label, "status": status, "claimed": claimed,
            "actual": round(actual, 4) if isinstance(actual, float) else actual,
            "diff": round(diff, 4) if isinstance(diff, float) else diff}


if __name__ == "__main__":
    print(f"Inspecting schema of {DB_PATH} ...\n")
    schema_info = inspect_schema(DB_PATH)

    print("=" * 70)
    print("CLAIM 1: Frame interval (claimed: 4.95 minutes, 'database-verified')")
    print("=" * 70)
    interval_result = verify_frame_interval(DB_PATH, schema_info)
    if interval_result["status"] == "OK":
        print(f"Source: {interval_result['table']}.{interval_result['time_column']} "
              f"({interval_result['n_rows']} rows, {interval_result['n_intervals']} intervals)")
        print(f"Mean interval : {interval_result['mean_interval_minutes']:.4f} min")
        print(f"Std dev       : {interval_result['std_interval_minutes']:.4f} min")
        print(f"Min / Max     : {interval_result['min_interval_minutes']:.4f} / "
              f"{interval_result['max_interval_minutes']:.4f} min")
        interval_claim = check("Mean frame interval (min)", 4.95,
                                interval_result["mean_interval_minutes"], 0.01)
    else:
        print(f"COULD NOT VERIFY: {interval_result['status']}")
        print(f"  {interval_result['detail']}")
        print("  This means the '4.95-minute, database-verified' claim in Methods 2.1")
        print("  could not be confirmed by this script and needs manual investigation")
        print("  before it can be trusted -- do not assume 4.95 is correct by default.")
        interval_claim = {"label": "Mean frame interval (min)", "status": "COULD_NOT_VERIFY",
                           "claimed": 4.95, "actual": None}

    print("\n" + "=" * 70)
    print("CLAIM 2: '>99% of observed cells' fall within polygon classes 4-8")
    print("=" * 70)
    frames_filtered, all_num_sides = extract_polygon_classes_unfiltered(DB_PATH)
    total_raw = len(all_num_sides)
    in_range = np.sum((all_num_sides >= 4) & (all_num_sides <= 8))
    pct_in_range = (in_range / total_raw * 100) if total_raw > 0 else None

    print(f"Total raw cell-frame observations (unfiltered): {total_raw}")
    print(f"Within 4-8 range: {in_range} ({pct_in_range:.4f}%)")
    unique_raw_sides = sorted(np.unique(all_num_sides).tolist())
    print(f"All observed polygon classes (unfiltered): {unique_raw_sides}")
    outside_range = total_raw - in_range
    print(f"Outside 4-8 range: {outside_range} cell-frame observations")
    range_claim = check("Percent within 4-8 range", 99.0, pct_in_range, None, direction="gte")

    print("\n" + "=" * 70)
    print("CLAIM 3: '710 unique cells' (within the 4-8-filtered dataset)")
    print("=" * 70)
    all_cell_ids = set()
    total_cell_frame_obs = 0
    for frame, cell_dict in frames_filtered.items():
        all_cell_ids.update(cell_dict.keys())
        total_cell_frame_obs += len(cell_dict)
    n_unique_cells = len(all_cell_ids)
    print(f"Unique cell_id values (4-8 filtered): {n_unique_cells}")
    print(f"Total cell-frame observations (4-8 filtered): {total_cell_frame_obs} "
          f"(sanity check: should be ~36,694)")
    unique_claim = check("Unique cell count", 710, n_unique_cells, 5)
    total_obs_claim = check("Total cell-frame observations (sanity check)", 36694,
                             total_cell_frame_obs, 5)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    claims = [interval_claim, range_claim, unique_claim, total_obs_claim]
    for c in claims:
        status = c["status"]
        print(f"  {c['label']:<45}{status:<18}"
              f"claimed={c['claimed']}  actual={c.get('actual', 'N/A')}")

    results = {
        "schema_info": schema_info,
        "frame_interval": interval_result,
        "range_check": {
            "total_raw": int(total_raw), "in_range": int(in_range),
            "pct_in_range": pct_in_range,
            "all_observed_polygon_classes": unique_raw_sides,
        },
        "unique_cells": {"n_unique_cells": n_unique_cells,
                          "total_cell_frame_obs": total_cell_frame_obs},
        "claim_check": claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_methods_dataset_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved full results to: {out_path}")
