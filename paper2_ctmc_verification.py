"""
paper2_ctmc_verification.py - Verifies Item 4's CTMC claims (Table 5,
Section 3.5: leave rates, dwell times, CTMC MFPTs, stationary %,
"four small negative off-diagonal elements")

WHY THIS EXISTS
----------------
Items 1, 2, and 3 are now independently verified. Item 4 (CTMC) has never
been checked this week -- its dwell times (11.8-40.6 min), leave rates,
CTMC MFPTs, and the specific claim of "four small negative off-diagonal
rates" requiring correction are all still inherited, untested numbers.

This script:
  1. Reuses extract_polygon_classes() and build_transition_matrix()
     VERBATIM from ras_corrected_markov_analysis.py -- the same verified
     P1 matrix as every other check this week.
  2. Derives the CTMC generator via matrix logarithm, Q_c = logm(P1)/dt,
     exactly as Methods 2.7 describes.
  3. Applies the SAME regularization the paper describes: any negative
     off-diagonal rate is set to zero, with the diagonal corrected so
     each row still sums to zero (a valid generator). Counts how many
     off-diagonal elements actually needed correction -- this is a
     directly checkable claim ("four"), not just a description.
  4. Computes leave rates, dwell times, the CTMC stationary distribution,
     and CTMC MFPTs to the hexagonal state via the continuous-time
     first-passage-time equation (-Q_trans @ m = 1, Norris 1997, the
     continuous analogue of the discrete fundamental-matrix approach
     already used and verified for Item 1).
  5. Identifies each state's dominant (highest-rate) outgoing transition.
  6. Automatically checks every Table 5 claim -- PASS/FAIL -- same
     pattern as the Item 1 and Item 3 checkers.

Run:
    conda activate ras_project
    python paper2_ctmc_verification.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import logm

DB_PATH = Path.home() / "RAS_Project" / "datasets" / "example_data" / "demo" / "demo.sqlite"
OUTPUT_DIR = Path.home() / "RAS_Project" / "results" / "paper2"

STATES = [4, 5, 6, 7, 8]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
TARGET_STATE = 6
DT_MINUTES = 4.95


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


# ---- new: CTMC generator derivation + regularization ----------------------
def derive_ctmc_generator(P1, dt=DT_MINUTES):
    """
    Q_c = logm(P1) / dt (Methods 2.7). logm can return a complex result
    even for a real input if P1 has negative real eigenvalues; the
    generator of a real-valued process must be real, so we take the real
    part, consistent with standard practice for this construction.
    """
    Qc_raw = logm(P1) / dt
    if np.iscomplexobj(Qc_raw):
        max_imag = np.abs(Qc_raw.imag).max()
    else:
        max_imag = 0.0
    Qc = np.real(Qc_raw)
    return Qc, max_imag


def regularize_generator(Qc):
    """
    Sets negative off-diagonal rates to zero and corrects the diagonal so
    each row still sums to zero (a valid CTMC generator). Returns the
    regularized matrix and the count/location of corrections made --
    directly checkable against the paper's "four small negative
    off-diagonal elements" claim.
    """
    Qc_reg = Qc.copy()
    n = Qc_reg.shape[0]
    corrections = []
    for i in range(n):
        for j in range(n):
            if i != j and Qc_reg[i, j] < 0:
                corrections.append({
                    "from_state": STATES[i], "to_state": STATES[j],
                    "original_value": float(Qc_reg[i, j]),
                })
                Qc_reg[i, j] = 0.0
    for i in range(n):
        off_diag_sum = sum(Qc_reg[i, j] for j in range(n) if j != i)
        Qc_reg[i, i] = -off_diag_sum
    return Qc_reg, corrections


def ctmc_mfpt(Qc, target_state=TARGET_STATE):
    """Continuous-time first-passage-time equation: -Q_trans @ m = 1
    (Norris, 'Markov Chains', 1997) -- the continuous analogue of the
    discrete fundamental-matrix approach already verified for Item 1."""
    transient_states = [s for s in STATES if s != target_state]
    transient_idx = [STATE_IDX[s] for s in transient_states]
    Q_trans = Qc[np.ix_(transient_idx, transient_idx)]
    m = np.linalg.solve(-Q_trans, np.ones(len(transient_states)))
    return {transient_states[i]: float(m[i]) for i in range(len(transient_states))}


def check(label, claimed, actual, tol):
    diff = abs(claimed - actual)
    status = "PASS" if diff <= tol else "FAIL"
    return {"label": label, "status": status, "claimed": claimed,
            "actual": round(actual, 4), "diff": round(diff, 4)}


if __name__ == "__main__":
    print("Loading polygon classes from " + str(DB_PATH) + " ...")
    frames = extract_polygon_classes(DB_PATH)
    print("Frames: " + str(len(frames)))

    counts, n_transitions = build_transition_matrix(frames)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P1 = counts / row_sums
    pi1_discrete = stationary_distribution(P1)
    print("P1 built from " + str(n_transitions) + " transitions. Discrete hexagonal stationary: "
          + f"{pi1_discrete[STATE_IDX[6]]*100:.2f}% (sanity check: should be ~58.47%)\n")

    print("Deriving CTMC generator Q_c = logm(P1) / dt ...")
    Qc_raw, max_imag = derive_ctmc_generator(P1)
    print(f"Max imaginary component discarded: {max_imag:.2e}")

    n_negative_before = int(np.sum((Qc_raw - np.diag(np.diag(Qc_raw))) < 0))
    print(f"Negative off-diagonal elements before regularization: {n_negative_before}")

    Qc, corrections = regularize_generator(Qc_raw)
    print(f"Corrected {len(corrections)} negative off-diagonal element(s):")
    for c in corrections:
        print(f"  {c['from_state']}->{c['to_state']}: {c['original_value']:.5f} -> 0")

    leave_rates = {s: float(-Qc[STATE_IDX[s], STATE_IDX[s]]) for s in STATES}
    dwell_times = {s: 1.0 / leave_rates[s] for s in STATES}

    dominant = {}
    for s in STATES:
        i = STATE_IDX[s]
        row = Qc[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        dominant[s] = STATES[j]

    eigvals, eigvecs = np.linalg.eig(Qc.T)
    idx = np.argmin(np.abs(eigvals))
    pi_ctmc = np.real(eigvecs[:, idx])
    pi_ctmc = np.abs(pi_ctmc) / np.abs(pi_ctmc).sum()

    mfpt_ctmc = ctmc_mfpt(Qc)

    print(f"\n{'State':<10}{'Leave Rate':>12}{'Dwell (min)':>13}{'CTMC MFPT':>12}{'Dominant':>10}")
    print("-" * 60)
    for s in STATES:
        mfpt_str = f"{mfpt_ctmc[s]:.1f}" if s != TARGET_STATE else "--- (target)"
        print(f"{s}-sided{'':<3}{leave_rates[s]:>12.4f}{dwell_times[s]:>13.2f}"
              f"{mfpt_str:>12}{'-> ' + str(dominant[s]):>10}")

    print(f"\nCTMC stationary hexagonal %: {pi_ctmc[STATE_IDX[6]]*100:.2f}% "
          f"(discrete: {pi1_discrete[STATE_IDX[6]]*100:.2f}%)")

    print("\n" + "=" * 70)
    print("CLAIM CHECK: Item 4 (Introduction) / Table 5 / Results 3.5")
    print("=" * 70)

    TOL_RATE = 0.001
    TOL_MIN = 0.15
    claims = [
        check("Negative off-diagonal count", 4, len(corrections), 0),
        check("4-sided leave rate", 0.0846, leave_rates[4], TOL_RATE),
        check("4-sided dwell time (min)", 11.82, dwell_times[4], TOL_MIN),
        check("5-sided leave rate", 0.0406, leave_rates[5], TOL_RATE),
        check("5-sided dwell time (min)", 24.62, dwell_times[5], TOL_MIN),
        check("6-sided leave rate", 0.0247, leave_rates[6], TOL_RATE),
        check("6-sided dwell time (min)", 40.55, dwell_times[6], TOL_MIN),
        check("7-sided leave rate", 0.0416, leave_rates[7], TOL_RATE),
        check("7-sided dwell time (min)", 24.03, dwell_times[7], TOL_MIN),
        check("8-sided leave rate", 0.0740, leave_rates[8], TOL_RATE),
        check("8-sided dwell time (min)", 13.51, dwell_times[8], TOL_MIN),
        check("4-sided CTMC MFPT (min)", 43.0, mfpt_ctmc[4], TOL_MIN),
        check("5-sided CTMC MFPT (min)", 33.2, mfpt_ctmc[5], TOL_MIN),
        check("7-sided CTMC MFPT (min)", 33.2, mfpt_ctmc[7], TOL_MIN),
        check("8-sided CTMC MFPT (min)", 45.4, mfpt_ctmc[8], TOL_MIN),
        check("CTMC stationary hexagonal %", 58.46, pi_ctmc[STATE_IDX[6]] * 100, 0.05),
    ]
    dominant_expected = {4: 5, 5: 6, 6: 5, 7: 6, 8: 7}
    for s in STATES:
        match = dominant[s] == dominant_expected[s]
        claims.append({"label": f"{s}-sided dominant transition",
                        "status": "PASS" if match else "FAIL",
                        "claimed": dominant_expected[s], "actual": dominant[s], "diff": "n/a"})

    print(f"\n{'Claim':<32}{'Status':<8}{'Claimed':>10}{'Actual':>10}{'Diff':>8}")
    print("-" * 68)
    for c in claims:
        print(f"{c['label']:<32}{c['status']:<8}{str(c['claimed']):>10}"
              f"{str(c['actual']):>10}{str(c['diff']):>8}")

    n_pass = sum(1 for c in claims if c["status"] == "PASS")
    n_fail = sum(1 for c in claims if c["status"] == "FAIL")
    print(f"\nSUMMARY: {n_pass}/{len(claims)} claims verified, {n_fail} mismatched")
    if n_fail > 0:
        print("\nDo not treat FAILed claims as correct until reconciled.")

    results = {
        "n_frames": len(frames), "n_transitions": n_transitions,
        "max_imaginary_component_discarded": max_imag,
        "n_negative_offdiagonal_corrected": len(corrections),
        "corrections": corrections,
        "leave_rates": leave_rates,
        "dwell_times_minutes": dwell_times,
        "ctmc_mfpt_minutes": mfpt_ctmc,
        "dominant_transitions": dominant,
        "ctmc_stationary_hexagonal_pct": round(float(pi_ctmc[STATE_IDX[6]]) * 100, 4),
        "discrete_stationary_hexagonal_pct": round(float(pi1_discrete[STATE_IDX[6]]) * 100, 4),
        "claim_check": claims,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "paper2_ctmc_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to: " + str(out_path))
