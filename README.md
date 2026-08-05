# Multi-Framework Markov Chain Analysis of Hexagonal Dominance in *Drosophila* Epithelium

Code accompanying the manuscript:

> J. Adger and N. Naheed, "Multi-Framework Markov Chain Analysis of Hexagonal
> Dominance in *Drosophila* Epithelium," submitted to *PLOS Computational
> Biology*, 2026.

This repository contains the analysis pipeline and independent verification
scripts behind every quantitative claim in the paper. It does **not** contain
raw dataset files — see [Data Availability](#data-availability) below.

## Overview

We build a first-order Markov chain model of epithelial cell-polygon
transitions from TissueMiner's `directed_bonds` structural data, extend it
with three formal statistical frameworks (second-order memory, spatial
neighbor-coupling via the Aboav-Weaire law, and a continuous-time
parameterization), and validate the results across four independent
*Drosophila* wing datasets plus an independent cross-tissue comparison. Every
formal statistical test (Anderson-Goodman order tests, negative-control
calibration simulations) is included and independently reproducible, not
just reported.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── analysis/
│   ├── ras_corrected_markov_analysis.py
│   │       First-order Markov model. The corrected, directed_bonds-based
│   │       pipeline (supersedes an earlier RGB-image-based extraction that
│   │       produced meaningless results -- see paper Methods/Limitations).
│   │       Produces the first-order stationary hexagonal percentages for
│   │       all four datasets (demo, WT_1, WT_2, WT_3).
│   ├── paper2_second_order_corrected.py
│   │       Second-order model: full vs. well-sampled-pairs-only stationary
│   │       distributions, the Anderson-Goodman order test (first- vs.
│   │       second-order), a derived (not asserted) sample-size threshold,
│   │       and a negative-control calibration check. Produces the results
│   │       behind Aim 2a / the second-order Results and Discussion sections.
│   └── paper2_spatial_corrected.py
│           Spatial neighbor-coupled model (Aboav-Weaire law). Environment-
│           conditioned transition matrices (low/hex/high neighbor
│           environments), a formal Anderson-Goodman test of whether
│           environment predicts transitions beyond the cell's own current
│           state, and a matching negative-control calibration check.
├── verification/
│   ├── paper2_methods_dataset_verification.py
│   │       Verifies Methods 2.1/2.2 dataset description: frame count,
│   │       frame interval, cell/observation counts.
│   ├── paper2_mfpt_verification.py
│   │       Verifies the Mean First Passage Time claims (Table 3) and the
│   │       347-minute biological-timescale comparison.
│   ├── paper2_ctmc_verification.py
│   │       Verifies the continuous-time Markov chain MFPT values (Table 5)
│   │       and generator-matrix properties (row-sums, non-negative
│   │       off-diagonals).
│   ├── paper2_triplet_cells_verification.py
│   │       Verifies the distinct-cell and triplet-count claims underlying
│   │       the second-order model (Methods 2.5).
│   ├── paper2_methods_2_6_2_7_verification.py
│   │       Verifies Methods 2.6/2.7 (spatial coupling and CTMC setup):
│   │       neighbor-pair counts, threshold completeness, generator-matrix
│   │       mathematical invariants.
│   ├── paper2_perframe_hexagonal_verification.py
│   │       Verifies the per-frame hexagonal-percentage range (min/mean/max)
│   │       reported in Results 3.1.
│   ├── paper2_aboav_correlation_verification.py
│   │       Independently reproduces the Aboav correlation (r = -0.611)
│   │       cited in Discussion 4.2.
│   ├── paper2_environment_transition_counts_verification.py
│   │       Verifies the neighbor-environment transition counts in
│   │       Limitations (distinguishes cell-frame classification counts
│   │       from true transition counts -- this script caught and fixed a
│   │       real mislabeling in an earlier draft).
│   ├── paper2_aigouy_verification.py
│   │       Verifies the cross-tissue histoblast nest comparison (Results
│   │       3.6): polygon-class filter range, cell count, hexagonal
│   │       percentage.
│   └── paper2_memory_effect_extraction.py
│           Extracts and verifies the second-order memory-effect deviation
│           values discussed in Discussion 4.2.
```

**Two scripts from this project's history are deliberately excluded:**
`drosophila_markov.py` (the original RGB-image-based extraction, confirmed
to produce meaningless results due to a color-vs-label decoding error) and
`week6_polygon_distribution.py` (uses a third, inconsistent data source and
reflects the project's debunked pre-correction framing). Their role in the
project's history is described in the paper's Methods and Limitations
sections rather than reproduced here, to avoid presenting superseded or
broken code as part of the paper's actual methodology.

## Environment

All results in this paper were produced locally on an Ubuntu workstation, in
a conda environment named `ras_project` (Python 3.10). `requirements.txt`
lists the Python dependencies; no R installation is required for any script
in this repository (all analysis reads TissueMiner's SQLite `directed_bonds`
table directly via Python's built-in `sqlite3` module).

## Reproducing the results

1. **Set up the environment.**
   ```bash
   conda activate ras_project
   pip install -r requirements.txt
   ```

2. **Obtain the datasets.** Raw data is not included in this repository —
   see [Data Availability](#data-availability). Update the `DATASETS` path
   dictionary at the top of each script in `analysis/` if your local paths
   differ from the defaults (`~/RAS_Project/datasets/...`,
   `~/TissueMiner_WT_Data/...`).

3. **Run the core analysis pipeline** (each script is independently
   runnable and saves its own results JSON):
   ```bash
   python analysis/ras_corrected_markov_analysis.py
   python analysis/paper2_second_order_corrected.py
   python analysis/paper2_spatial_corrected.py
   ```

4. **Run the verification scripts** to independently confirm the specific
   numbers reported in the paper (each prints a PASS/FAIL comparison
   against the claimed value and saves its own results JSON):
   ```bash
   python verification/paper2_methods_dataset_verification.py
   python verification/paper2_mfpt_verification.py
   python verification/paper2_ctmc_verification.py
   python verification/paper2_triplet_cells_verification.py
   python verification/paper2_methods_2_6_2_7_verification.py
   python verification/paper2_perframe_hexagonal_verification.py
   python verification/paper2_aboav_correlation_verification.py
   python verification/paper2_environment_transition_counts_verification.py
   python verification/paper2_aigouy_verification.py
   python verification/paper2_memory_effect_extraction.py
   ```

Each verification script is self-contained and states in its own docstring
exactly which paper claim it checks.

## Data Availability

The *Drosophila* pupal-wing epithelium demo dataset and three independent
wild-type replicates (WT_1, WT_2, WT_3) are available from the original
[TissueMiner project](https://github.com/mpicbg-scicomp/tissue_miner)
(Etournay et al., 2016, *eLife*). The *Drosophila* histoblast nest dataset
was provided by Dr. Benoît Aigouy (IBDM, Marseille) through direct
collaboration and is **not redistributed here**; researchers seeking access
should contact Dr. Aigouy directly.

## Citation

If you use this code, please cite:

```bibtex
@article{adger2026markov,
  title   = {Multi-Framework Markov Chain Analysis of Hexagonal Dominance
             in Drosophila Epithelium},
  author  = {Adger, Jada and Naheed, Naima},
  journal = {PLOS Computational Biology},
  year    = {2026},
  note    = {Code: [Zenodo DOI to be added]}
}
```

## License

[Choose a license when creating the repository — MIT or BSD-3-Clause are
common, permissive choices for academic research code.]

## Acknowledgments

This work was supported by the Resilient Autonomous Systems (RAS) Program
through the Office of Naval Research (ONR). We thank Dr. Raphaël Etournay
(Institut Pasteur) for the TissueMiner demo dataset and Dr. Benoît Aigouy
(IBDM, Marseille) for the TissueAnalyzer and EPySeg tools and for sharing
the histoblast nest segmentation dataset.

## Contact

Naima Naheed — naima.naheed@benedict.edu
Department of Computer Science and Engineering, Benedict College
