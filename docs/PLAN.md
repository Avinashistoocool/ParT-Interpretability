# Plan: ParT interpretability → ML4PS workshop paper

## Target venue

- **ML4PS 2026** (Machine Learning and the Physical Sciences), an independent workshop held next to NeurIPS 2026 in Atlanta on **11 Dec 2026**. The [website](https://ml4physicalsciences.github.io/2026/) was checked on 2026-11-21. It lists the paper submission deadline as **19 Sep 2026 (AoE, extended)**, which **has already passed**. The camera-ready date on the page still says "November 14, 2025", which looks like a stale template. *Verify* whether there are any late/2027 calls.
- Realistic targets therefore: **(a)** an arXiv note as soon as the Kaggle runs and the replica agree; **(b)** ML4PS 2027 (deadline expected around Aug-Sep 2027, *verify*); **(c)** alternatives to *verify*: the ML4Jets workshop, the EurIPS / ICLR 2027 AI4Science-type workshops, or *Machine Learning: Science and Technology* as a short paper.
- Format: ML4PS asks for 4 pages plus references in its own LaTeX style. `paper/main.tex` uses a neutral 4-page skeleton. Swap in the official `.sty` when the call opens.

## Milestones

| # | milestone | status | exit criterion |
|---|---|---|---|
| M0 | Instrumented ParT/GDS-ParT, exactness tests | **done** | `pytest` passes; instrumented logits = original; qk+bias reconstructs logits |
| M1 | Local pilot on 4×5k test jets (ParT_full + GDS-ParT) | **done** | `results/REPORT_local.md`, figures |
| M2 | Kaggle main runs: 10 classes × 20k val jets, both models | pushed | `results_<model>.tgz` downloaded to `results/kaggle_*` |
| M3 | Replica (independent files, new seed) agrees with M2 | pending | headline numbers inside each other's 68% CIs |
| M4 | Robustness: displacement threshold 2/3/5σ; the d0-only vs d0+dz corruption; mean- vs zero-ablation; ParT_kin (no d0 inputs) as a negative control, where displaced-track structure must vanish | todo | table in appendix |
| M5 | Circuit write-up: trace top heads (L→C) for H→bb̄; attention-path / head-to-head patching | todo | 1 schematic figure |
| M6 | Paper draft (4 pp) + internal review (Prof. Iyer) | todo | PDF |
| M7 | arXiv submission; code tag + Zenodo DOI | todo | arXiv ID |

## Pre-registered claims to test (fill from M2/M3)

1. The CLS token over-weights displaced tracks in H→bb̄ (enrichment > 1 with the CI excluding 1). The pT-matched prompt-track knockout hurts much less than the displaced-track knockout.
2. At least one particle head attends displaced→displaced above the permutation null. This "SV-like" head is driven by QK (content), not ψ(U), because ψ(U) has no access to d0.
3. ψ(U) dominates attention-logit variance in early layers, and QK dominates in deeper layers.
4. Displacement is linearly decodable at every depth (it is an input). "Displaced track within ΔR<0.4" becomes more decodable with depth, and more than in the random-init control. That would be evidence of neighbourhood aggregation.
5. GDS-ParT shows the same or a different mechanism (comparison paragraph).

Any claim that fails is reported as such.

## Compute

Local laptop (RTX 4050 6 GB, 16 GB RAM): pilots only, ≤ 5k jets/class. Everything heavier runs on Kaggle (P100/T4) via `scripts/push_kaggle.ps1`.
