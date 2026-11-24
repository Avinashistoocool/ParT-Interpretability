# Focused ParT robustness controls (JetClass validation subset)

**Run.** Official `ParT_full` and `ParT_kin` checkpoints, first 300 jets each
from `HToBB_120.root` and `ZJetsToNuNu_120.root` (600 total), seed 17,
100 jet-bootstrap resamples, RTX 4050 GPU. The two files are validation data
and **overlap the Kaggle main run**, which also used `*_120.root`: these are
paired sensitivity checks, not an independent replication. Results are in
[`results.json`](results.json); intervals below are 68% bootstrap intervals
over jets, not systematic uncertainties. Baseline Hbb-vs-QCD AUC on this
subset is 0.99948 for ParT_full and 0.98648 for ParT_kin. R50 is saturated
on this small sample and is not interpreted.

| Threshold, abs(d0)/sigma(d0) | Displaced charged tracks (both classes) | ParT_full displaced-key delta AUC [68% interval] | ParT_full pT-matched prompt-key delta AUC | ParT_kin displaced-key delta AUC [68% interval] | ParT_kin pT-matched prompt-key delta AUC |
|---|---:|---:|---:|---:|---:|
| >2 | 3229 | -0.02522 [-0.03055, -0.02015] | 0.00000 | -0.03737 [-0.04272, -0.03228] | -0.00513 |
| >3 | 2711 | -0.02083 [-0.02553, -0.01650] | -0.00017 | -0.03114 [-0.03639, -0.02519] | -0.00587 |
| >5 | 2393 | -0.01127 [-0.01526, -0.00862] | -0.00011 | -0.02406 [-0.02905, -0.02054] | -0.00704 |

The threshold changes **which and how many keys are removed**. Knockout
costs remain substantial at all three cuts for both models, but the
kinematics-only model has an effect even without d0/dz inputs. Therefore key
knockout is not evidence that impact-parameter information specifically
causes the performance change: a displaced-track label also selects
kinematically informative particles. The matched control does not eliminate
all such differences.

| ParT_full impact-parameter corruption at >3 | Delta AUC [68% interval] |
|---|---:|
| Zero d0 feature on selected tracks | -0.00247 [-0.00318, -0.00153] |
| Zero d0 and dz features on selected tracks | -0.02348 [-0.02854, -0.01795] |
| Replace both features with each jet's prompt-charged mean | -0.02098 [-0.02527, -0.01597] |

The zero-both and prompt-mean outcomes are similar on this subset; d0-only
corruption is much smaller. This does **not** isolate dz's effect, because
dz-only corruption was not run. No jet with selected displaced tracks lacked
a prompt-charged reference (0 unchanged jets). The replacement is a mean of
the model's transformed feature values, not a resampling of physically
matched prompt tracks; it does not establish an in-distribution intervention.

For Hbb jets, ParT_full CLS C1 displaced attention/population enrichment is
1.68, 1.70, and 1.65 at thresholds 2, 3, and 5. At fixed L6.h4,
displaced-to-displaced enrichment is 3.10, 3.40, and 3.46 with full
attention; 2.92, 3.16, and 3.16 with QK only; 1.59, 1.67, and 1.66
with pairwise bias only. The corresponding per-jet charged-track
permutation null at this head is 1.60, 1.60, and 1.62. The >3 estimate
uses 298 Hbb jets with at least two selected tracks. These attention
measurements are correlational, and the head was preselected from the
overlapping baseline sample. In ParT_kin, L6.h4 full enrichment is only
0.21, 0.20, and 0.20, but its **maximum across heads** is 2.24, 2.36,
and 2.32; its C1 enrichment is 1.43, 1.47, and 1.45. Thus the original
head-specific pattern does not carry over, while some labelled-track
attention persists without impact parameters. Head maxima are exploratory,
not held-out significance tests.

**Reproduce** after downloading the official checkpoints and the two
validation ROOT files as described in the README:

```powershell
python scripts\run_robustness.py --ckpt checkpoints\ParT_full.pt --kin_ckpt checkpoints\ParT_kin.pt `
  --data_dir data\JetClass\val --n_per_class 300 --batch_size 16 --n_boot 100 `
  --seed 17 --device cuda --out results\robustness_val
```

SHA-256 of input files (Hbb ROOT, QCD ROOT, full checkpoint, kin checkpoint):

```text
CFF7FED8490A3BA2C3A971A3017D2D7922C4E39E854432AAD04D5FFE241FBC60
8C507379E286AF15BA410B88448D0CF626D1888542190C6B1DF31A9113C53FEA
61E752F80D7C237D4B18B97705DF416A8518DD9E3D5A78A8BBDEEBADD787FEC0
A123C697EB0F942D7CECB7D5E8892B21B77B3B2D37C93120ED4F78C76E456C97
```

**Remaining controls.** Use a non-overlapping file/seed to check whether
these trends replicate; add dz-only and prompt-track-resampled corruption;
test the joint dependence on d0/dz and constituent kinematics. The saved
200k-jet main and independent 40k-jet replica findings remain documented
separately in [`docs/FINDINGS.md`](../../docs/FINDINGS.md).
