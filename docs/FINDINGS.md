# Findings (Kaggle runs, 2026-11-21)

**Runs.** Both kernels ran in GPU mode (`torch 2.11.0+cu130`, Tesla T4, `cuda available: True` in the logs) with the full configuration.

| run | model | data (JetClass **validation**, not used in training) | jets | wall time |
|---|---|---|---|---|
| `part_main` | official `ParT_full` | `*_120.root`, 10 classes × 20 000 | 200 000 | 2802 s |
| `part_replica` | official `ParT_full` | `*_121.root` (independent jets), QCD/Hbb/Hcc/Tbqq × 10 000, seed 1 | 40 000 | 1286 s |
| `gds_main` | GDS-ParT v2 (shared gate, 10M, 10 ep) | as `part_main` | 200 000 | 2955 s |
| `gds_replica` | GDS-ParT v2 | as `part_replica` | 40 000 | 1381 s |

Each analysis uses a subsample of these jets:

| analysis | jets used |
|---|---|
| attention statistics | 3 000 jets/class (subsampled) |
| causal tests | 3 000 jets/class |
| activation patching | ≈3 000 H→bb̄ jets with at least one displaced track |
| probes | 2 000 jets/class, ≤ 80k particles |

All brackets are **68% bootstrap intervals over jets** (300 resamples). These intervals do not include systematic choices: the displacement threshold, the corruption scheme, or the choice of jet sample. The **main-vs-replica agreement** is therefore the better guide to what is robust. Full tables are in [`results/REPORT_kaggle.md`](../results/REPORT_kaggle.md).

"Displaced" means a charged track with |d0|/σ(d0) > 3. This is a reco-level proxy, not truth matching.

**Sanity check.** H→bb̄ vs QCD AUC is 0.99903 (main) and 0.99904 (replica) for ParT, and 0.99836 / 0.99839 for GDS-ParT. The ParT 10-way accuracy is 0.8605 over the 10 classes, consistent with the published 0.861. R50 is saturated: only 0–2 of 20k QCD jets pass, so R50 ratios in the causal tables are not informative. Use ΔAUC and R99, where 322 QCD jets pass.

---

## Robust findings (reproduce in the replica within about 0.05)

### F1. ParT's classifier reads displaced tracks and leptons preferentially, and more so in heavy-flavour jets

The table gives CLS attention share divided by population share (1 = no preference), averaged over heads.

| | C0 main / replica | C1 main / replica |
|---|---|---|
| Hbb, displaced | 1.54 [1.53,1.54] / 1.52 | 1.65 [1.64,1.66] / 1.61 |
| QCD, displaced | 1.12 [1.10,1.13] / 1.11 | 1.15 [1.12,1.18] / 1.21 |
| Hcc, displaced | 1.33 / 1.34 | 2.27 / 2.29 |
| Hbb, e/μ | 2.03 / 2.01 | 3.36 / 3.41 |
| Hbb, neutral | 0.80 / 0.80 | 0.88 / 0.88 |

- In a within-jet regression for Hbb jets, the coefficient of the "displaced" indicator on log attention is +0.22 (C0) and +0.57 [0.56, 0.58] (C1). It is the largest coefficient, ahead of ln pT (+0.23). Main and replica agree to 0.01.
- The coefficient on log(1+|d0|/σ) is negative (−0.34) once the indicator is included. Attention behaves like a **threshold** ("is displaced") rather than growing with significance.
- Leptons are read even more strongly, up to 9.2× in C1 for H→WW→qqℓν. This is expected, but it is the largest single preference.

### F2. Displaced-labelled attention keys affect performance more than matched prompt keys

Attention knockout removes particles as attention keys in every layer, ParT main / replica:

| | H→bb̄ vs QCD | H→cc̄ vs QCD | top vs QCD |
|---|---|---|---|
| displaced tracks, ΔAUC | −0.0241 [−0.0259, −0.0228] / −0.0235 | −0.021 / −0.023 | −0.009 / −0.011 |
| pT-matched prompt tracks, same number, ΔAUC | −0.0005 / −0.0007 | −0.0013 / −0.0016 | −0.0010 / −0.0007 |
| displaced, R99 ratio | 0.049 / 0.057 | | |

- The displaced-track knockout is **≈40× more damaging** than the pT-matched prompt control. For H→bb̄ it shifts the mean signal logit by −7.4, and R99 falls to about 1/20 of its value.
- Knocking out displaced tracks only in the two CLS blocks has a much smaller effect: ΔAUC −0.0006 (control 0.0000 / −0.0002). Most of the information is routed through the particle layers before the CLS reads it.
- The later `ParT_kin` negative control also loses AUC when these keys are removed, despite having no d0/dz inputs. This knockout cannot by itself isolate the contribution of impact-parameter information.

### F3. In ParT the displacement information stays on the displaced tokens

Activation patching uses H→bb̄ jets with the impact parameters of their displaced tracks zeroed. This moves log p(Hbb)/p(QCD) from 7.31 to 0.14. The table shows the fraction of that gap recovered by restoring the clean residual stream at each depth.

| position | displaced tokens only | all other tokens |
|---|---|---|
| input of L1 | 0.99 | 0.11 |
| input of L4 | 0.93 | 0.41 |
| input of CLS | **0.80** [0.80, 0.81] | **0.69** [0.69, 0.69] |

Main and replica agree to ≤ 0.01. Even after 8 layers, the displaced tokens alone carry 80% of the effect, and spreading to other particles is gradual. The CLS heads that matter most when patched are C0.7 (0.355), C0.1 (0.321) and C0.6 (0.261). These numbers are identical in the replica.

### F4. ψ(U) dominates the attention logits, but the displaced→displaced pattern comes from QK

**Logit variance.** ψ(U) accounts for 0.88 of within-row logit variance in L0, falling to 0.66 in L7. QK and ψ are weakly anti-correlated (−0.08 to −0.31). Main and replica agree to 0.01.

**Removing either term.**

| intervention | ΔAUC (H→bb̄ vs QCD) | effect on logits |
|---|---|---|
| remove ψ(U) in **all** layers | −0.0078 [−0.0085, −0.0068] (replica −0.0069) | — |
| remove ψ(U) in **one** layer | ≤ 0.0004 | signal logit still shifts by −0.2 to −0.5 |
| remove QK in all layers | −0.0046 (replica −0.0050) | — |

The bias effect is distributed across layers and redundant.

**"Secondary-vertex-like" attention.** This measures attention from displaced to displaced tracks relative to random. In Hbb jets the strongest head is **L6.h4** with enrichment 3.35 (lower CI 3.32); the replica gives 3.31 at the same head. The permutation null (displaced flags shuffled among charged tracks in each jet) gives at most 1.62 in any head. 43/64 heads exceed 1.5 at their lower CI. Recomputing the attention with only one logit term:

| attention recomputed with | max enrichment |
|---|---|
| QK only | 3.02 |
| ψ(U) only | 1.70 (about the null level; partly geometric, since b-decay tracks are collimated) |

Thus QK accounts for the head's observed grouping more than the kinematic
bias does. QK can read d0/dz, but this decomposition does **not** show that
the head uses those inputs: the later kinematics-only negative control still
shows some displaced-labelled grouping in other heads.

**Pairwise kinematics.** Many heads are strongly ordered by pairwise kinematics: Spearman ρ(attention, ln m²) = −0.82 for L0.h4, and ρ(attention, ln kT) = −0.79. Correlation with ln z is weak everywhere (|ρ| ≤ 0.19). Head-level kinematic correlations weaken with depth (layer-mean |ρ| 0.55 → 0.35).

### F5. Neighbourhood displacement becomes linearly decodable after one block, beyond a random-network control

The probe target "a displaced track lies within ΔR < 0.4 of this particle" gives these test AUCs (main / replica):

| position | trained | random init |
|---|---|---|
| embedding | 0.67 / 0.74 | 0.64 / 0.70 |
| after L0 | 0.88 / 0.92 | 0.74 / 0.77 |
| after L2 (peak) | 0.90 / 0.93 | — |
| after L7 | 0.86 / 0.91 | 0.72 / 0.77 |

The shuffled-label controls give 0.45–0.55. Trained minus random is about 0.14 at every depth after L0, in both samples. The absolute values differ between main and replica because the class mix differs (10 vs 4 classes).

The single-particle "is displaced" and "is lepton" probes are ≈1.0 at every depth. This is uninformative, because both are input features.

### F6. GDS-ParT reaches the same tagging performance through a different route

| property | GDS-ParT | ParT |
|---|---|---|
| displaced-token-only patch, recovered fraction at CLS input | 0.74 | 0.80 |
| all-other-token patch, recovered fraction at CLS input | 0.94 | 0.69 |
| best single-head context patch | L0.5 (0.61) | C0.7 (0.36) |
| CLS-only knockout of displaced tracks, ΔAUC | ≈0 (−0.0000 / −0.0001) | −0.0006 |
| remove ψ(U) everywhere, ΔAUC | −0.036 | −0.008 |
| remove ψ(U) everywhere, H→cc̄ ΔAUC | −0.13 | — |

- GDS-ParT spreads displacement information to the rest of the jet early. Two other-token patch values show this: 0.27 at L1 and 0.85 at L5. Its first-layer heads are where patching helps most.
- Its CLS barely depends on reading the displaced tokens directly.
- It still has a QK-driven displaced→displaced head (L6.h2: 3.53 / 3.48, null ≤ 1.68).
- It depends **more** on its ψ(U) bias than ParT does, even though it also has the multiplicative gates.

Every number in this table reproduces in the GDS replica within 0.01.

---

## Not robust / null results (do not claim)

- **No single head is necessary.** The largest single-head zero-ablation is ΔAUC ≈ −0.0001 for both models. This is comparable to random 5-head ablations (−0.0001 median, −0.0003 worst). The computation is redundant across heads.
- **The "top-5 heads" joint ablation does not replicate.** ParT main gives −0.0028, the replica −0.0005, and the two runs pick different heads (L0.4/L1.4/L2.2… vs L0.5/L2.0…). The heads were chosen on the same sample they were scored on (winner's curse). A held-out selection is needed before any "important heads" claim.
- **Per-layer ψ(U) removal** effects (≤ 0.0004 ΔAUC) are mostly within about 2σ of zero and change ordering between main and replica.
- **The location of the maximum displaced→displaced head depends on class.** In QCD, H→cc̄ and top jets the maximum is L2.h4 (enrichment 4–6, with only ≈1 100 QCD jets having ≥ 2 displaced tracks). In H→bb̄ it is L6.h4. L2.h4 is high in every class, so it looks like a generic "charged, collinear" head, not a b-specific one. Treat "L6.h4 is the b-tagging head" as a hypothesis.
- **R50 ratios** are uninformative because of saturation (see the sanity check above).
- **The bootstrap intervals are very narrow**, often ±0.01. They only cover jet-sampling noise. Some threshold and corruption variations have now been run on a small, overlapping subset; independent-file confirmation, prompt-resampled corruption, and another training seed remain outstanding (see M4 in `docs/PLAN.md`).

## Focused robustness controls (300 Hbb + 300 QCD validation jets)

The measured [control report](../results/robustness_val/REPORT.md) and
[JSON](../results/robustness_val/results.json) use the first 300 jets in
each `*_120.root` file, which **overlap the main run**. They are not another
independent replica. At displaced-track thresholds >2, >3, and >5, the
ParT_full displaced-key knockouts change Hbb-vs-QCD AUC by -0.0252, -0.0208,
and -0.0113; pT-matched prompt-key knockouts change it by 0.0000, -0.0002,
and -0.0001. At fixed L6.h4, full displaced-to-displaced enrichment is
3.10, 3.40, and 3.46, with QK-only enrichment 2.92, 3.16, and 3.16.
These are threshold-sensitivity checks, not held-out head selection.

The official `ParT_kin` model does not read d0/dz yet displaced-key
knockout also changes its AUC by -0.0374, -0.0311, and -0.0241. Its
L6.h4 pattern is absent (enrichment ~0.2), while another head reaches
2.36 at >3 and CLS C1 enrichment reaches 1.47. Thus the baseline's
displaced-key effect **cannot be attributed uniquely to impact parameters**,
and attention enrichment alone does not establish a secondary-vertex
representation. Qualify F2/F4 accordingly. On ParT_full at >3, d0-only
zeroing changes AUC by -0.0025, zeroing both d0/dz by -0.0235, and
per-jet prompt-mean replacement of both by -0.0210. These interventions
do not isolate dz or establish that replacements are in-distribution.
All control intervals are 68% jet-bootstrap intervals (100 resamples);
sample overlap and choice of corruption are not included.

## Next checks

1. Select heads on the main sample and score them on the replica, for the head ablation and the circuit claim.
2. Replicate threshold and `ParT_kin` controls on non-overlapping validation files; interpret the non-vanishing negative-control knockouts.
3. Run dz-only corruption and resample impact parameters from comparable prompt tracks; compare kinematic-matched interventions before attributing the effect to displacement.
5. Do path patching from L6.h4 / L2.h4 into C0.7 / C0.1 to test a two-step circuit.
