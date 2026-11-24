---
title: ParT Interpretability
---

# What does ParT learn about jet constituents?

This project studies the official pretrained **Particle Transformer (ParT_full)** and
compares it with **GDS-ParT** on JetClass jets. Instrumented forward passes separate
attention content (QK) from pairwise-interaction bias and test interventions on
displaced-track proxies. Code, analysis settings, baseline outputs, and independent
replica results are available in the
[repository](https://github.com/Avinashistoocool/ParT-Interpretability).

## What has been measured

The existing runs used JetClass **validation** files on Kaggle GPUs, with a
200,000-jet main sample and an independent 40,000-jet replica for each model.
For official ParT_full, removing displaced tracks as attention keys reduced
Hbb-vs-QCD AUC by 0.0241 in the main run and 0.0235 in the replica; removing
pT-matched prompt tracks reduced it by 0.0005 and 0.0007. In Hbb jets, a
displaced-to-displaced attention pattern reached enrichment 3.35 (3.31 in
the replica) in one head. Recomputing attention with QK alone gave a maximum
enrichment of 3.02, compared with 1.70 using only pairwise bias.

These are **model-level observations**, not identified decay vertices or evidence
that a particular head is necessary. A "displaced track" here is a charged
constituent with reconstructed absolute transverse impact parameter significance
above 3, not a truth-matched b-decay product. No single head had a reproducible
causal ablation effect. See the [full findings](FINDINGS.html) for uncertainty
intervals, GDS-ParT comparisons, null results and caveats, and the
[baseline tables](https://github.com/Avinashistoocool/ParT-Interpretability/blob/main/results/REPORT_kaggle.md)
for the measured outputs.

## New sensitivity checks

On a smaller, **overlapping** validation subset (300 Hbb + 300 QCD jets),
ParT_full displaced-key knockout changes Hbb-vs-QCD AUC by -0.0252,
-0.0208, and -0.0113 when the displaced-track threshold is >2, >3, and
>5. At >3, zeroing both d0/dz changes AUC by -0.0235; replacing both
with the jet's prompt-track mean changes it by -0.0210, whereas zeroing
only d0 changes it by -0.0025. The official kinematics-only model, which
cannot read d0/dz, also suffers displaced-key knockouts (delta AUC -0.0311
at >3). Thus this knockout does **not** isolate impact-parameter information.
Its L6.h4 attention pattern differs from ParT_full, but other heads still
prefer labelled displaced tracks. See the [measured control report](https://github.com/Avinashistoocool/ParT-Interpretability/blob/avinashistoocool-interpretability-robustness/results/robustness_val/REPORT.md)
and [JSON](https://github.com/Avinashistoocool/ParT-Interpretability/blob/avinashistoocool-interpretability-robustness/results/robustness_val/results.json).

## Reproduce and extend

The [README](https://github.com/Avinashistoocool/ParT-Interpretability#usage)
documents installation, checkpoints, data preparation, tests, and the original
analysis command. The [plan](PLAN.html) separates completed main/replica runs and the smaller
sensitivity checks from remaining independent-file, corruption and circuit
checks. The new control subset overlaps the main sample, so it is not an
independent replication.
