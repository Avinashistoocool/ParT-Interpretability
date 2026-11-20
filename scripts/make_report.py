"""Write a plain-language REPORT.md from one or more results.json files and,
for two models, a comparison figure.

    python scripts/make_report.py results/local_part_test5k results/local_gds_test5k --out results/REPORT_local.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from partinterp import plots  # noqa: E402


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}g}"


def section(r, d):
    s, b = r["signal"], r["background"]
    key = f"{s}_vs_{b}"
    L = []
    perf = r["performance"]
    p = perf["pairs"].get(key, {})
    L.append(f"## {r['model']}  (`{d}`)\n")
    L.append(f"Jets: {perf['n_jets']} ({', '.join(r['classes'])}); 10-way accuracy restricted to these classes: "
             f"{perf['accuracy_10way']:.4f}. {key}: AUC {f(p.get('auc'), 5)}, "
             f"R50 {f(p.get('R50'))} ({p.get('n_bkg_pass_50')} bkg jets pass), "
             f"R90 {f(p.get('R90'))} ({p.get('n_bkg_pass_90')} pass)."
             + (f" R99 {f(p.get('R99'))} ({p.get('n_bkg_pass_99')} pass)." if "R99" in p else "") + "\n")

    L.append("### 1. Which particles does the CLS token read?\n")
    L.append("Attention share / population share (1 = no preference), head-averaged, 68% jet-bootstrap CI.\n")
    L.append("| class | block | displaced | e/mu | prompt ch. | neutral | 3 leading |\n|---|---|---|---|---|---|---|")
    for c, e in r["cls_attention"]["enrichment"].items():
        for l in range(2):
            L.append(f"| {c} | C{l} | " + " | ".join(
                f"{e[k]['layer_mean'][l]:.2f} [{e[k]['layer_mean_lo'][l]:.2f}, {e[k]['layer_mean_hi'][l]:.2f}]"
                for k in ["displaced", "lepton", "prompt_ch", "neutral", "lead3"]) + " |")
    if s in r["cls_attention"]["regression"]:
        L.append(f"\nWithin-jet regression of log CLS attention ({s} jets, standardised features, 68% CI):\n")
        L.append("| block | " + " | ".join(r["cls_attention"]["regression"][s]["C0"]) + " |\n|---|" +
                 "---|" * len(r["cls_attention"]["regression"][s]["C0"]))
        for blk, co in r["cls_attention"]["regression"][s].items():
            L.append(f"| {blk} | " + " | ".join(f"{v['beta']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
                                                for v in co.values()) + " |")

    hp = r["head_pairwise"]
    L.append("\n### 2. Head specialisation\n")
    rho = {k: np.array(v) for k, v in hp["spearman"].items()}
    for k, m in rho.items():
        i, j = np.unravel_index(np.abs(m).argmax(), m.shape)
        L.append(f"- strongest |Spearman(attention, {k})|: L{i}.h{j} = {m[i, j]:+.2f} "
                 f"(layer-mean |rho|: {', '.join(f'{x:.2f}' for x in np.abs(m).mean(1))})")
    e = hp["disp_disp_enrichment"]
    tag = s if s in e["full"] else "all"
    full, perm = np.array(e["full"][tag]["est"]), np.array(e["permuted"][tag]["est"])
    lo = np.array(e["full"][tag]["lo"])
    i, j = np.unravel_index(full.argmax(), full.shape)
    L.append(f"- displaced->displaced attention enrichment in {tag} jets ({e['full'][tag]['n_jets']} jets with >=2 "
             f"displaced tracks): max L{i}.h{j} = {full[i, j]:.2f} (68% CI low {lo[i, j]:.2f}); permuted-label control "
             f"max {perm.max():.2f}; heads with CI low > 1.5: {int((lo > 1.5).sum())}/64.")
    for v in ("bias_only", "qk_only"):
        if tag in e.get(v, {}):
            L.append(f"  - same with only the {v.replace('_only', '')} term in the softmax: max "
                     f"{np.array(e[v][tag]['est']).max():.2f}")

    bq = r["bias_vs_qk"]
    fv = np.array(bq["frac_var_bias"])
    L.append("\n### 3. Interaction bias psi(U) vs content QK\n")
    L.append("| layer | Var share of psi(U) (head mean, min-max) | corr(QK, psi) | TV(attn, no psi) | TV(attn, no QK) |\n|---|---|---|---|---|")
    for l in range(len(fv)):
        L.append(f"| L{l} | {fv[l].mean():.2f} ({fv[l].min():.2f}-{fv[l].max():.2f}) | "
                 f"{np.mean(bq['corr_qk_bias'][l]):+.2f} | {np.mean(bq['tv_drop_bias'][l]):.2f} | "
                 f"{np.mean(bq['tv_drop_qk'][l]):.2f} |")

    ab = r["ablation"]
    L.append(f"\n### 4. Causal tests ({key})\n")
    hd = {k: v[key]["d_auc"] for k, v in ab["heads"].items() if key in v}
    worst = sorted(hd, key=hd.get)[:6]
    L.append("Most important single heads (zero-ablation dAUC, 68% CI): " + "; ".join(
        f"{k} {hd[k]:+.4f} [{ab['heads'][k][key]['d_auc_ci'][0]:+.4f}, {ab['heads'][k][key]['d_auc_ci'][1]:+.4f}]"
        for k in worst))
    if "top_vs_random" in ab:
        tv = ab["top_vs_random"]
        L.append(f"\nJoint ablation of the top-5 particle heads {tv['top_heads']}: dAUC {tv['d_auc_top']:+.4f}; "
                 f"random 5-head sets: median {np.median(tv['d_auc_random']):+.4f}, min {np.min(tv['d_auc_random']):+.4f}.")
    L.append("\n| intervention | dAUC [68% CI] | mean d logit(sig) | mean d logit(bkg) | R50 ratio | R99 ratio |\n|---|---|---|---|---|---|")
    for grp in ("bias", "knockout"):
        for t, v in ab[grp].items():
            v = v[key]
            L.append(f"| {t} | {v['d_auc']:+.4f} [{v['d_auc_ci'][0]:+.4f}, {v['d_auc_ci'][1]:+.4f}] | "
                     f"{v['d_logit_sig']:+.2f} | {v['d_logit_bkg']:+.2f} | {f(v['R50_ratio'])} | {f(v.get('R99_ratio'))} |")
    pt = r["patching"]
    L.append(f"\nActivation patching on {pt['n_jets']} {s} jets with >=1 displaced track. Corruption: impact "
             f"parameters of displaced tracks set to 0. Metric log p({s})/p({b}): clean {pt['m_clean']:.2f}, "
             f"corrupt {pt['m_corrupt']:.2f}. Recovered fraction when restoring the clean residual stream:\n")
    L.append("| position | displaced tracks only | all other particles |\n|---|---|---|")
    for i, (a_, b_) in enumerate(zip(pt["resid_disp"], pt["resid_rest"])):
        L.append(f"| input of {'L' + str(i) if i < 8 else 'CLS'} | {a_['est']:.2f} [{a_['lo']:.2f}, {a_['hi']:.2f}] | "
                 f"{b_['est']:.2f} [{b_['lo']:.2f}, {b_['hi']:.2f}] |")
    hh = sorted(pt["heads"].items(), key=lambda kv: -kv[1]["est"])[:6]
    L.append("\nSingle-head context patching, largest recovery: " + "; ".join(
        f"{k} {v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in hh))

    pr = r["probes"]
    L.append("\n### 5. Linear probes on the residual stream (test AUC, held-out jets)\n")
    L.append("| target | " + " | ".join(f"pos {i}" for i in range(9)) + " |\n|---|" + "---|" * 9)
    for t, v in pr["trained"].items():
        L.append(f"| {t} (trained) | " + " | ".join(f(x) for x in v["auc"]) + " |")
        if "random_init" in pr:
            L.append(f"| {t} (random init) | " + " | ".join(f(x) for x in pr["random_init"][t]["auc"]) + " |")
        if v.get("shuffled_auc"):
            L.append(f"| {t} (shuffled labels) | " + " | ".join(f(x) for x in v["shuffled_auc"]) + " |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = ["# Results report\n", "Generated by `scripts/make_report.py`. All intervals are 68% bootstrap intervals "
             "over jets. Figures are in each run directory.\n"]
    for d in a.dirs:
        r = json.load(open(os.path.join(d, "results.json")))
        parts.append(section(r, d))
    if len(a.dirs) > 1:
        plots.fig_compare(a.dirs, os.path.dirname(os.path.abspath(a.out)))
        parts.append("Comparison figure: `fig_compare.png`.\n")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
