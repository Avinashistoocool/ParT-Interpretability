"""Figures from results.json (one directory per model)."""
import json
import os

import numpy as np

PAIR_FEATS = ["ln_dR", "ln_kT", "ln_z", "ln_m2"]
CAT_LABEL = {"displaced": "displaced trk", "lepton": "e/mu", "prompt_ch": "prompt ch.", "neutral": "neutral",
             "lead3": "3 leading"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    return plt


def fig_cls_enrichment(res, out):
    plt = _plt()
    enr = res["cls_attention"]["enrichment"]
    classes = list(enr)
    cats = list(CAT_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    w = 0.8 / len(classes)
    for l, ax in enumerate(axes):
        for k, c in enumerate(classes):
            est = [enr[c][cat]["layer_mean"][l] for cat in cats]
            lo = [enr[c][cat]["layer_mean_lo"][l] for cat in cats]
            hi = [enr[c][cat]["layer_mean_hi"][l] for cat in cats]
            x = np.arange(len(cats)) + (k - (len(classes) - 1) / 2) * w
            ax.bar(x, est, w, label=c, yerr=[np.subtract(est, lo), np.subtract(hi, est)], capsize=2)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(cats)), [CAT_LABEL[c] for c in cats], rotation=20)
        ax.set_title(f"class-attention block C{l} (head mean)")
        ax.set_yscale("log")
    axes[0].set_ylabel("CLS attention share / population share")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{res['model']}: which particles the classifier reads")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_cls_enrichment.png"))
    plt.close(fig)

    sig = res["signal"]
    if sig in enr:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        m = np.array(enr[sig]["displaced"]["per_head"])
        im = ax.imshow(np.log2(np.maximum(m, 1e-3)), cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_yticks([0, 1], ["C0", "C1"])
        ax.set_xlabel("head")
        for (i, j), val in np.ndenumerate(m):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, label="log2 enrichment")
        ax.set_title(f"{res['model']}: CLS attention enrichment on displaced tracks, {sig} jets")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig1b_cls_displaced_per_head.png"))
        plt.close(fig)


def fig_head_pairwise(res, out):
    plt = _plt()
    hp = res["head_pairwise"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for k, f in enumerate(PAIR_FEATS):
        im = axes[k].imshow(np.array(hp["spearman"][f]), cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        axes[k].set_title(f"Spearman(attn, {f})")
        axes[k].set_xlabel("head")
        axes[k].set_ylabel("layer")
    enr = hp["disp_disp_enrichment"]["full"]
    key = res["signal"] if res["signal"] in enr else "all"
    m = np.array(enr[key]["est"])
    im2 = axes[4].imshow(np.log2(np.maximum(m, 1e-3)), cmap="PuOr_r", vmin=-3, vmax=3)
    axes[4].set_title(f"displaced->displaced, {key}\n(log2 enrichment)")
    axes[4].set_xlabel("head")
    fig.colorbar(im, ax=axes[:4], fraction=0.02)
    fig.colorbar(im2, ax=axes[4], fraction=0.05)
    fig.suptitle(f"{res['model']}: head specialisation")
    fig.savefig(os.path.join(out, "fig2_head_pairwise.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    for v, st in [("full", "-"), ("bias_only", "--"), ("qk_only", ":"), ("permuted", "-.")]:
        if key in hp["disp_disp_enrichment"].get(v, {}):
            e = np.array(hp["disp_disp_enrichment"][v][key]["est"])
            ax.plot(range(e.shape[0]), e.max(1), st, marker="o", label=f"{v} (max over heads)")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("displaced->displaced enrichment")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(f"{res['model']}: secondary-vertex-like attention ({key})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2b_disp_disp_by_layer.png"))
    plt.close(fig)


def fig_bias_vs_qk(res, out):
    plt = _plt()
    b = res["bias_vs_qk"]
    fv = np.array(b["frac_var_bias"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    im = axes[0].imshow(fv, cmap="viridis", vmin=0, vmax=1)
    for (i, j), val in np.ndenumerate(fv):
        axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="w" if val < 0.5 else "k")
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("layer")
    axes[0].set_title("Var[psi(U)] / (Var[psi(U)] + Var[QK])")
    fig.colorbar(im, ax=axes[0])
    L = np.arange(fv.shape[0])
    axes[1].plot(L, np.mean(b["tv_drop_bias"], 1), "o-", label="TV(attn, attn without psi(U))")
    axes[1].plot(L, np.mean(b["tv_drop_qk"], 1), "s-", label="TV(attn, attn without QK)")
    axes[1].plot(L, fv.mean(1), "^-", label="bias variance fraction")
    axes[1].set_xlabel("layer")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{res['model']}: interaction bias vs content")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_bias_vs_qk.png"))
    plt.close(fig)


def fig_causal(res, out):
    plt = _plt()
    key = f"{res['signal']}_vs_{res['background']}"
    ab = res["ablation"]
    names = list(ab["heads"])
    grid = np.full((10, 8), np.nan)
    rows = [f"L{i}" for i in range(8)] + ["C0", "C1"]
    for n in names:
        l, h = n.split(".")
        if key in ab["heads"][n]:
            grid[rows.index(l), int(h)] = ab["heads"][n][key]["d_auc"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    lim = max(np.nanmax(np.abs(grid)), 1e-4)
    im = axes[0].imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[0].set_yticks(range(10), rows)
    axes[0].set_xlabel("head")
    axes[0].set_title(f"single-head zero ablation: dAUC ({key})")
    fig.colorbar(im, ax=axes[0])
    tags = list(ab["bias"]) + list(ab["knockout"])
    vals = [(ab["bias"].get(t) or ab["knockout"].get(t))[key] for t in tags]
    y = np.arange(len(tags))
    d = np.array([v["d_auc"] for v in vals])
    err = np.array([[v["d_auc"] - v["d_auc_ci"][0], v["d_auc_ci"][1] - v["d_auc"]] for v in vals]).T
    axes[1].barh(y, d, xerr=np.abs(err), color=["C0"] * len(ab["bias"]) + ["C3"] * len(ab["knockout"]))
    axes[1].set_yticks(y, tags, fontsize=7)
    axes[1].set_xscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("dAUC")
    axes[1].set_title("bias removal (blue) / attention knockout (red)")
    p = res["patching"]
    xs = np.arange(len(p["resid_disp"]))
    for k, lab in [("resid_disp", "patch displaced-track residuals"), ("resid_rest", "patch all other residuals")]:
        e = np.array([[r["est"], r["lo"], r["hi"]] for r in p[k]])
        axes[2].errorbar(xs, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], marker="o", label=lab, capsize=2)
    axes[2].set_xlabel("residual stream position (input of block i; 8 = before CLS)")
    axes[2].set_ylabel("recovered fraction of logit gap")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].axhline(1, color="k", lw=0.6, ls="--")
    axes[2].legend(fontsize=7)
    axes[2].set_title(f"activation patching, {p['n_jets']} {res['signal']} jets")
    fig.suptitle(f"{res['model']}: causal tests")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_causal.png"))
    plt.close(fig)


def fig_probes(res, out):
    plt = _plt()
    pr = res["probes"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, (task, r) in enumerate(pr["trained"].items()):
        x = np.arange(len(r["auc"]))
        a = np.array(r["auc"], float)
        lo = np.nan_to_num(np.clip(a - np.array(r["lo"], float), 0, None))
        hi = np.nan_to_num(np.clip(np.array(r["hi"], float) - a, 0, None))
        ax.errorbar(x, a, yerr=[lo, hi], marker="o",
                    color=f"C{k}", label=task, capsize=2)
        if "random_init" in pr:
            ax.plot(x, pr["random_init"][task]["auc"], ls=":", color=f"C{k}")
        if r.get("shuffled_auc"):
            ax.plot(x, r["shuffled_auc"], ls="--", lw=0.7, color=f"C{k}", alpha=0.6)
    ax.set_xlabel("residual stream position (0 = embedding)")
    ax.set_ylabel("probe test AUC")
    ax.set_title(f"{res['model']}: linear probes (dotted: random-init model; dashed: shuffled labels)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_probes.png"))
    plt.close(fig)


def make_all(res, out):
    for f in (fig_cls_enrichment, fig_head_pairwise, fig_bias_vs_qk, fig_causal, fig_probes):
        try:
            f(res, out)
        except Exception as e:  # a missing class must not kill the remaining figures
            print(f"[plots] {f.__name__} failed: {e!r}")


def fig_compare(dirs, out):
    """Side-by-side ParT vs GDS-ParT summary."""
    plt = _plt()
    rs = [json.load(open(os.path.join(d, "results.json"))) for d in dirs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for r in rs:
        fv = np.array(r["bias_vs_qk"]["frac_var_bias"])
        axes[0].plot(fv.mean(1), "o-", label=r["model"])
        e = np.array(r["head_pairwise"]["disp_disp_enrichment"]["full"].get(r["signal"], {"est": [[np.nan]]})["est"])
        axes[1].plot(e.max(1), "o-", label=r["model"])
        a = r["probes"]["trained"]["displaced (charged only)"]["auc"]
        axes[2].plot(a, "o-", label=r["model"])
    axes[0].set_title("bias variance fraction (head mean)")
    axes[1].set_title("max displaced->displaced enrichment")
    axes[1].set_yscale("log")
    axes[2].set_title("probe AUC: displaced track")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_compare.png"))
    plt.close(fig)
