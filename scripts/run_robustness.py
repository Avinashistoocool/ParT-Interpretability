"""Focused, paired robustness controls on JetClass jets.

python scripts/run_robustness.py --ckpt checkpoints/ParT_full.pt \
    --kin_ckpt checkpoints/ParT_kin.pt --data_dir data/JetClass/test \
    --out results/robustness_test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from partinterp import CLASS_LIST, analyses  # noqa: E402
from partinterp.data import FEAT, find_file, load_jetclass  # noqa: E402
from partinterp.models import CLS_LAYERS, PARTICLE_LAYERS, Ctrl, load_part, load_part_kin  # noqa: E402


def displaced(store, idx, P, cut):
    return analyses.particle_physics(store, idx, P, sig_cut=cut)["displaced"]


def corrupt_features(feats, flags, method):
    out = feats.clone()
    channels = [FEAT["part_d0"]] if method == "zero_d0" else [FEAT["part_d0"], FEAT["part_dz"]]
    for channel in channels:
        if method == "prompt_mean":
            # Mean among charged, non-displaced particles in the same jet.
            prompt = flags["charged"] & ~flags["displaced"]
            values = feats[:, channel]
            weight = torch.as_tensor(prompt, device=values.device)
            available = weight.any(1)
            mean = (values * weight).sum(1) / weight.sum(1).clamp(min=1)
            out[:, channel] = torch.where(torch.as_tensor(flags["displaced"], device=values.device)
                                          & available[:, None],
                                          mean[:, None], values)
        else:
            out[:, channel] = out[:, channel].masked_fill(
                torch.as_tensor(flags["displaced"], device=out.device), 0)
    return out


def knockout_mask(store, idx, P, cut, matched):
    ph = analyses.particle_physics(store, idx, P, sig_cut=cut)
    ko = ph["displaced"].copy()
    if matched:
        ctl = np.zeros_like(ko)
        for j in range(len(idx)):
            candidates = np.where(ph["charged"][j] & ~ko[j])[0]
            for i in np.where(ko[j])[0]:
                if len(candidates) == 0:
                    break
                k = np.argmin(np.abs(ph["ln_pt"][j, candidates] - ph["ln_pt"][j, i]))
                ctl[j, candidates[k]] = True
                candidates = np.delete(candidates, k)
        ko = ctl
    ko &= ko.sum(1, keepdims=True) < ph["mask"].sum(1, keepdims=True)
    return ko


def cls_enrichment(rows, label, n_boot, rng):
    sel = (rows["label"] == label) & rows["displaced"]
    jet = rows["jet"][sel]
    if len(jet) == 0:
        return None
    n = int(rows["jet"].max()) + 1
    all_jet = rows["jet"]
    count = np.bincount(all_jet, minlength=n)
    pop = np.bincount(jet, minlength=n) / np.maximum(count, 1)
    share = np.stack([
        np.bincount(jet, weights=rows["a"][sel, layer].mean(1) / count[jet], minlength=n)
        for layer in range(2)
    ], -1)
    keep = pop > 0
    est, lo, hi = analyses.boot_ratio(share[keep], np.broadcast_to(pop[keep, None], share[keep].shape),
                                     n_boot, rng)
    return {"C0": [float(est[0]), float(lo[0]), float(hi[0])],
            "C1": [float(est[1]), float(lo[1]), float(hi[1])], "n_jets": int(keep.sum())}


def qk_enrichment(rec, label, n_boot, rng):
    selected = (rec["labels"] == label) & (rec["enrich"]["full"][1][:, 0, 0] > 0)
    out = {}
    for name in ("full", "qk_only", "bias_only", "permuted"):
        num, den = rec["enrich"][name]
        if not selected.any():
            out[name] = None
            continue
        est, lo, hi = analyses.boot_ratio(num[selected], den[selected], n_boot, rng)
        out[name] = {"L6.h4": [float(x[6, 4]) for x in (est, lo, hi)],
                     "maximum": float(est.max()), "n_jets": int(selected.sum())}
    return out


def run(ad, store, idx, device, batch_size, n_boot, seed, record, has_impact_parameters=True):
    rng = np.random.default_rng(seed)
    labels = store.labels[idx]
    pair = [("Hbb", "QCD")]
    base = analyses.predict(ad, store, idx, device, batch_size)
    result = {"baseline": analyses.performance(base, labels, pair)["pairs"]["Hbb_vs_QCD"],
              "thresholds": {}, "corruptions": {}}
    for cut in (2.0, 3.0, 5.0):
        counts = [int(displaced(store, b, P, cut).sum()) for b, P in analyses._batches(store, idx, batch_size)]
        entry = {"n_displaced": int(sum(counts)), "knockout": {}}
        for matched in (False, True):
            probs = analyses.predict(
                ad, store, idx, device, batch_size,
                make_ctrl=lambda b, P, mk, c=cut, match=matched: Ctrl(
                    key_knockout=torch.as_tensor(knockout_mask(store, b, P, c, match), device=mk.device),
                    knockout_layers=set(PARTICLE_LAYERS + CLS_LAYERS)))
            entry["knockout"]["ptmatched_prompt" if matched else "displaced"] = \
                analyses._pair_metrics(base, probs, labels, pair, n_boot, rng)["Hbb_vs_QCD"]
        if record:
            rec = analyses.recording_pass(ad, store, idx, device, bs=max(4, batch_size // 2),
                                          max_pairs=10000, rng=np.random.default_rng(seed), sig_cut=cut)
            entry["cls_enrichment_Hbb"] = cls_enrichment(rec["cls"], CLASS_LIST.index("Hbb"), n_boot, rng)
            entry["pair_enrichment_Hbb"] = qk_enrichment(rec, CLASS_LIST.index("Hbb"), n_boot, rng)
        result["thresholds"][str(int(cut))] = entry
    if has_impact_parameters:
        ph = analyses.particle_physics(store, idx, store.mask.shape[1])
        result["prompt_mean_unchanged_jets"] = int(
            (ph["displaced"].any(1) & ~(ph["charged"] & ~ph["displaced"]).any(1)).sum())
        for method in ("zero_d0", "zero_d0_dz", "prompt_mean"):
            probs = analyses.predict(
                ad, store, idx, device, batch_size,
                feats_fn=lambda f, b, P, m=method: corrupt_features(
                    f, analyses.particle_physics(store, b, P), m))
            result["corruptions"][method] = analyses._pair_metrics(
                base, probs, labels, pair, n_boot, rng)["Hbb_vs_QCD"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--kin_ckpt", help="official ParT_kin checkpoint for negative control")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_per_class", type=int, default=300)
    parser.add_argument("--file_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.n_per_class < 2 or args.batch_size < 1 or args.n_boot < 1:
        parser.error("n_per_class >= 2, batch_size >= 1 and n_boot >= 1 are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    store = load_jetclass(args.data_dir, ["QCD", "Hbb"], args.n_per_class, file_index=args.file_index)
    idx = np.arange(len(store))
    files = {cls: os.path.basename(find_file(args.data_dir, cls, args.file_index))
             for cls in ("QCD", "Hbb")}
    out = {"args": vars(args), "source": {"files": files, "data_dir": args.data_dir,
           "selection": f"first {args.n_per_class} jets per class; may overlap other runs using these files"},
           "part": run(load_part(args.ckpt, args.device), store, idx, args.device,
                       args.batch_size, args.n_boot, args.seed, record=True)}
    if args.kin_ckpt:
        out["part_kin"] = run(load_part_kin(args.kin_ckpt, args.device), store, idx, args.device,
                              args.batch_size, args.n_boot, args.seed, record=True,
                              has_impact_parameters=False)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as f:
        json.dump(analyses._jsonable(out), f, indent=1)
    print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
