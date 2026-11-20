"""The five analyses of the study, plus statistics helpers.

All analyses take an adapter from partinterp.models (ParT or GDS-ParT) and a
JetStore. Uncertainties are bootstrap percentile intervals over *jets*
(particles and pairs within a jet are not independent).

  1. cls_attention   which particles the class-attention (CLS) heads read
  2. head_pairwise   per-head relation to pairwise kinematics and to
                     displaced-track pairs ("same secondary vertex" proxy)
  3. bias_vs_qk      share of attention-logit variance from psi(U) vs QK^T
  4. ablation        head / bias ablation, attention knockout of displaced tracks
     patching        activation patching with "un-displaced" counterfactual jets
  5. probes          linear probes on the per-particle residual stream
"""
import json
import math
import os
import time

import numpy as np
import torch

from . import CLASS_LIST
from .models import CLS_LAYERS, PARTICLE_LAYERS, Ctrl

SIG_CUT = 3.0        # |d0|/sigma(d0) above which a charged track is "displaced"
NEAR_DR = 0.4        # neighbourhood radius for "displaced tracks near particle"
PAIR_FEATS = ["ln_dR", "ln_kT", "ln_z", "ln_m2"]
CATS = ["displaced", "lepton", "prompt_ch", "neutral", "lead3"]


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def auc(score, y):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rejection(score, y, eff):
    """1/eps_B at signal efficiency `eff`; returns (R, n_bkg_passing)."""
    y = np.asarray(y, bool)
    thr = np.quantile(score[y], 1 - eff)
    nb = int((score[~y] >= thr).sum())
    return float((~y).sum() / max(nb, 1)), nb


def ci(samples, alpha=0.32):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))]


def boot_ratio(num, den, n_boot, rng):
    """Ratio of sums over jets with a jet-level bootstrap. num, den: (J, ...)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum(0) / np.maximum(den.sum(0), 1e-12)
    J = len(num)
    bs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, J, J), minlength=J).astype(float)
        sh = (slice(None),) + (None,) * (num.ndim - 1)
        bs.append((num * w[sh]).sum(0) / np.maximum((den * w[sh]).sum(0), 1e-12))
    bs = np.array(bs)
    return est, np.quantile(bs, 0.16, axis=0), np.quantile(bs, 0.84, axis=0)


def boot_delta_auc(d0, d1, y, n_boot, rng):
    y = np.asarray(y, bool)
    base, new = auc(d0, y), auc(d1, y)
    i1, i0 = np.where(y)[0], np.where(~y)[0]
    out = []
    for _ in range(n_boot):
        s = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        out.append(auc(d1[s], y[s]) - auc(d0[s], y[s]))
    return new - base, ci(out)


def pair_disc(probs, labels, sig, bkg):
    s, b = CLASS_LIST.index(sig), CLASS_LIST.index(bkg)
    sel = (labels == s) | (labels == b)
    p = probs[sel]
    d = p[:, s] / np.maximum(p[:, s] + p[:, b], 1e-30)
    return d, labels[sel] == s


def safe_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


# ----------------------------------------------------------------------------
# per-particle physics
# ----------------------------------------------------------------------------
def particle_physics(store, idx, P):
    r = {k: v[idx, :P] for k, v in store.raw.items() if v.ndim == 2}
    mask = store.mask[idx, :P]
    charged = (r["charge"] != 0) & mask
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(charged & (r["d0err"] > 0), np.abs(r["d0"]) / r["d0err"], 0.0)
    pid = r["pid"]
    displaced = charged & (sig > SIG_CUT)
    lepton = mask & ((pid == 3) | (pid == 4))
    neutral = mask & ((pid == 1) | (pid == 2))
    pt = np.where(mask, r["pt"], 0.0)
    order = np.argsort(-np.where(mask, pt, -1.0), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(P)[None].repeat(len(idx), 0), axis=1)
    deta, dphi = r["deta"], r["dphi"]
    dphi_ij = np.remainder(dphi[:, :, None] - dphi[:, None, :] + np.pi, 2 * np.pi) - np.pi
    dr_ij = np.hypot(deta[:, :, None] - deta[:, None, :], dphi_ij)
    near = (dr_ij < NEAR_DR) & ~np.eye(P, dtype=bool)[None]
    n_disp_near = (near & displaced[:, None, :]).sum(-1) * mask
    return {
        "mask": mask, "charged": charged, "sig": sig, "log_sig": np.log1p(sig), "displaced": displaced,
        "lepton": lepton, "neutral": neutral, "prompt_ch": charged & ~displaced & ~lepton,
        "lead3": mask & (rank < 3), "rank": rank, "ln_pt": np.log(np.maximum(pt, 1e-6)),
        "dR": np.hypot(deta, dphi), "n_disp_near": n_disp_near,
    }


# ----------------------------------------------------------------------------
# forward helpers
# ----------------------------------------------------------------------------
def _batches(store, idx, bs):
    idx = np.asarray(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        yield b, max(int(store.mask[b].sum(1).max()), 2)


@torch.no_grad()
def predict(ad, store, idx, device, bs=256, make_ctrl=None, feats_fn=None):
    out = []
    for b, P in _batches(store, idx, bs):
        f, v, mk = ad.inputs(store, b, P, device)
        if feats_fn is not None:
            f = feats_fn(f, b, P)
        ctrl = make_ctrl(b, P, mk) if make_ctrl else None
        out.append(torch.softmax(ad.forward(f, v, mk, ctrl).float(), -1).cpu().numpy())
    return np.concatenate(out)


def _pick(labels, classes, n_per_class, rng):
    out = []
    for c in classes:
        w = np.where(labels == CLASS_LIST.index(c))[0]
        out.append(np.sort(rng.choice(w, min(n_per_class, len(w)), replace=False)) if len(w) else w)
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# performance
# ----------------------------------------------------------------------------
def performance(probs, labels, pairs):
    res = {"n_jets": int(len(labels)), "accuracy_10way": float((probs.argmax(1) == labels).mean()), "pairs": {}}
    for s, b in pairs:
        d, y = pair_disc(probs, labels, s, b)
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        r50, n50 = rejection(d, y, 0.5)
        r90, n90 = rejection(d, y, 0.9)
        r99, n99 = rejection(d, y, 0.99)
        res["pairs"][f"{s}_vs_{b}"] = {"auc": auc(d, y), "R50": r50, "n_bkg_pass_50": n50,
                                       "R90": r90, "n_bkg_pass_90": n90, "R99": r99, "n_bkg_pass_99": n99, "n_sig": int(y.sum()), "n_bkg": int((~y).sum())}
    return res


# ----------------------------------------------------------------------------
# analyses 1-3: one recording pass
# ----------------------------------------------------------------------------
@torch.no_grad()
def recording_pass(ad, store, idx, device, bs=48, max_pairs=150000, rng=None):
    rng = rng or np.random.default_rng(0)
    L, H = len(PARTICLE_LAYERS), 8
    acc = {k: np.zeros((L, H)) for k in ["vqk", "vb", "cov", "tv_noBias", "tv_noQK", "nrows",
                                          "mean_dR", "entropy"]}
    enrich = {v: {"num": [], "den": []} for v in ["full", "bias_only", "qk_only", "permuted"]}
    pair_att, pair_x, pair_flags = [], [], []
    cls_rows = []
    jet_labels = []
    jet_offset = 0
    pairs_per_batch = max(50, int(max_pairs * bs / max(len(idx), 1)))
    for b, P in _batches(store, idx, bs):
        f, v, mk = ad.inputs(store, b, P, device)
        ctrl = Ctrl(record={"qk", "bias", "attn"})
        ad.forward(f, v, mk, ctrl)
        ph = particle_physics(store, b, P)
        B = len(b)
        mkf = mk.float()
        n = mkf.sum(1)
        eye = torch.eye(P, dtype=torch.bool, device=device)
        vp = mk[:, :, None] & mk[:, None, :]
        off = vp & ~eye
        from .models import pairwise_features
        U = pairwise_features(v.transpose(1, 2), mk)                     # (B,P,P,4)
        dR = U[..., 0].exp() * off
        disp = torch.from_numpy(ph["displaced"]).to(device)
        # permutation null: shuffle the displaced flag among charged tracks of each jet
        perm = ph["displaced"].copy()
        for j in range(B):
            ch = np.where(ph["charged"][j])[0]
            perm[j, ch] = rng.permutation(perm[j, ch])
        disp_perm = torch.from_numpy(perm).to(device)
        # sampled pairs for rank correlations
        nz = off.nonzero()
        sel = nz[torch.from_numpy(rng.choice(len(nz), min(pairs_per_batch, len(nz)), replace=False)).to(device)]
        bi, ii, jj = sel.unbind(1)
        pair_x.append(U[bi, ii, jj].cpu().numpy())
        pair_flags.append(np.stack([(disp[bi, ii] & disp[bi, jj]).cpu().numpy(),
                                    np.asarray(jet_offset + bi.cpu().numpy())], 1))
        att_cols = []
        rowmask = mk[:, None, :, None]                                     # real queries
        keym = mk[:, None, None, :].float()
        for l, name in enumerate(PARTICLE_LAYERS):
            a, qk, bias = ctrl.out[f"attn/{name}"], ctrl.out[f"qk/{name}"], ctrl.out[f"bias/{name}"]
            att_cols.append((a[bi, :, ii, jj] * n[bi, None]).T.cpu().numpy())
            # 3. logit decomposition, centred within each attention row over real keys
            nk = keym.sum(-1, keepdim=True)
            mq = (qk * keym).sum(-1, keepdim=True) / nk
            mb = (bias * keym).sum(-1, keepdim=True) / nk
            cq, cb = (qk - mq) * keym, (bias - mb) * keym
            rows = mkf[:, None, :]                                         # (B,1,P)
            acc["vqk"][l] += ((cq ** 2).sum(-1) / nk[..., 0] * rows).sum((0, 2)).cpu().numpy()
            acc["vb"][l] += ((cb ** 2).sum(-1) / nk[..., 0] * rows).sum((0, 2)).cpu().numpy()
            acc["cov"][l] += ((cq * cb).sum(-1) / nk[..., 0] * rows).sum((0, 2)).cpu().numpy()
            acc["nrows"][l] += float(mkf.sum())
            neg = ~mk[:, None, None, :]
            a_qk = torch.softmax(qk.masked_fill(neg, float("-inf")), -1)
            a_b = torch.softmax(bias.masked_fill(neg, float("-inf")), -1)
            acc["tv_noBias"][l] += (0.5 * (a - a_qk).abs().sum(-1) * rows).sum((0, 2)).cpu().numpy()
            acc["tv_noQK"][l] += (0.5 * (a - a_b).abs().sum(-1) * rows).sum((0, 2)).cpu().numpy()
            acc["mean_dR"][l] += ((a * dR[:, None]).sum(-1) * rows).sum((0, 2)).cpu().numpy()
            ent = -(a.clamp(min=1e-12).log() * a).sum(-1) / n.clamp(min=2).log()[:, None, None]
            acc["entropy"][l] += (ent * rows).sum((0, 2)).cpu().numpy()
            # 2b. displaced -> displaced attention enrichment, per jet
            for vname, att, dflag in [("full", a, disp), ("bias_only", a_b, disp), ("qk_only", a_qk, disp),
                                      ("permuted", a, disp_perm)]:
                attx = att * off[:, None].float()
                tot = attx.sum(-1)                                          # (B,H,P)
                sh = (attx * dflag[:, None, None, :].float()).sum(-1) / tot.clamp(min=1e-12)
                nd = dflag.float().sum(1)
                pop = ((nd[:, None] - 1) / (n[:, None] - 1).clamp(min=1)).expand(B, P)
                qmask = (dflag & (nd[:, None] >= 2)).float()
                if l == 0:
                    enrich[vname]["num"].append(np.zeros((B, L, H)))
                    enrich[vname]["den"].append(np.zeros((B, L, H)))
                enrich[vname]["num"][-1][:, l] = (sh * qmask[:, None]).sum(-1).cpu().numpy()
                enrich[vname]["den"][-1][:, l] = (pop * qmask).sum(-1)[:, None].expand(B, H).cpu().numpy()
        pair_att.append(np.concatenate(att_cols, 0).astype(np.float32))                       # (L*H, k)
        # 1. CLS attention per real particle, renormalised over particles (CLS self-key excluded)
        a_np = []
        for name in CLS_LAYERS:
            ac = ctrl.out[f"attn/{name}"][:, :, 0, 1:]                    # (B,H,P)
            a_np.append((ac / ac.sum(-1, keepdim=True).clamp(min=1e-12)).cpu().numpy())
        a_np = np.stack(a_np, 0)                                           # (2,B,H,P)
        jb, pb = np.nonzero(ph["mask"])
        # advanced indices separated by a slice move to the front: (n, 2, H); scaled by n so 1 = uniform
        rows = {"jet": jet_offset + jb, "label": store.labels[b][jb],
                "a": a_np[:, jb, :, pb] * n.cpu().numpy()[jb][:, None, None]}
        for k in ["displaced", "lepton", "prompt_ch", "neutral", "lead3", "ln_pt", "dR", "log_sig", "charged", "rank"]:
            rows[k] = ph[k][jb, pb]
        cls_rows.append(rows)
        jet_labels.append(store.labels[b])
        jet_offset += B
    labels = np.concatenate(jet_labels)
    rows = {k: np.concatenate([r[k] for r in cls_rows]) for k in cls_rows[0]}
    pairs = {"att": np.concatenate(pair_att, 1), "x": np.concatenate(pair_x), "flags": np.concatenate(pair_flags)}
    enr = {k: (np.concatenate(v["num"]), np.concatenate(v["den"])) for k, v in enrich.items()}
    return {"acc": acc, "enrich": enr, "pairs": pairs, "cls": rows, "labels": labels}


def analyse_cls(rows, labels, classes, n_boot, rng):
    """Attention share / population share per particle category, plus a
    within-jet regression of log attention on particle properties."""
    out = {"enrichment": {}, "regression": {}, "spearman": {}}
    J = int(rows["jet"].max()) + 1
    for c in classes:
        ci_ = CLASS_LIST.index(c)
        sel = rows["label"] == ci_
        if not sel.any():
            continue
        jet = rows["jet"][sel]
        a = rows["a"][sel]                         # (n, 2, H), = n_jet * share
        npart = np.bincount(jet, minlength=J)
        out["enrichment"][c] = {}
        for cat in CATS:
            m = rows[cat][sel].astype(float)
            share = np.stack([np.bincount(jet, weights=a[:, l, h] * m / np.maximum(npart[jet], 1), minlength=J)
                              for l in range(2) for h in range(a.shape[2])], 1)
            pop = np.bincount(jet, weights=m, minlength=J) / np.maximum(npart, 1)
            keep = (np.bincount(jet, weights=m, minlength=J) > 0)
            sh, pp = share[keep], pop[keep][:, None].repeat(share.shape[1], 1)
            est, lo, hi = boot_ratio(sh, pp, n_boot, rng)
            lm, llo, lhi = boot_ratio(sh.reshape(len(sh), 2, -1).mean(2), pp[:, :2], n_boot, rng)
            out["enrichment"][c][cat] = {
                "per_head": est.reshape(2, -1).tolist(), "per_head_lo": lo.reshape(2, -1).tolist(),
                "per_head_hi": hi.reshape(2, -1).tolist(),
                "layer_mean": lm.tolist(), "layer_mean_lo": llo.tolist(), "layer_mean_hi": lhi.tolist(),
                "frac_jets_with": float(keep[np.unique(jet)].mean()),
            }
        # regression: log(n * a) on standardised features, head-averaged attention per CLS layer
        X = np.stack([rows["ln_pt"][sel], rows["dR"][sel], rows["displaced"][sel], rows["lepton"][sel],
                      rows["charged"][sel], rows["log_sig"][sel]], 1).astype(float)
        names = ["ln_pT", "dR_axis", "displaced", "lepton", "charged", "log1p_sig"]
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        Xs = np.concatenate([np.ones((len(Xs), 1)), Xs], 1)
        out["regression"][c] = {}
        out["spearman"][c] = {}
        for l, name in enumerate(CLS_LAYERS):
            yv = np.log(np.maximum(a[:, l].mean(1), 1e-6))
            beta = np.linalg.lstsq(Xs, yv, rcond=None)[0]
            bs = []
            ujets = np.unique(jet)
            remap = np.searchsorted(ujets, jet)
            for _ in range(n_boot):
                w = np.bincount(rng.integers(0, len(ujets), len(ujets)), minlength=len(ujets))[remap].astype(float)
                sw = np.sqrt(w)
                bs.append(np.linalg.lstsq(Xs * sw[:, None], yv * sw, rcond=None)[0])
            bs = np.array(bs)
            out["regression"][c][name] = {nm: {"beta": float(beta[k + 1]), "ci": ci(bs[:, k + 1])}
                                          for k, nm in enumerate(names)}
            from scipy.stats import spearmanr
            out["spearman"][c][name] = {
                v: float(spearmanr(yv, rows[k][sel]).statistic)
                for v, k in [("log1p_sig", "log_sig"), ("ln_pT", "ln_pt"), ("pT_rank", "rank"), ("dR_axis", "dR")]}
    return out


def analyse_pairs(rec, classes, n_boot, rng, sig="Hbb", bkg="QCD"):
    acc = rec["acc"]
    nrows = np.maximum(acc["nrows"], 1)
    vqk, vb, cov = acc["vqk"] / nrows, acc["vb"] / nrows, acc["cov"] / nrows
    bias = {
        "frac_var_bias": (vb / (vb + vqk)).tolist(),
        "var_qk": vqk.tolist(), "var_bias": vb.tolist(),
        "corr_qk_bias": (cov / np.sqrt(np.maximum(vqk * vb, 1e-30))).tolist(),
        "tv_drop_bias": (acc["tv_noBias"] / nrows).tolist(),
        "tv_drop_qk": (acc["tv_noQK"] / nrows).tolist(),
    }
    pairs = rec["pairs"]
    ra = np.apply_along_axis(_rank, 1, pairs["att"])
    rx = np.apply_along_axis(_rank, 0, pairs["x"])
    za = (ra - ra.mean(1, keepdims=True)) / ra.std(1, keepdims=True)
    zx = (rx - rx.mean(0)) / rx.std(0)
    rho = (za @ zx) / za.shape[1]                                          # (L*H, 4)
    # jet-level bootstrap for rho would need re-ranking; use a split-half spread as uncertainty
    jet = pairs["flags"][:, 1]
    half = (jet % 2) == 0
    rho_h = []
    for hm in (half, ~half):
        ra_h = np.apply_along_axis(_rank, 1, pairs["att"][:, hm])
        rx_h = np.apply_along_axis(_rank, 0, pairs["x"][hm])
        a_ = (ra_h - ra_h.mean(1, keepdims=True)) / ra_h.std(1, keepdims=True)
        x_ = (rx_h - rx_h.mean(0)) / rx_h.std(0)
        rho_h.append((a_ @ x_) / a_.shape[1])
    head = {
        "spearman": {f: rho[:, k].reshape(8, 8).tolist() for k, f in enumerate(PAIR_FEATS)},
        "spearman_halfdiff": {f: (np.abs(rho_h[0] - rho_h[1])[:, k] / 2).reshape(8, 8).tolist()
                              for k, f in enumerate(PAIR_FEATS)},
        "mean_dR": (acc["mean_dR"] / nrows).tolist(),
        "entropy": (acc["entropy"] / nrows).tolist(),
        "n_pairs": int(pairs["att"].shape[1]),
    }
    head["dominant_feature"] = np.array(PAIR_FEATS)[np.abs(rho).argmax(1)].reshape(8, 8).tolist()
    enr = {}
    for vname, (num, den) in rec["enrich"].items():
        enr[vname] = {}
        for c in ["all"] + list(classes):
            sel = np.ones(len(num), bool) if c == "all" else rec["labels"] == CLASS_LIST.index(c)
            sel &= den[:, 0, 0] > 0
            if sel.sum() < 5:
                continue
            est, lo, hi = boot_ratio(num[sel], den[sel], n_boot, rng)
            enr[vname][c] = {"est": est.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n_jets": int(sel.sum())}
    head["disp_disp_enrichment"] = enr
    return head, bias


# ----------------------------------------------------------------------------
# 4. causal tests
# ----------------------------------------------------------------------------
def _pair_metrics(p0, p1, labels, pairs, n_boot, rng):
    res = {}
    for s, b in pairs:
        d0, y = pair_disc(p0, labels, s, b)
        d1, _ = pair_disc(p1, labels, s, b)
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        dauc, dci = boot_delta_auc(d0, d1, y, n_boot, rng)
        l0, l1 = safe_logit(d0), safe_logit(d1)
        r0, n0 = rejection(d0, y, 0.5)
        r1, n1 = rejection(d1, y, 0.5)
        q0, _ = rejection(d0, y, 0.99)
        q1, _ = rejection(d1, y, 0.99)
        # R50 saturates (no background passing) on small samples; R99 stays measurable
        res[f"{s}_vs_{b}"] = {"d_auc": dauc, "d_auc_ci": dci,
                              "d_logit_sig": float((l1 - l0)[y].mean()), "d_logit_bkg": float((l1 - l0)[~y].mean()),
                              "R50_ratio": r1 / r0, "R50_nbkg": [n0, n1], "R99_ratio": q1 / q0}
    return res


def ablation(ad, store, idx, device, pairs, n_boot, rng, bs=256, quick=False, n_random=20, top_k=5):
    labels = store.labels[idx]
    t0 = time.time()
    base = predict(ad, store, idx, device, bs)
    res = {"heads": {}, "bias": {}, "knockout": {}}
    layers = PARTICLE_LAYERS + CLS_LAYERS
    heads = [(l, h) for l in layers for h in range(8)]
    if quick:
        heads = heads[::7]
    for (l, h) in heads:
        p = predict(ad, store, idx, device, bs, make_ctrl=lambda b, P, mk: Ctrl(zero_heads={(l, h)}))
        res["heads"][f"{l}.{h}"] = _pair_metrics(base, p, labels, pairs, n_boot, rng)
    # remove psi(U) (layer by layer and everywhere); remove QK everywhere
    for tag, kw in ([(f"drop_bias_{l}", {"drop_bias": {l}}) for l in PARTICLE_LAYERS] +
                    [("drop_bias_all", {"drop_bias": set(PARTICLE_LAYERS)}),
                     ("drop_qk_all", {"drop_qk": set(PARTICLE_LAYERS)})]):
        p = predict(ad, store, idx, device, bs, make_ctrl=lambda b, P, mk, kw=kw: Ctrl(**kw))
        res["bias"][tag] = _pair_metrics(base, p, labels, pairs, n_boot, rng)
    # joint ablation of the top-k particle heads vs random k-head sets
    key = f"{pairs[0][0]}_vs_{pairs[0][1]}"
    ph = [k for k in res["heads"] if k.startswith("L") and key in res["heads"][k]]
    if len(ph) >= top_k:
        top = sorted(ph, key=lambda k: res["heads"][k][key]["d_auc"])[:top_k]
        zs = lambda ks: {(k.split(".")[0], int(k.split(".")[1])) for k in ks}
        p = predict(ad, store, idx, device, bs, make_ctrl=lambda b, P, mk: Ctrl(zero_heads=zs(top)))
        top_m = _pair_metrics(base, p, labels, pairs[:1], 0, rng)[key]["d_auc"]
        rand = []
        all_ph = [f"{l}.{h}" for l in PARTICLE_LAYERS for h in range(8)]
        for _ in range(2 if quick else n_random):
            ks = list(rng.choice(all_ph, top_k, replace=False))
            p = predict(ad, store, idx, device, bs, make_ctrl=lambda b, P, mk, ks=ks: Ctrl(zero_heads=zs(ks)))
            rand.append(_pair_metrics(base, p, labels, pairs[:1], 0, rng)[key]["d_auc"])
        res["top_vs_random"] = {"top_heads": top, "d_auc_top": top_m, "d_auc_random": rand}
    # attention knockout of displaced tracks vs pT-matched prompt charged tracks
    def ko_masks(b, P, control):
        ph_ = particle_physics(store, b, P)
        ko = ph_["displaced"].copy()
        if control:
            ctl = np.zeros_like(ko)
            for j in range(len(b)):
                cand = np.where(ph_["charged"][j] & ~ph_["displaced"][j])[0]
                for i in np.where(ko[j])[0]:
                    if len(cand) == 0:
                        break
                    k = np.argmin(np.abs(ph_["ln_pt"][j, cand] - ph_["ln_pt"][j, i]))
                    ctl[j, cand[k]] = True
                    cand = np.delete(cand, k)
            ko = ctl
        ko &= ~(ko.sum(1, keepdims=True) >= ph_["mask"].sum(1, keepdims=True))   # never remove everything
        return torch.from_numpy(ko)
    for where, layers_ko in [("cls", set(CLS_LAYERS)), ("all", set(PARTICLE_LAYERS + CLS_LAYERS))]:
        for control in (False, True):
            p = predict(ad, store, idx, device, bs, make_ctrl=lambda b, P, mk, c=control, w=layers_ko:
                        Ctrl(key_knockout=ko_masks(b, P, c).to(mk.device), knockout_layers=w))
            res["knockout"][f"{where}_{'ptmatched_prompt' if control else 'displaced'}"] = \
                _pair_metrics(base, p, labels, pairs, n_boot, rng)
    res["seconds"] = time.time() - t0
    return res, base


@torch.no_grad()
def patching(ad, store, idx, device, sig, bkg, n_boot, rng, bs=128, quick=False):
    """Clean = real signal jet; corrupt = same jet with the impact parameters of its
    displaced tracks set to zero (tanh d0 = tanh dz = 0). Metric m = log p_sig - log p_bkg.
    Recovery = (m_patched - m_corrupt) / (m_clean - m_corrupt), averaged as a ratio of means."""
    from .data import FEAT
    s, b_ = CLASS_LIST.index(sig), CLASS_LIST.index(bkg)
    idx = np.asarray(idx)
    idx = idx[store.labels[idx] == s]
    ndisp = np.array([particle_physics(store, [i], int(store.mask[i].sum()))["displaced"].sum() for i in idx])
    idx = idx[ndisp > 0]
    heads = [(l, h) for l in PARTICLE_LAYERS + CLS_LAYERS for h in range(8)]
    if quick:
        heads = heads[::9]
    resid_pts = list(range(len(PARTICLE_LAYERS) + 1))
    names = ["clean", "corrupt"] + [f"resid_disp.{i}" for i in resid_pts] + \
            [f"resid_rest.{i}" for i in resid_pts] + [f"head.{l}.{h}" for l, h in heads]
    m = {k: [] for k in names}
    for b, P in _batches(store, idx, bs):
        f, v, mk = ad.inputs(store, b, P, device)
        disp = torch.from_numpy(particle_physics(store, b, P)["displaced"]).to(device)
        fc = f.clone()
        for k in (FEAT["part_d0"], FEAT["part_dz"]):
            fc[:, k] = fc[:, k].masked_fill(disp, 0.0)
        met = lambda lo: (torch.log_softmax(lo.float(), -1)[:, s] - torch.log_softmax(lo.float(), -1)[:, b_]).cpu().numpy()
        rec = Ctrl(record={"resid", "ctx"})
        m["clean"].append(met(ad.forward(f, v, mk, rec)))
        m["corrupt"].append(met(ad.forward(fc, v, mk)))
        rest = mk & ~disp
        for i in resid_pts:
            clean_r = rec.out[f"resid/L{i}"]
            m[f"resid_disp.{i}"].append(met(ad.forward(fc, v, mk, Ctrl(patch_resid={i: (clean_r, disp)}))))
            m[f"resid_rest.{i}"].append(met(ad.forward(fc, v, mk, Ctrl(patch_resid={i: (clean_r, rest)}))))
        for l, h in heads:
            m[f"head.{l}.{h}"].append(met(ad.forward(fc, v, mk, Ctrl(patch_ctx={(l, h): rec.out[f"ctx/{l}"][:, h]}))))
    m = {k: np.concatenate(v) for k, v in m.items()}
    J = len(m["clean"])
    gap = m["clean"] - m["corrupt"]

    def rec_frac(k):
        num = (m[k] - m["corrupt"])[:, None]
        est, lo, hi = boot_ratio(num, gap[:, None], n_boot, rng)
        return {"est": float(est[0]), "lo": float(lo[0]), "hi": float(hi[0])}
    out = {"n_jets": J, "m_clean": float(m["clean"].mean()), "m_corrupt": float(m["corrupt"].mean()),
           "auc_like_flip_frac": float((m["corrupt"] < 0).mean() - (m["clean"] < 0).mean()),
           "resid_disp": [rec_frac(f"resid_disp.{i}") for i in resid_pts],
           "resid_rest": [rec_frac(f"resid_rest.{i}") for i in resid_pts],
           "heads": {f"{l}.{h}": rec_frac(f"head.{l}.{h}") for l, h in heads}}
    return out


# ----------------------------------------------------------------------------
# 5. linear probes
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_resid(ad, store, idx, device, max_particles_total, rng, bs=64):
    per_jet = max(1, max_particles_total // max(len(idx), 1))
    X = {i: [] for i in range(len(PARTICLE_LAYERS) + 1)}
    T = {"displaced": [], "lepton": [], "disp_neighbour": [], "charged": [], "jet": []}
    off = 0
    for b, P in _batches(store, idx, bs):
        f, v, mk = ad.inputs(store, b, P, device)
        ctrl = Ctrl(record={"resid"})
        ad.forward(f, v, mk, ctrl)
        ph = particle_physics(store, b, P)
        jb, pb = [], []
        for j in range(len(b)):
            real = np.where(ph["mask"][j])[0]
            pick = rng.choice(real, min(per_jet, len(real)), replace=False)
            jb += [j] * len(pick)
            pb += list(pick)
        jb, pb = np.array(jb), np.array(pb)
        for i in X:
            X[i].append(ctrl.out[f"resid/L{i}"][torch.from_numpy(jb), torch.from_numpy(pb)].float().cpu().numpy())
        T["displaced"].append(ph["displaced"][jb, pb])
        T["lepton"].append(ph["lepton"][jb, pb])
        T["disp_neighbour"].append(ph["n_disp_near"][jb, pb] >= 1)
        T["charged"].append(ph["charged"][jb, pb])
        T["jet"].append(off + jb)
        off += len(b)
    return {i: np.concatenate(x) for i, x in X.items()}, {k: np.concatenate(v) for k, v in T.items()}


def probes(Xs, T, n_boot, rng, shuffle_control=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    jet = T["jet"]
    test_jets = rng.random(jet.max() + 1) < 0.3
    te = test_jets[jet]
    tasks = {"displaced (charged only)": (T["displaced"], T["charged"]),
             "lepton": (T["lepton"], np.ones_like(T["lepton"])),
             "displaced track within dR<0.4": (T["disp_neighbour"], np.ones_like(T["lepton"]))}
    out = {}
    for name, (y, use) in tasks.items():
        y, use = y.astype(bool), use.astype(bool)
        out[name] = {"auc": [], "lo": [], "hi": [], "shuffled_auc": [], "pos_frac": float(y[use].mean())}
        for i in sorted(Xs):
            X = Xs[i]
            tr_, te_ = use & ~te, use & te
            if y[tr_].sum() < 5 or (~y[tr_]).sum() < 5 or y[te_].sum() < 2:
                for k in ("auc", "lo", "hi", "shuffled_auc"):
                    out[name][k].append(float("nan"))
                continue
            sc = StandardScaler().fit(X[tr_])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(X[tr_]), y[tr_])
            s = clf.decision_function(sc.transform(X[te_]))
            a = auc(s, y[te_])
            bs = [auc(s[k], y[te_][k]) for k in (rng.integers(0, len(s), len(s)) for _ in range(n_boot))]
            out[name]["auc"].append(a)
            lo, hi = ci(bs)
            out[name]["lo"].append(lo)
            out[name]["hi"].append(hi)
            if shuffle_control:
                ys = rng.permutation(y[tr_])
                clf2 = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(X[tr_]), ys)
                out[name]["shuffled_auc"].append(auc(clf2.decision_function(sc.transform(X[te_])), y[te_]))
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run_all(ad, store, out_dir=None, batch_size=128, device="cpu", n_boot=200, signal="Hbb", background="QCD",
            n_attn_per_class=2000, n_causal_per_class=2000, n_probe_per_class=1000, probe_max=60000,
            control=None, quick=False, seed=0, model_name=None):
    rng = np.random.default_rng(seed)
    present = [c for c in CLASS_LIST if (store.labels == CLASS_LIST.index(c)).any()]
    others = [c for c in present if c not in (signal, background)]
    pairs = [(signal, background)] + [(c, background) for c in others]
    timings = {}
    res = {"model": model_name or getattr(ad, "name", "model"), "classes": present, "signal": signal,
           "background": background, "sig_cut": SIG_CUT, "n_boot": n_boot}

    t = time.time()
    all_idx = np.arange(len(store))
    probs = predict(ad, store, all_idx, device, batch_size * 2)
    res["performance"] = performance(probs, store.labels, pairs)
    timings["performance"] = time.time() - t
    print(f"[perf] {json.dumps(res['performance']['pairs'].get(f'{signal}_vs_{background}'))} "
          f"acc10={res['performance']['accuracy_10way']:.4f}", flush=True)

    t = time.time()
    idx_a = _pick(store.labels, present, n_attn_per_class, rng)
    rec = recording_pass(ad, store, idx_a, device, bs=max(8, batch_size // 3), rng=rng)
    res["cls_attention"] = analyse_cls(rec["cls"], rec["labels"], present, n_boot, rng)
    res["head_pairwise"], res["bias_vs_qk"] = analyse_pairs(rec, present, n_boot, rng, signal, background)
    timings["recording"] = time.time() - t
    print(f"[attn] done in {timings['recording']:.0f}s", flush=True)

    t = time.time()
    idx_c = _pick(store.labels, present, n_causal_per_class, rng)
    res["ablation"], _ = ablation(ad, store, idx_c, device, pairs, n_boot, rng, bs=batch_size * 2, quick=quick)
    timings["ablation"] = time.time() - t
    print(f"[ablation] done in {timings['ablation']:.0f}s", flush=True)

    t = time.time()
    res["patching"] = patching(ad, store, idx_c, device, signal, background, n_boot, rng, bs=batch_size, quick=quick)
    timings["patching"] = time.time() - t
    print(f"[patching] done in {timings['patching']:.0f}s", flush=True)

    t = time.time()
    idx_p = _pick(store.labels, present, n_probe_per_class, rng)
    Xs, T = collect_resid(ad, store, idx_p, device, probe_max, rng, bs=batch_size // 2)
    res["probes"] = {"trained": probes(Xs, T, n_boot, rng)}
    if control is not None:
        Xc, Tc = collect_resid(control, store, idx_p, device, probe_max, np.random.default_rng(seed), bs=batch_size // 2)
        res["probes"]["random_init"] = probes(Xc, Tc, n_boot, rng, shuffle_control=False)
    timings["probes"] = time.time() - t
    print(f"[probes] done in {timings['probes']:.0f}s", flush=True)
    res["timings_s"] = timings

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(_jsonable(res), fh, indent=1)
        from . import plots
        plots.make_all(res, out_dir)
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o
