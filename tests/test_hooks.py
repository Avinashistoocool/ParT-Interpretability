"""CPU smoke tests on synthetic jets with tiny randomly initialised models.

    python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partinterp.data import synthetic_store  # noqa: E402
from partinterp.models import (CLS_LAYERS, PARTICLE_LAYERS, Ctrl, GDSAdapter, ParTAdapter,  # noqa: E402
                               build_part)

torch.manual_seed(0)


def tiny_part():
    m = build_part(num_layers=8, num_heads=8, embed_dims=[32, 64, 32], pair_embed_dims=[16, 16])
    for mod in m.modules():  # non-trivial BN statistics so the test is not degenerate
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return ParTAdapter(m)


def tiny_gds():
    from partinterp.vendor.gds_part_colab import GDSParT, GDSParTConfig
    cfg = GDSParTConfig(embed_dim=32, n_heads=8, n_layers=8, ffn_dim=64, pairwise_embed_dim=16,
                        per_layer_gate=False, pair_bias=True, n_cls_layers=2)
    m = GDSParT(cfg)
    with torch.no_grad():  # move the identity-initialised gates away from 1
        for p in m.rel_gate.parameters():
            p.normal_(0, 0.2)
    return GDSAdapter(m)


@pytest.fixture(scope="module")
def batch():
    store = synthetic_store(n_jets=12, max_particles=24, seed=1)
    idx = np.arange(12)
    return ParTAdapter.inputs(store, idx, 24, "cpu")


@pytest.mark.parametrize("make", [tiny_part, tiny_gds])
def test_reproduces_reference(make, batch):
    ad = make()
    with torch.no_grad():
        ref = ad.reference(*batch)
        out = ad.forward(*batch)
    assert out.shape == (12, 10)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("make", [tiny_part, tiny_gds])
def test_hook_shapes(make, batch):
    ad = make()
    f, v, mask = batch
    B, P = mask.shape
    ctrl = Ctrl(record={"qk", "bias", "attn", "ctx", "resid", "cls"})
    with torch.no_grad():
        ad.forward(f, v, mask, ctrl)
    o = ctrl.out
    for L in PARTICLE_LAYERS:
        assert o[f"attn/{L}"].shape == (B, 8, P, P)
        assert o[f"qk/{L}"].shape == (B, 8, P, P)
        assert o[f"bias/{L}"].shape == (B, 8, P, P)
        assert o[f"ctx/{L}"].shape[:3] == (B, 8, P)
        assert o[f"resid/L{PARTICLE_LAYERS.index(L)}"].shape[:2] == (B, P)
    for L in CLS_LAYERS:
        assert o[f"attn/{L}"].shape == (B, 8, 1, P + 1)
        assert f"bias/{L}" not in o
    # attention rows are distributions over real keys only
    a = o["attn/L3"]
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))
    assert float(a.masked_select(~mask[:, None, None, :].expand_as(a)).abs().max()) == 0.0
    c = o["attn/C1"][:, :, 0, 1:]
    assert float(c.masked_select(~mask[:, None, :].expand_as(c)).abs().max()) == 0.0


@pytest.mark.parametrize("make", [tiny_part, tiny_gds])
def test_bias_plus_qk_reconstructs_logits(make, batch):
    """softmax(qk + bias) over real keys must equal the recorded attention, and the
    logit decomposition must be exact (bias is the psi(U) term, qk the content term)."""
    ad = make()
    f, v, mask = batch
    ctrl = Ctrl(record={"qk", "bias", "attn"})
    with torch.no_grad():
        ad.forward(f, v, mask, ctrl)
    for L in PARTICLE_LAYERS:
        logits = (ctrl.out[f"qk/{L}"] + ctrl.out[f"bias/{L}"]).masked_fill(~mask[:, None, None, :], float("-inf"))
        torch.testing.assert_close(torch.softmax(logits, -1), ctrl.out[f"attn/{L}"], rtol=0, atol=1e-6)
    if isinstance(ad, ParTAdapter):
        # the bias is exactly weaver's PairEmbed output, i.e. psi(U) seen by nn.MultiheadAttention
        B, P = mask.shape
        psi = ad.m.pair_embed(v, None).view(B, 8, P, P)
        torch.testing.assert_close(ctrl.out["bias/L0"], psi)
        # and the attention matches what nn.MultiheadAttention returns in block 0
        blk = ad.m.blocks[0]
        x = ad.m.embed(f).masked_fill(~mask[:, None, :].permute(2, 0, 1), 0)
        u = blk.pre_attn_norm(x)
        _, w = blk.attn(u, u, u, key_padding_mask=~mask, attn_mask=psi.reshape(B * 8, P, P),
                        need_weights=True, average_attn_weights=False)
        rows = mask[:, None, :, None].expand_as(w)  # compare real query rows only
        torch.testing.assert_close(w[rows], ctrl.out["attn/L0"][rows], rtol=1e-4, atol=1e-6)


def test_padding_invariance():
    """Adding extra padding must not change real-particle outputs."""
    ad = tiny_part()
    store = synthetic_store(n_jets=6, max_particles=20, seed=3)
    a = ParTAdapter.inputs(store, np.arange(6), 20, "cpu")
    pad = lambda t, n: torch.cat([t, torch.zeros(*t.shape[:-1], n, dtype=t.dtype)], -1)
    b = (pad(a[0], 7), pad(a[1], 7), pad(a[2], 7))
    with torch.no_grad():
        torch.testing.assert_close(ad.forward(*a), ad.forward(*b), rtol=1e-4, atol=1e-5)


def test_interventions_change_output(batch):
    ad = tiny_part()
    with torch.no_grad():
        base = ad.forward(*batch)
        for c in [Ctrl(drop_bias=set(PARTICLE_LAYERS)), Ctrl(zero_heads={("L0", 0), ("C1", 3)}),
                  Ctrl(uniform_heads={("L2", 1)}),
                  Ctrl(key_knockout=batch[2] & (torch.rand(batch[2].shape) < 0.3), knockout_layers={"C0", "C1"})]:
            out = ad.forward(*batch, c)
            assert torch.isfinite(out).all()
            assert (out - base).abs().max() > 1e-6
        # patching a head with its own clean context is the identity
        rec = Ctrl(record={"ctx"})
        ad.forward(*batch, rec)
        same = ad.forward(*batch, Ctrl(patch_ctx={("L4", 2): rec.out["ctx/L4"][:, 2]}))
        torch.testing.assert_close(same, base)


def test_analysis_pipeline_smoke(tmp_path_factory):
    """End-to-end run of every analysis on synthetic jets (tiny model, CPU)."""
    from partinterp import analyses
    ad = tiny_part()
    store = synthetic_store(n_jets=48, max_particles=24, seed=5)
    out = analyses.run_all(ad, store, out_dir=None, batch_size=16, device="cpu", n_boot=20,
                           signal="Hbb", background="QCD", probe_max=2000, quick=True)
    assert set(out) >= {"cls_attention", "head_pairwise", "bias_vs_qk", "ablation", "patching", "probes", "performance"}
    assert len(out["bias_vs_qk"]["frac_var_bias"]) == 8
