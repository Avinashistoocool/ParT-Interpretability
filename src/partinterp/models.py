"""Instrumented forward passes for ParT (official weaver-core implementation) and
GDS-ParT.

Both adapters re-run the model with the *same modules and weights*, but compute
every attention explicitly so that for each layer and head we can

  * record the content term  QK^T/sqrt(d)   ("qk"),
  * record the pair-interaction bias  psi(U)  ("bias"),
  * record the attention probabilities  softmax(qk + bias + padding)  ("attn"),
  * record per-head context vectors ("ctx") and the residual stream ("resid"),
  * intervene: drop the bias or the QK term, zero/uniform-ablate heads,
    knock out keys, patch head contexts or residual-stream vectors.

tests/test_hooks.py checks that with no intervention the instrumented forward
reproduces the original model's logits and that qk + bias reconstructs the
pre-softmax logits exactly.

Layer names: "L0".."L7" are particle self-attention blocks, "C0","C1" the
class-attention blocks (single CLS query; key 0 is the CLS token itself).
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

PARTICLE_LAYERS = [f"L{i}" for i in range(8)]
CLS_LAYERS = ["C0", "C1"]


@dataclass
class Ctrl:
    record: set = field(default_factory=set)          # subset of {"qk","bias","attn","ctx","resid","cls"}
    record_layers: set = None                           # None = all layers
    drop_bias: set = field(default_factory=set)        # layer names where psi(U) is set to 0
    drop_qk: set = field(default_factory=set)          # layer names where QK^T is set to 0
    zero_heads: set = field(default_factory=set)       # {(layer, head)}: head context := 0
    uniform_heads: set = field(default_factory=set)    # {(layer, head)}: attention := uniform over real keys
    key_knockout: torch.Tensor = None                   # (B, P) bool, True = key removed
    knockout_layers: set = field(default_factory=set)  # layers where key_knockout applies
    patch_ctx: dict = field(default_factory=dict)      # {(layer, head): (B, Pq, hd) tensor}
    patch_resid: dict = field(default_factory=dict)    # {block index: ((B,P,D) tensor, (B,P) bool)}
    out: dict = field(default_factory=dict)

    def want(self, kind, layer):
        return kind in self.record and (self.record_layers is None or layer in self.record_layers)


def attend(ctrl, name, q, k, v, bias, keymask):
    """q: (B,H,Pq,hd); k, v: (B,H,Pk,hd); bias: (B,H,Pq,Pk) or None; keymask: (B,Pk) True = real."""
    B, H, Pq, hd = q.shape
    qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
    b = bias if bias is not None else torch.zeros_like(qk)
    if name in ctrl.drop_bias:
        b = torch.zeros_like(b)
    qk_used = torch.zeros_like(qk) if name in ctrl.drop_qk else qk
    logits = qk_used + b
    keep = keymask[:, None, None, :]
    if ctrl.key_knockout is not None and name in ctrl.knockout_layers:
        ko = ctrl.key_knockout
        if ko.shape[1] == keymask.shape[1] - 1:  # CLS layers: key 0 is the CLS token
            ko = torch.cat([torch.zeros_like(ko[:, :1]), ko], 1)
        keep = keep & ~ko[:, None, None, :]
    logits = logits.masked_fill(~keep, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    for (ln, h) in ctrl.uniform_heads:
        if ln == name:
            u = keep[:, 0, 0, :].float()
            attn[:, h] = (u / u.sum(-1, keepdim=True).clamp(min=1))[:, None, :].expand(B, Pq, -1)
    if ctrl.want("qk", name):
        ctrl.out[f"qk/{name}"] = qk
    if ctrl.want("bias", name) and bias is not None:
        ctrl.out[f"bias/{name}"] = bias
    if ctrl.want("attn", name):
        ctrl.out[f"attn/{name}"] = attn
    ctx = torch.matmul(attn, v)
    for (ln, h) in ctrl.zero_heads:
        if ln == name:
            ctx[:, h] = 0
    for (ln, h), val in ctrl.patch_ctx.items():
        if ln == name:
            ctx[:, h] = val
    if ctrl.want("ctx", name):
        ctrl.out[f"ctx/{name}"] = ctx
    return ctx


def _patch(ctrl, i, x_bpd):
    if i in ctrl.patch_resid:
        val, m = ctrl.patch_resid[i]
        x_bpd = torch.where(m[..., None], val.to(x_bpd.dtype), x_bpd)
    if ctrl.want("resid", f"L{i}"):
        ctrl.out[f"resid/L{i}"] = x_bpd
    return x_bpd


# ----------------------------------------------------------------------------
# ParT (weaver-core)
# ----------------------------------------------------------------------------
class ParTAdapter:
    """Wraps weaver's ParticleTransformer (``model.mod`` in the official wrapper).
    Inputs follow weaver: feats (B,17,P), vectors (B,4,P), mask (B,P) bool."""
    name = "ParT"

    def __init__(self, part):
        self.m = part.eval()

    @staticmethod
    def _mha_proj(attn, x_q, x_kv):
        D = attn.embed_dim
        W, b = attn.in_proj_weight, attn.in_proj_bias
        q = F.linear(x_q, W[:D], b[:D])
        k = F.linear(x_kv, W[D:2 * D], b[D:2 * D])
        v = F.linear(x_kv, W[2 * D:], b[2 * D:])
        return q, k, v

    @staticmethod
    def _heads(t, H):  # (P,B,D) -> (B,H,P,hd)
        P, B, D = t.shape
        return t.view(P, B, H, D // H).permute(1, 2, 0, 3)

    def _block_tail(self, blk, attn_ctx, residual):
        # identical to weaver Block.forward after self.attn(...)
        x = attn_ctx
        if blk.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, blk.num_heads, blk.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, blk.c_attn)
            x = x.reshape(tgt_len, -1, blk.embed_dim)
        if blk.post_attn_norm is not None:
            x = blk.post_attn_norm(x)
        x = blk.dropout(x)
        x = x + residual
        residual = x
        x = blk.pre_fc_norm(x)
        x = blk.act(blk.fc1(x))
        x = blk.act_dropout(x)
        if blk.post_fc_norm is not None:
            x = blk.post_fc_norm(x)
        x = blk.fc2(x)
        x = blk.dropout(x)
        if blk.w_resid is not None:
            residual = torch.mul(blk.w_resid, residual)
        return x + residual

    def _ctx_to_seq(self, ctx, attn_mod):  # (B,H,P,hd) -> out_proj((P,B,D))
        B, H, P, hd = ctx.shape
        return attn_mod.out_proj(ctx.permute(2, 0, 1, 3).reshape(P, B, H * hd))

    def forward(self, feats, vectors, mask, ctrl=None):
        ctrl = ctrl or Ctrl()
        m = self.m
        mask3 = mask[:, None, :]
        x = m.embed(feats).masked_fill(~mask3.permute(2, 0, 1), 0)      # (P,B,C)
        bias = m.pair_embed(vectors, None)                               # (B,H,P,P)
        B, P = mask.shape
        H = m.blocks[0].num_heads
        bias = bias.view(B, H, P, P)
        for i, blk in enumerate(m.blocks):
            name = f"L{i}"
            x = _patch(ctrl, i, x.transpose(0, 1)).transpose(0, 1)
            residual = x
            u = blk.pre_attn_norm(x)
            q, k, v = self._mha_proj(blk.attn, u, u)
            ctx = attend(ctrl, name, self._heads(q, H), self._heads(k, H), self._heads(v, H), bias, mask)
            x = self._block_tail(blk, self._ctx_to_seq(ctx, blk.attn), residual)
        x = _patch(ctrl, len(m.blocks), x.transpose(0, 1)).transpose(0, 1)
        cls = m.cls_token.expand(1, B, -1)
        keymask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        for c, blk in enumerate(m.cls_blocks):
            name = f"C{c}"
            residual = cls
            u = blk.pre_attn_norm(torch.cat((cls, x), 0))
            q, k, v = self._mha_proj(blk.attn, cls, u)
            ctx = attend(ctrl, name, self._heads(q, H), self._heads(k, H), self._heads(v, H), None, keymask)
            cls = self._block_tail(blk, self._ctx_to_seq(ctx, blk.attn), residual)
        x_cls = m.norm(cls).squeeze(0)
        if ctrl.want("cls", "C1"):
            ctrl.out["cls"] = x_cls
        return m.fc(x_cls)

    def reference(self, feats, vectors, mask):
        return self.m(feats, v=vectors, mask=mask[:, None, :].float())

    @staticmethod
    def inputs(store, idx, P, device):
        f = torch.from_numpy(store.feats[idx, :P]).to(device).transpose(1, 2).contiguous()
        v = torch.from_numpy(store.vectors[idx, :P]).to(device).transpose(1, 2).contiguous()
        mk = torch.from_numpy(store.mask[idx, :P]).to(device)
        return f, v, mk


def build_part(num_classes=10, input_dim=17, **over):
    """Official ParT_full configuration (particle_transformer/networks/example_ParticleTransformer.py)."""
    from .vendor.ParticleTransformer import ParticleTransformer
    cfg = dict(input_dim=input_dim, num_classes=num_classes, pair_input_dim=4, use_pre_activation_pair=False,
               embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
               num_cls_layers=2, block_params=None,
               cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
               fc_params=[], activation='gelu', trim=False, for_inference=False)
    cfg.update(over)
    return ParticleTransformer(**cfg)


def load_part(path, device="cpu"):
    sd = torch.load(path, map_location="cpu")
    sd = {k[len("mod."):]: v for k, v in sd.items() if k.startswith("mod.")}
    model = build_part()
    model.load_state_dict(sd)
    return ParTAdapter(model.to(device))


class ParTKinAdapter(ParTAdapter):
    """Official ParT_kin: only the seven kinematic features from JetClass_kin.yaml."""

    @staticmethod
    def inputs(store, idx, P, device):
        f, v, mk = ParTAdapter.inputs(store, idx, P, device)
        return f[:, [0, 1, 2, 3, 4, 15, 16]], v, mk


def load_part_kin(path, device="cpu"):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[len("mod."):]: v for k, v in sd.items() if k.startswith("mod.")}
    model = build_part(input_dim=7)
    model.load_state_dict(sd)
    return ParTKinAdapter(model.to(device))


# ----------------------------------------------------------------------------
# GDS-ParT
# ----------------------------------------------------------------------------
# GDS-ParT cache order: pt_log, e_log, logptrel, logerel, deltaR, deta, dphi, charge,
# 5 x pid, tanh d0, d0err, tanh dz, dzerr.  ParT order puts deta/dphi last.
PART_TO_GDS = [0, 1, 2, 3, 4, 15, 16, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def pairwise_features(vectors, mask, eps=1e-8):
    """GDS-ParT pairwise inputs (B,N,N,4): ln dR, ln kT, ln z, ln m2; zero on
    self-pairs and padding (identical to gds_part data/jetclass_cache.py).
    vectors: (B,N,4)."""
    v = vectors.float()
    px, py, pz, e = v.unbind(-1)
    pt = torch.sqrt(px ** 2 + py ** 2).clamp(min=eps)
    rap = 0.5 * torch.log(((e + pz).clamp(min=eps)) / ((e - pz).clamp(min=eps)))
    phi = torch.atan2(py, px)
    drap = rap[:, :, None] - rap[:, None, :]
    dphi = torch.remainder(phi[:, :, None] - phi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
    delta = torch.sqrt(drap ** 2 + dphi ** 2).clamp(min=eps)
    pt_min = torch.minimum(pt[:, :, None], pt[:, None, :])
    kt = (pt_min * delta).clamp(min=eps)
    z = (pt_min / (pt[:, :, None] + pt[:, None, :])).clamp(min=eps)
    s = v[:, :, None, :] + v[:, None, :, :]
    m2 = (s[..., 3] ** 2 - s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2).clamp(min=eps)
    out = torch.stack([torch.log(delta), torch.log(kt), torch.log(z), torch.log(m2)], dim=-1)
    n = mask.shape[1]
    valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=mask.device)
    return out * valid[..., None]


class GDSAdapter:
    """GDS-ParT: attention logits = (gate_q*Q)(gate_k*K)^T/sqrt(d) + psi(U).
    Inputs (ParT order) are converted internally; shapes follow ParTAdapter.inputs."""
    name = "GDS-ParT"

    def __init__(self, model):
        self.m = model.eval()

    def forward(self, feats, vectors, mask, ctrl=None):
        ctrl = ctrl or Ctrl()
        m = self.m
        pf = feats.transpose(1, 2)[..., PART_TO_GDS].clamp(-5, 5)
        vec = vectors.transpose(1, 2)
        pair = pairwise_features(vec, mask)
        x = m.particle_embed(pf)
        rel = m.pairwise_embed(pair)
        bias = m.pair_bias(rel).permute(0, 3, 1, 2) if m.pair_bias is not None else None
        gates = m.rel_gate(rel, key_padding_mask=mask) if m.rel_gate is not None else None
        B, P, D = x.shape
        H = m.cfg.n_heads
        split = lambda t: t.view(B, -1, H, D // H).transpose(1, 2)
        for i, blk in enumerate(m.blocks):
            name = f"L{i}"
            x = _patch(ctrl, i, x)
            h = blk.norm1(x)
            if blk.gate is not None:
                gq, gk = blk.gate(rel, mask, x=h)
            else:
                gq, gk = gates if gates is not None else (None, None)
            a = blk.attn
            q, k = a.q_proj(h), a.k_proj(h)
            if gq is not None:
                q, k = q * gq, k * gk
            ctx = attend(ctrl, name, split(q), split(k), split(a.v_proj(h)), bias, mask)
            x = x + a.out_proj(ctx.transpose(1, 2).reshape(B, P, D))
            x = x + blk.ffn(blk.norm2(x))
        x = _patch(ctrl, len(m.blocks), x)
        cls = m.cls_token.expand(B, -1, -1)
        keymask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        for c, blk in enumerate(m.cls_blocks):
            kv = blk.norm1(torch.cat([cls, x], 1))
            mha = blk.attn
            W, b = mha.in_proj_weight, mha.in_proj_bias
            q = F.linear(kv[:, :1], W[:D], b[:D])
            k = F.linear(kv, W[D:2 * D], b[D:2 * D])
            v = F.linear(kv, W[2 * D:], b[2 * D:])
            ctx = attend(ctrl, f"C{c}", split(q), split(k), split(v), None, keymask)
            cls = cls + mha.out_proj(ctx.transpose(1, 2).reshape(B, 1, D))
            cls = cls + blk.ffn(blk.norm2(cls))
        x_cls = m.final_norm(cls.squeeze(1))
        if ctrl.want("cls", "C1"):
            ctrl.out["cls"] = x_cls
        return m.classifier(x_cls)

    def reference(self, feats, vectors, mask):
        pf = feats.transpose(1, 2)[..., PART_TO_GDS].clamp(-5, 5)
        return self.m(pf, pairwise_features(vectors.transpose(1, 2), mask), mask)

    inputs = staticmethod(ParTAdapter.inputs)


def load_gds(path, device="cpu"):
    from .vendor.gds_part_colab import GDSParT, GDSParTConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GDSParT(GDSParTConfig(**ck["config"]))
    model.load_state_dict(ck["model_state"])
    return GDSAdapter(model.to(device))


def load_model(kind, path, device):
    return load_part(path, device) if kind.lower() == "part" else load_gds(path, device)
