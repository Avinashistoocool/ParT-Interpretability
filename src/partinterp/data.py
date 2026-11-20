"""JetClass loading with the exact preprocessing of the official ParT_full model
(particle_transformer/data/JetClass/JetClass_full.yaml), plus the raw per-particle
physics needed by the analyses.

Jets are stored zero-padded to `max_particles` (128). Padding never influences a
real particle or the CLS token because every attention masks padded keys; the
unit tests check this.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import CLASS_FILE_PREFIX, CLASS_LIST

# ParT_full input order (JetClass_full.yaml, pf_features)
PART_FEATURES = ["part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR",
                 "part_charge", "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
                 "part_isElectron", "part_isMuon", "part_d0", "part_d0err", "part_dz", "part_dzerr",
                 "part_deta", "part_dphi"]
FEAT = {n: i for i, n in enumerate(PART_FEATURES)}

BRANCHES = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
            "part_charge", "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
            "part_isElectron", "part_isMuon", "part_d0val", "part_d0err", "part_dzval", "part_dzerr",
            "jet_pt", "jet_energy", "jet_sdmass", "jet_nparticles"]
PID_NAMES = ["ch.had", "neu.had", "photon", "electron", "muon"]


@dataclass
class JetStore:
    feats: np.ndarray      # (J, P, 17) float32, ParT_full order and normalisation
    vectors: np.ndarray    # (J, P, 4) float32 (px, py, pz, E)
    mask: np.ndarray       # (J, P) bool
    labels: np.ndarray     # (J,) int64, index into CLASS_LIST
    raw: dict = field(default_factory=dict)  # (J, P) raw particle arrays and (J,) jet arrays

    def __len__(self):
        return len(self.labels)

    def subset(self, idx):
        raw = {k: v[idx] for k, v in self.raw.items()}
        return JetStore(self.feats[idx], self.vectors[idx], self.mask[idx], self.labels[idx], raw)

    def batches(self, batch_size, idx=None):
        """Yields (indices, trimmed numpy views) with P cut to the longest jet in the batch."""
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            n = int(self.mask[b].sum(1).max())
            yield b, n


def build_features(px, py, pz, e, deta, dphi, charge, pid, d0, d0err, dz, dzerr, jet_pt, jet_e, mask):
    """All inputs (J, P) except jet_pt/jet_e (J,). Returns (J, P, 17) float32."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.hypot(px, py)
        cols = [
            np.clip((np.log(pt) - 1.7) * 0.7, -5, 5),
            np.clip((np.log(e) - 2.0) * 0.7, -5, 5),
            np.clip((np.log(pt / jet_pt[:, None]) + 4.7) * 0.7, -5, 5),
            np.clip((np.log(e / jet_e[:, None]) + 4.7) * 0.7, -5, 5),
            np.clip((np.hypot(deta, dphi) - 0.2) * 4.0, -5, 5),
            charge, pid[..., 0], pid[..., 1], pid[..., 2], pid[..., 3], pid[..., 4],
            np.tanh(d0), np.clip(d0err, 0, 1), np.tanh(dz), np.clip(dzerr, 0, 1),
            deta, dphi,
        ]
    f = np.stack(cols, -1).astype(np.float32)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f * mask[..., None]


def find_file(data_dir, cls, index=0):
    prefix = CLASS_FILE_PREFIX[cls]
    pat = re.compile(rf"^{prefix}_\d+\.root$")
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "**", "*.root"), recursive=True)
                   if pat.match(os.path.basename(f)))
    if len(files) <= index:
        raise FileNotFoundError(f"no {prefix}_*.root (index {index}) under {data_dir}")
    return files[index]


def _pad(arr, P):
    import awkward as ak
    return ak.to_numpy(ak.fill_none(ak.pad_none(arr, P, clip=True), 0)).astype(np.float32)


def load_jetclass(data_dir, classes, n_per_class, max_particles=128, start=0, file_index=0):
    """Reads `n_per_class` jets of each class from the first matching ROOT file."""
    import uproot
    parts = []
    for cls in classes:
        path = find_file(data_dir, cls, file_index)
        with uproot.open(path) as f:
            a = f["tree"].arrays(BRANCHES, entry_start=start, entry_stop=start + n_per_class)
        P = max_particles
        d = {b: _pad(a[b], P) for b in BRANCHES if b.startswith("part_")}
        jets = {b: np.asarray(a[b], dtype=np.float32) for b in BRANCHES if b.startswith("jet_")}
        mask = _pad(a["part_energy"], P) > 0
        pid = np.stack([d["part_isChargedHadron"], d["part_isNeutralHadron"], d["part_isPhoton"],
                        d["part_isElectron"], d["part_isMuon"]], -1)
        feats = build_features(d["part_px"], d["part_py"], d["part_pz"], d["part_energy"],
                               d["part_deta"], d["part_dphi"], d["part_charge"], pid,
                               d["part_d0val"], d["part_d0err"], d["part_dzval"], d["part_dzerr"],
                               jets["jet_pt"], jets["jet_energy"], mask)
        vectors = np.stack([d["part_px"], d["part_py"], d["part_pz"], d["part_energy"]], -1) * mask[..., None]
        raw = {
            "pt": np.hypot(d["part_px"], d["part_py"]), "deta": d["part_deta"], "dphi": d["part_dphi"],
            "charge": d["part_charge"], "pid": pid.argmax(-1) * mask - (~mask),
            "d0": d["part_d0val"], "d0err": d["part_d0err"], "dz": d["part_dzval"], "dzerr": d["part_dzerr"],
            "jet_pt": jets["jet_pt"], "jet_sdmass": jets["jet_sdmass"],
        }
        n = len(mask)
        parts.append(JetStore(feats, vectors.astype(np.float32), mask,
                              np.full(n, CLASS_LIST.index(cls), dtype=np.int64), raw))
        print(f"[data] {cls:5s} {n} jets from {os.path.basename(path)}", flush=True)
    return concat(parts)


def concat(stores):
    raw = {k: np.concatenate([s.raw[k] for s in stores]) for k in stores[0].raw}
    return JetStore(np.concatenate([s.feats for s in stores]), np.concatenate([s.vectors for s in stores]),
                    np.concatenate([s.mask for s in stores]), np.concatenate([s.labels for s in stores]), raw)


def synthetic_store(n_jets=32, max_particles=40, seed=0, classes=("QCD", "Hbb")):
    """Random but physically shaped jets for CPU tests (no ROOT files needed)."""
    rng = np.random.default_rng(seed)
    P = max_particles
    nconst = rng.integers(5, P + 1, n_jets)
    mask = np.arange(P)[None] < nconst[:, None]
    pt = np.sort(rng.exponential(10.0, (n_jets, P)) + 0.5, axis=1)[:, ::-1] * mask
    deta, dphi = rng.normal(0, 0.15, (2, n_jets, P)) * mask
    eta0, phi0 = rng.normal(0, 1, n_jets), rng.uniform(-np.pi, np.pi, n_jets)
    eta, phi = eta0[:, None] + deta, phi0[:, None] + dphi
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2) * mask
    pidx = rng.choice(5, (n_jets, P), p=[0.55, 0.1, 0.3, 0.025, 0.025])
    pid = np.eye(5, dtype=np.float32)[pidx] * mask[..., None]
    charged = (pidx == 0) | (pidx >= 3)
    charge = np.where(charged, rng.choice([-1.0, 1.0], (n_jets, P)), 0.0) * mask
    labels = rng.integers(0, len(classes), n_jets)
    d0err = np.where(charged, rng.uniform(0.01, 0.06, (n_jets, P)), 0.0) * mask
    d0 = np.where(charged, rng.normal(0, 0.03, (n_jets, P)) * (1 + 10 * (labels[:, None] == 1) * (rng.random((n_jets, P)) < 0.2)), 0.0) * mask
    dz, dzerr = d0 * 1.1, d0err * 1.2
    jet_pt = np.hypot(px.sum(1), py.sum(1))
    feats = build_features(px, py, pz, e, deta, dphi, charge, pid, d0, d0err, dz, dzerr, jet_pt, e.sum(1), mask)
    vectors = np.stack([px, py, pz, e], -1).astype(np.float32) * mask[..., None]
    raw = {"pt": pt.astype(np.float32), "deta": deta, "dphi": dphi, "charge": charge, "pid": np.where(mask, pidx, -1),
           "d0": d0, "d0err": d0err, "dz": dz, "dzerr": dzerr, "jet_pt": jet_pt, "jet_sdmass": np.full(n_jets, 100.0)}
    lab = np.array([CLASS_LIST.index(c) for c in classes])[labels]
    return JetStore(feats, vectors, mask, lab, {k: np.asarray(v, dtype=np.float32) if k != "pid" else v for k, v in raw.items()})
