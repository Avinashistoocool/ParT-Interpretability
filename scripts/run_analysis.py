"""Run the full interpretability pipeline on one model.

    python scripts/run_analysis.py --model part --ckpt checkpoints/ParT_full.pt \
        --data_dir D:/JetClass/test --classes QCD,Hbb,Hcc,Tbqq --n_per_class 5000 --out results/part_local

    python scripts/run_analysis.py --model gds --ckpt ../gds_part/checkpoints/v2_shared_10M_10ep/best.pt ...

--random_control adds a randomly initialised model of the same architecture as a
probe baseline. --start skips the first jets of each file (use different
--start for independent samples).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from partinterp import analyses  # noqa: E402
from partinterp.data import load_jetclass  # noqa: E402
from partinterp.models import GDSAdapter, ParTAdapter, build_part, load_model  # noqa: E402


def random_control(kind, device, seed=123):
    torch.manual_seed(seed)
    if kind == "part":
        m = build_part()
        # BN running stats from init (mean 0 / var 1) are meaningless for raw pair features; keep them,
        # the control only needs an untrained network with identical architecture
        return ParTAdapter(m.to(device))
    from partinterp.vendor.gds_part_colab import GDSParT, GDSParTConfig
    return GDSAdapter(GDSParT(GDSParTConfig(per_layer_gate=False)).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["part", "gds"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", default="QCD,Hbb,Hcc,Tbqq")
    ap.add_argument("--n_per_class", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--file_index", type=int, default=0)
    ap.add_argument("--n_attn_per_class", type=int, default=2000)
    ap.add_argument("--n_causal_per_class", type=int, default=2000)
    ap.add_argument("--n_probe_per_class", type=int, default=1500)
    ap.add_argument("--probe_max", type=int, default=60000)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--signal", default="Hbb")
    ap.add_argument("--background", default="QCD")
    ap.add_argument("--random_control", action="store_true")
    ap.add_argument("--quick", action="store_true", help="subset of heads, for wiring tests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 so the decomposition is not blurred by TF32
    torch.backends.cudnn.allow_tf32 = False
    store = load_jetclass(a.data_dir, a.classes.split(","), a.n_per_class, start=a.start, file_index=a.file_index)
    ad = load_model(a.model, a.ckpt, a.device)
    ctl = random_control(a.model, a.device) if a.random_control else None
    res = analyses.run_all(ad, store, out_dir=a.out, batch_size=a.batch_size, device=a.device, n_boot=a.n_boot,
                           signal=a.signal, background=a.background, n_attn_per_class=a.n_attn_per_class,
                           n_causal_per_class=a.n_causal_per_class, n_probe_per_class=a.n_probe_per_class,
                           probe_max=a.probe_max, control=ctl, quick=a.quick, seed=a.seed,
                           model_name={"part": "ParT (official ParT_full)", "gds": "GDS-ParT v2"}[a.model])
    meta = {"args": vars(a), "total_seconds": time.time() - t0, "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"}
    with open(os.path.join(a.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {a.out} in {meta['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
