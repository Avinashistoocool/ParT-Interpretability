"""Kaggle kernel: full ParT-interpretability pipeline on JetClass validation jets.

Generated from kaggle/kernel_template.py by scripts/push_kaggle.ps1 (MODEL is
substituted per kernel). Data: the attached Kaggle dataset
suryatrainer/jetclass-val (JetClass val_5M ROOT files, files *_120..124);
falls back to an HTTP range download of the JetClass test tar from Zenodo
(record 6619768) if the dataset is not mounted.
"""
import glob
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "__MODEL__"          # "part" or "gds"
REPO = "https://github.com/Avinashistoocool/ParT-Interpretability.git"
CKPT_URL = {
    "part": "https://raw.githubusercontent.com/jet-universe/particle_transformer/main/models/ParT_full.pt",
    "gds": "https://raw.githubusercontent.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/main/"
           "checkpoints/v2_shared_10M_10ep/best.pt",
}
CLASSES = "QCD,Hbb,Hcc,Hgg,H4q,Hqql,Zqq,Wqq,Tbqq,Tbl"
WORK = "/kaggle/working"
t0 = time.time()


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("pip install -q uproot awkward")
if not os.path.exists(f"{WORK}/repo"):
    sh(f"git clone --depth 1 {REPO} {WORK}/repo")
sys.path.insert(0, f"{WORK}/repo/src")

ckpt = f"{WORK}/{MODEL}.pt"
if not os.path.exists(ckpt):
    urllib.request.urlretrieve(CKPT_URL[MODEL], ckpt)
print("checkpoint", ckpt, os.path.getsize(ckpt), flush=True)

roots = glob.glob("/kaggle/input/**/HToBB_*.root", recursive=True)
if roots:
    data_dir = os.path.dirname(roots[0])
else:  # fallback: one test file per class straight from Zenodo
    sys.path.insert(0, f"{WORK}/repo/scripts")
    data_dir = f"{WORK}/jetclass_test"
    sh(f"python {WORK}/repo/scripts/fetch_jetclass_test.py --out {data_dir} --classes {CLASSES}")
print("data_dir", data_dir, sorted(os.listdir(data_dir))[:12], flush=True)

common = (f"--model {MODEL} --ckpt {ckpt} --data_dir {data_dir} --batch_size 256 --n_boot 300 "
          f"--random_control --device cuda")
# main sample: 20k jets per class (all 10 classes) from the first file of each class
sh(f"python {WORK}/repo/scripts/run_analysis.py {common} --classes {CLASSES} --n_per_class 20000 "
   f"--n_attn_per_class 3000 --n_causal_per_class 3000 --n_probe_per_class 2000 --probe_max 80000 "
   f"--out {WORK}/results/{MODEL}_main")
# replica: independent jets (second file of each class), Hbb/QCD/Hcc/Tbqq only, different seed
sh(f"python {WORK}/repo/scripts/run_analysis.py {common} --classes QCD,Hbb,Hcc,Tbqq --file_index 1 "
   f"--n_per_class 10000 --n_attn_per_class 3000 --n_causal_per_class 3000 --n_probe_per_class 2000 "
   f"--probe_max 60000 --seed 1 --out {WORK}/results/{MODEL}_replica")
sh(f"cd {WORK} && tar czf results_{MODEL}.tgz results && rm -rf repo jetclass_test {MODEL}.pt")
print(f"total {time.time() - t0:.0f}s", flush=True)
