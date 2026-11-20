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
CKPT_IN_BUNDLE = {"part": "ParT_full.pt", "gds": "gds_v2_shared_10M_10ep.pt"}
CLASSES = "QCD,Hbb,Hcc,Hgg,H4q,Hqql,Zqq,Wqq,Tbqq,Tbl"
WORK = "/kaggle/working"
t0 = time.time()


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


# The kernel may run without internet (DNS to pypi/github failed on the first push), so
# everything needed is also attached as the Kaggle dataset avinashchandragupta/part-interp-bundle
# (code snapshot, Linux wheels for uproot/awkward, both checkpoints).
hits = glob.glob("/kaggle/input/**/partinterp/__init__.py", recursive=True)
bundle_code = os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))) if hits else None
bundle = os.path.dirname(bundle_code) if bundle_code else None
print("bundle", bundle, flush=True)

try:
    import uproot  # noqa: F401
    import awkward  # noqa: F401
except ImportError:
    wheels = glob.glob(f"{bundle}/**/wheels", recursive=True) if bundle else []
    if wheels:
        sh(f"pip install -q --no-index --find-links {wheels[0]} uproot awkward")
    else:
        sh("pip install -q uproot awkward")

code = None
if sh(f"git clone -q --depth 1 {REPO} {WORK}/repo", check=False) == 0:
    code = f"{WORK}/repo"
elif bundle_code:
    code = bundle_code
    print("[warn] no internet: using the code snapshot from the bundle dataset", flush=True)
else:
    raise SystemExit("no code: neither git clone nor bundle dataset available")
sys.path.insert(0, f"{code}/src")

ckpt = f"{WORK}/{MODEL}.pt"
in_bundle = glob.glob(f"{bundle}/**/{CKPT_IN_BUNDLE[MODEL]}", recursive=True) if bundle else []
if in_bundle:
    ckpt = in_bundle[0]
elif not os.path.exists(ckpt):
    urllib.request.urlretrieve(CKPT_URL[MODEL], ckpt)
print("checkpoint", ckpt, os.path.getsize(ckpt), flush=True)

roots = glob.glob("/kaggle/input/**/HToBB_*.root", recursive=True)
if roots:
    data_dir = os.path.dirname(roots[0])
else:  # fallback: one test file per class straight from Zenodo
    data_dir = f"{WORK}/jetclass_test"
    sh(f"python {code}/scripts/fetch_jetclass_test.py --out {data_dir} --classes {CLASSES} --files_per_class 2")
print("data_dir", data_dir, sorted(os.listdir(data_dir))[:12], flush=True)

common = (f"--model {MODEL} --ckpt {ckpt} --data_dir {data_dir} --batch_size 256 --n_boot 300 "
          f"--random_control --device cuda")
# main sample: 20k jets per class (all 10 classes) from the first file of each class
sh(f"python {code}/scripts/run_analysis.py {common} --classes {CLASSES} --n_per_class 20000 "
   f"--n_attn_per_class 3000 --n_causal_per_class 3000 --n_probe_per_class 2000 --probe_max 80000 "
   f"--out {WORK}/results/{MODEL}_main")
# replica: independent jets (second file of each class), Hbb/QCD/Hcc/Tbqq only, different seed
sh(f"python {code}/scripts/run_analysis.py {common} --classes QCD,Hbb,Hcc,Tbqq --file_index 1 "
   f"--n_per_class 10000 --n_attn_per_class 3000 --n_causal_per_class 3000 --n_probe_per_class 2000 "
   f"--probe_max 60000 --seed 1 --out {WORK}/results/{MODEL}_replica")
sh(f"cd {WORK} && python {code}/scripts/make_report.py results/{MODEL}_main results/{MODEL}_replica "
   f"--out results/REPORT_{MODEL}.md", check=False)
sh(f"cd {WORK} && tar czf results_{MODEL}.tgz results && rm -rf repo jetclass_test {MODEL}.pt")
print(f"total {time.time() - t0:.0f}s", flush=True)
