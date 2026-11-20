"""Download single JetClass ROOT files (100k jets of one class each) from the
JetClass test tar on Zenodo (record 6619768) using HTTP range requests, without
fetching the 20M-jet archive. Adapted from data/fetch_jetclass_subset.py in
Gated-Dual-Stream-Particle-Transformer (MIT, (c) 2026 Avinash).

    python scripts/fetch_jetclass_test.py --out data/JetClass/test --classes QCD,Hbb
"""
import argparse
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from partinterp import CLASS_FILE_PREFIX  # noqa: E402

URL = "https://zenodo.org/records/6619768/files/JetClass_Pythia_test_20M.tar"


def _range(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=120)


def index_tar():
    members, off = [], 0
    while True:
        h = _range(off, off + 511).read()
        if len(h) < 512 or h[:1] == b"\0":
            break
        name = h[:100].rstrip(b"\0").decode()
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        members.append((name, off + 512, size))
        off += 512 + ((size + 511) // 512) * 512
    return members


def download(offset, size, dest):
    tmp = dest + ".part"
    while (have := os.path.getsize(tmp) if os.path.exists(tmp) else 0) < size:
        try:
            with _range(offset + have, offset + size - 1) as r, open(tmp, "ab") as f:
                while buf := r.read(8 << 20):
                    f.write(buf)
        except Exception as e:  # resume on transient errors
            print(f"[retry] {dest}: {e}", flush=True)
            time.sleep(5)
    os.replace(tmp, dest)
    print(f"[done] {dest} ({size / 1e6:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="QCD,Hbb")
    ap.add_argument("--files_per_class", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    members = index_tar()
    for c in a.classes.split(","):
        pat = re.compile(rf"(^|/){CLASS_FILE_PREFIX[c]}_\d+\.root$")
        for name, off, size in sorted(m for m in members if pat.search(m[0]))[:a.files_per_class]:
            dest = os.path.join(a.out, os.path.basename(name))
            if not (os.path.exists(dest) and os.path.getsize(dest) == size):
                download(off, size, dest)


if __name__ == "__main__":
    main()
