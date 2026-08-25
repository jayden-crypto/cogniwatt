#!/usr/bin/env python3
"""Fetch the GGUF weights for the local tiers.

    python scripts/r2_get_models.py --tiers think_tiny
    python scripts/r2_get_models.py --tiers think_tiny think_big
    python scripts/r2_get_models.py --list

Downloads are resumable and verified by size. Quantisation is part of the
identity of a measurement -- it changes both accuracy and energy -- so the
exact file that was downloaded is recorded in `models/MANIFEST.json` and
carried into `meta_json` by `r2_freeze.py`.

VRAM LADDER (docs/R2_HANDOFF.md §5.1)
------------------------------------
On a 4 GB card only the 3B model is tight. If Q4_K_M will not load fully on
the GPU, step down -- a lower quant is not a compromise, it is what actually
ships on 4 GB edge hardware. Just record which one was used.

    Q4_K_M  ~1.9 GB   first choice
    Q4_K_S  ~1.7 GB   if Q4_K_M will not fully load for the profiling pass
    Q3_K_M  ~1.5 GB   last resort
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

HF = "https://huggingface.co"

CATALOG: Dict[str, Dict[str, object]] = {
    "think_tiny": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "quant": "Q8_0",
        "approx_mb": 640,
        "role": "the INT8 on-device tier",
    },
    "think_big": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "approx_mb": 1925,
        "role": "the on-device large tier; ~2.4 GB resident with CUDA context",
    },
    # Fallbacks for the 4 GB ladder. Same repo, smaller quants.
    "think_big_q4ks": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_S.gguf",
        "quant": "Q4_K_S",
        "approx_mb": 1740,
        "role": "fallback if Q4_K_M will not fully load",
    },
    "think_big_q3km": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q3_K_M.gguf",
        "quant": "Q3_K_M",
        "approx_mb": 1600,
        "role": "last resort on 4 GB",
    },
}


def url_for(spec: Dict[str, object]) -> str:
    return f"{HF}/{spec['repo']}/resolve/main/{spec['file']}?download=true"


def download(spec: Dict[str, object], dest_dir: str, target_name: str,
             attempts: int = 6) -> str:
    """Fetch with resume and backoff.

    Hugging Face redirects LFS objects to a signed CDN URL on a *different*
    host, and that host intermittently fails to resolve when the link is
    saturated -- which is exactly what happens if a large pip download is
    running at the same time. Retrying from the byte offset already on disk
    costs nothing and turns a 675 MB restart into a few seconds.
    """
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return _download_once(spec, dest_dir, target_name)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            wait = min(60, 2 ** i)
            print(f"\n  attempt {i + 1}/{attempts} failed ({e}); "
                  f"resuming in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def _download_once(spec: Dict[str, object], dest_dir: str, target_name: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, target_name)
    url = url_for(spec)

    tmp = dest + ".part"
    if os.path.exists(dest):
        have_b = os.path.getsize(dest)
        want = remote_size(spec)
        if want is None or have_b == want:
            print(f"  already present: {dest} ({have_b / 1e6:.0f} MB)")
            return dest
        # Demote a short file back to .part so the next pass resumes it rather
        # than re-downloading from zero.
        print(f"  present but TRUNCATED: {have_b / 1e6:.0f} of {want / 1e6:.0f} MB"
              f" -- demoting to .part and resuming")
        os.replace(dest, tmp)

    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "cogniwatt-r2"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {have / 1e6:.0f} MB")

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have else "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) + have
        got = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            el = time.time() - t0
            pct = 100.0 * got / total if total else 0.0
            print(f"\r  {got / 1e6:7.0f} / {total / 1e6:.0f} MB  {pct:5.1f}%  "
                  f"{(got - have) / 1e6 / max(el, 1e-9):5.1f} MB/s", end="", flush=True)
    print()
    # A dropped connection surfaces as a clean EOF from `read()`, which is
    # indistinguishable from "the file ended" unless the byte count is checked.
    # Without this the .part gets promoted to the real filename and you end up
    # with a truncated GGUF whose magic header still reads "GGUF" -- it loads,
    # then fails deep inside a tensor read with an unrelated-looking error.
    if total and got != total:
        raise ConnectionError(
            f"truncated: got {got} of {total} bytes "
            f"({100.0 * got / total:.1f}%) -- retrying resumes from here")
    os.replace(tmp, dest)
    return dest


def remote_size(spec: Dict[str, object]) -> Optional[int]:
    """Content-Length after following the LFS redirect, for verification."""
    req = urllib.request.Request(url_for(spec), method="HEAD",
                                 headers={"User-Agent": "cogniwatt-r2"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def sha256_head(path: str, nbytes: int = 1 << 24) -> str:
    """Hash of the first 16 MB. A full hash of a 2 GB file costs ~10 s for no
    extra confidence about *which build* this is -- the header and the first
    tensors already differ between quants and repos."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cogniwatt.dialogue.backends import TIER_MODEL_FILES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="*", default=["think_tiny"])
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in CATALOG.items():
            print(f"{k:<16} {v['quant']:<8} ~{v['approx_mb']:>5} MB  "
                  f"{v['repo']}/{v['file']}\n{'':<16} {v['role']}")
        return 0

    manifest_path = os.path.join(args.models_dir, "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    for tier in args.tiers:
        if tier not in CATALOG:
            print(f"unknown tier {tier!r}; --list to see the catalog", file=sys.stderr)
            return 1
        spec = CATALOG[tier]
        # Land on the filename the backend expects, whichever quant was chosen.
        base = tier.split("_q")[0]
        target = TIER_MODEL_FILES.get(base, str(spec["file"]))
        print(f"[{tier}] {spec['repo']}/{spec['file']}  ({spec['quant']}, "
              f"~{spec['approx_mb']} MB) -> {target}")
        path = download(spec, args.models_dir, target)
        want = remote_size(spec)
        got = os.path.getsize(path)
        if want is not None and got != want:
            print(f"  SIZE MISMATCH: {got} != {want}", file=sys.stderr)
            return 1
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                print(f"  {path} is not a GGUF file", file=sys.stderr)
                return 1
        manifest[base] = {
            "expected_bytes": want,
            "verified": want is not None and got == want,
            "catalog_key": tier, "repo": spec["repo"],
            "source_file": spec["file"], "quant": spec["quant"],
            "local_file": target,
            "size_bytes": os.path.getsize(path),
            "sha256_first_16mb": sha256_head(path),
            "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"  ok: {path} ({os.path.getsize(path) / 1e6:.0f} MB)")

    os.makedirs(args.models_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
