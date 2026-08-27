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
    # --- the `offload` tier (R1 reply 4) -------------------------------------
    # NOT a commercial API. What the env consumes is an accuracy, a decay
    # constant, and a modelled network-latency/radio-energy cost -- none of
    # which require the model to be remote or commercial. The definition that
    # matters is "too large to reside on the edge device, reached over a
    # network", and a 7B Q4_K_M at ~4.7 GB is exactly that on a 4 GB card.
    #
    # Measuring it locally is also a *better* experiment: keeping every tier on
    # one llama.cpp path means an accuracy difference is model scale, not a
    # different tokenizer, sampler and safety layer.
    "offload": {
        # Single-file build. Qwen's own repo ships this quant split across two
        # shards, which would need assembling before use -- one more step that
        # can silently produce a plausible-looking wrong file.
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "approx_mb": 4680,
        "role": "the offload tier; too large for the device, partial CPU offload",
    },
    # --- quantisation ablation (R1 reply 3 §6) -------------------------------
    # Is `think_big` at 0.603 a statement about 3B-scale models, or about what
    # Q4_K_M does to one? Same model, near-lossless quant, same prompt/parser/
    # permutations. NOTE: ~3.4 GB of weights against ~3.3 GB free VRAM, so this
    # one will NOT be fully GPU-resident and is *not* a valid hardware anchor.
    # That is fine here: partial CPU offload changes speed, not outputs, and
    # this run measures accuracy only (docs/R2_HANDOFF.md §5.1).
    "think_big_q8": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "quant": "Q8_0",
        "approx_mb": 3420,
        "role": "quantisation ablation; accuracy only, not a device anchor",
    },
    # A middle point on *scale* at a good quant, to separate scale from quant.
    # Fits fully in VRAM, so it doubles as a valid anchor if wanted.
    "think_mid": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q8_0.gguf",
        "quant": "Q8_0",
        "approx_mb": 1650,
        "role": "scale ablation between 0.5B and 3B, fully GPU-resident",
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

    # A .part is only resumable if it is a prefix of *this* object. Nothing
    # checked that, and the consequence was not a corrupt download but a
    # perfectly plausible one: a Q8 fetch aimed at the Q4 filename found a
    # complete 2.02 GB Q4 file, compared it against Q8's 3.42 GB length,
    # concluded "truncated", and resumed -- appending the tail of Q8 to the
    # head of Q4. The result had the exact expected byte count, a valid GGUF
    # magic, and a header that read Q4_K_M. Every check passed. The identity of
    # the partial file has to be recorded, not inferred from its length.
    side = tmp + ".json"
    ident = {"repo": spec["repo"], "file": spec["file"], "expected": remote_size(spec)}
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if have:
        prev = None
        if os.path.exists(side):
            try:
                with open(side, encoding="utf-8") as fh:
                    prev = json.load(fh)
            except Exception:
                prev = None
        if prev != ident:
            print(f"  .part belongs to a different object ({prev}); discarding "
                  f"{have / 1e6:.0f} MB rather than splicing")
            os.remove(tmp)
            have = 0
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(ident, fh)

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
    if os.path.exists(side):
        os.remove(side)
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


def remote_sha256(spec: Dict[str, object]) -> Optional[str]:
    """The file's true SHA-256, from Hugging Face's LFS metadata.

    Length is a weak fingerprint -- a spliced file can match it exactly. The
    LFS object id is the content hash, so comparing against it catches any
    corruption, whatever produced it.
    """
    api = f"{HF}/api/models/{spec['repo']}/tree/main?recursive=1"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "cogniwatt-r2"})
        with urllib.request.urlopen(req, timeout=30) as r:
            for e in json.loads(r.read().decode()):
                if e.get("path") == spec["file"]:
                    return (e.get("lfs") or {}).get("oid")
    except Exception:
        return None
    return None


def sha256_full(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def gguf_file_type(path: str) -> Optional[int]:
    """Read `general.file_type` from the GGUF header without loading the model.

    This is the direct question -- *which quantisation is this?* -- rather than
    a proxy for it. Reading it costs microseconds and would have caught the
    splice immediately, since the spliced file's header still said Q4_K_M while
    its name and length said Q8_0.
    """
    import struct

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        struct.unpack("<I", f.read(4))[0]                     # version
        struct.unpack("<Q", f.read(8))[0]                     # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        # GGUF value type ids -> (struct code, size); 8=string, 9=array.
        prim = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
                10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

        def skip(t: int):
            if t in prim:
                c, n = prim[t]
                return struct.unpack(c, f.read(n))[0]
            if t == 8:
                return rd_str()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip(et)
                return None
            raise ValueError(f"unknown gguf type {t}")

        for _ in range(n_kv):
            key = rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            v = skip(t)
            if key == "general.file_type":
                return int(v)
    return None


#: GGUF `general.file_type` -> quant name, for the builds this project uses.
GGUF_FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
              12: "Q3_K_M", 14: "Q4_K_S", 15: "Q4_K_M", 17: "Q5_K_M",
              18: "Q6_K"}


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
    ap.add_argument("--as-tier", action="store_true",
                    help="write to the filename the backend expects for this "
                         "tier instead of the source filename. For the quant "
                         "ladder (substituting Q4_K_S for Q4_K_M), NOT for "
                         "ablations where both quants must coexist.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a file the manifest records as a different build")
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
        # Drop entries whose file no longer exists. A manifest that asserts
        # `verified: true` for a deleted file is worse than no manifest -- it is
        # the same class of silent provenance rot that let a spliced GGUF be
        # trusted, and it makes the clash guard below reason about ghosts.
        stale = [k for k, v in manifest.items()
                 if not os.path.exists(os.path.join(args.models_dir,
                                                    v.get("local_file", "")))]
        for k in stale:
            print(f"  manifest: dropping stale entry {k!r} "
                  f"({manifest[k].get('local_file')} is gone)")
            manifest.pop(k)

    for tier in args.tiers:
        if tier not in CATALOG:
            print(f"unknown tier {tier!r}; --list to see the catalog", file=sys.stderr)
            return 1
        spec = CATALOG[tier]
        # Default to the SOURCE filename, so two quants of one model can coexist.
        #
        # This used to collapse `think_big_q8` -> `think_big` and write to the
        # tier's expected filename. That is right for the *quant ladder*, where
        # you are substituting a smaller quant for the same tier, and badly
        # wrong for an *ablation*, where you need both side by side: it wrote
        # 3.4 GB of Q8 weights over the Q4_K_M file and rewrote its provenance
        # in the manifest. Nothing warned, because the file was a valid GGUF of
        # exactly the expected size -- for the wrong quant. Later runs would
        # have measured Q8 while every report said Q4_K_M.
        target = str(spec["file"]).lower()
        if args.as_tier:
            base = tier.split("_q")[0]
            target = TIER_MODEL_FILES.get(base, target)
        print(f"[{tier}] {spec['repo']}/{spec['file']}  ({spec['quant']}, "
              f"~{spec['approx_mb']} MB) -> {target}")

        # Refuse to silently replace a different build at the same path.
        # Quantisation is part of the identity of every accuracy and energy
        # number, so a mismatched overwrite corrupts provenance invisibly.
        clash = [k for k, v in manifest.items()
                 if v.get("local_file") == target
                 and v.get("source_file") != spec["file"]]
        if clash and not args.force:
            print(f"  REFUSING: {target} is recorded as {manifest[clash[0]]['quant']} "
                  f"({manifest[clash[0]]['source_file']}). Use --as-tier "
                  f"deliberately, or --force.", file=sys.stderr)
            return 1
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

        # Which quantisation did we actually get? Asked directly.
        ft = gguf_file_type(path)
        name = GGUF_FTYPE.get(ft, f"ftype:{ft}")
        if name != spec["quant"]:
            print(f"  QUANT MISMATCH: {path} reports {name}, catalog says "
                  f"{spec['quant']}", file=sys.stderr)
            return 1

        want_sha = remote_sha256(spec)
        if want_sha:
            got_sha = sha256_full(path)
            if got_sha != want_sha:
                print(f"  SHA MISMATCH: {got_sha[:16]} != {want_sha[:16]}",
                      file=sys.stderr)
                return 1
            print(f"  verified {name}, sha256 {got_sha[:16]}...")
        else:
            print(f"  verified {name} (remote sha unavailable)")
        manifest[tier] = {
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
