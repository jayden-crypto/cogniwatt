#!/usr/bin/env python3
"""ASR degradation of the `ask` channel -> DEFAULT_ASR_WER (R2 §6.3).

    python scripts/r2_asr.py --synth                 # build the audio corpus
    python scripts/r2_asr.py --transcribe            # run faster-whisper
    python scripts/r2_asr.py --report --figures      # WER vs SNR + operating point

Pipeline: short human answers to clarification questions -> TTS -> additive
noise at a controlled SNR -> ASR -> WER *and* referent recovery.

WER IS NOT THE METRIC THAT MATTERS
----------------------------------
The env does not consume a word error rate; it consumes "did the `ask` action
deliver the right referent". Those come apart badly. "the one on the counter"
mistranscribed as "the one on the corner" is one substitution -- a low WER --
but the referent is *lost*. "uh the red one please" -> "the red one please"
is also one error and the referent survives intact. So this script reports
both, and the operating point is chosen on **referent recovery**, with WER
reported alongside because that is what the interface field is called.

TTS WITHOUT AN INSTALL
----------------------
The handoff suggests `piper` or macOS `say`. Neither exists here, but Windows
ships SAPI (`System.Speech.Synthesis`), which is driven from PowerShell and
gives several distinct voices for free. `--tts sapi` is the default on Windows;
`--tts piper` and `--tts say` are kept for the other machines. Which engine and
which voices produced the corpus is recorded in the manifest -- a WER number is
only interpretable against the speech that produced it.

NOISE
-----
Noise is scaled to hit a target SNR against each utterance's own RMS, so the
SNR label means the same thing on a quiet voice and a loud one. Babble is
built by overlaying the corpus's own utterances, which is a far better
distractor for an ASR model than white noise -- it is speech-shaped by
construction.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_DIR = os.path.join("data", "r2_asr_audio")
MANIFEST = os.path.join("data", "r2_asr_manifest.json")
RESULTS = os.path.join("data", "r2_asr_results.json")
FIG_DIR = os.path.join("results", "figures")
SNRS = (20, 15, 10, 5, 0)
#: A living room sits around 15 dB SNR; that is the defensible operating point
#: unless the measured curve says otherwise.
OPERATING_SNR = 15
SR = 16000          # what Whisper wants anyway


# ---------------------------------------------------------------------------
# utterances
# ---------------------------------------------------------------------------
def utterances_from_scenarios(path: str, n: int, seed: int = 0) -> List[Dict]:
    """Short human answers, grounded in real scenes.

    The answer names the *distinguishing span* rather than the colour, matching
    what `clarify.py` asks and what a person would actually say.
    """
    from cogniwatt.dialogue.clarify import best_question, distinguishing_span
    from cogniwatt.dialogue.schema import load_scenarios

    scen = load_scenarios(path)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(scen), size=min(n, len(scen)), replace=False)

    out: List[Dict] = []
    for i in pick:
        s = scen[int(i)]
        descs = [c.desc for c in s.candidates]
        q = best_question(descs)
        span = distinguishing_span(descs[s.true_idx])
        # Natural short answers, the register people actually use.
        forms = [f"the one {span}", f"the {span} one" if len(span.split()) <= 2
                 else f"the one {span}", f"I mean the one {span}"]
        text = forms[int(rng.integers(0, len(forms)))]
        out.append({"scenario_id": s.id, "k": s.k, "true_idx": s.true_idx,
                    "question": q.text, "text": text,
                    "target_span": span,
                    "all_spans": [distinguishing_span(d) for d in descs]})
    return out


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def sapi_voices() -> List[str]:
    ps = ("Add-Type -AssemblyName System.Speech; "
          "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
          ".GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }")
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps],
                                      text=True, stderr=subprocess.DEVNULL)
        return [v.strip() for v in out.splitlines() if v.strip()]
    except Exception:
        return []


def sapi_synth(text: str, out_wav: str, voice: Optional[str], rate: int = 0) -> bool:
    esc = text.replace("'", "''")
    sel = f"$s.SelectVoice('{voice}'); " if voice else ""
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"{sel}$s.Rate = {rate}; "
          f"$s.SetOutputToWaveFile('{os.path.abspath(out_wav)}'); "
          f"$s.Speak('{esc}'); $s.Dispose()")
    try:
        subprocess.check_call(["powershell", "-NoProfile", "-Command", ps],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024
    except Exception:
        return False


def say_synth(text: str, out_wav: str, voice: Optional[str]) -> bool:
    cmd = ["say", "-o", out_wav, "--data-format=LEI16@16000"]
    if voice:
        cmd += ["-v", voice]
    try:
        subprocess.check_call(cmd + [text], stderr=subprocess.DEVNULL)
        return os.path.exists(out_wav)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
        ch = w.getnchannels()
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sw == 1:
        x = (x - 128.0) / 128.0
    else:
        x = x / float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x * 32767.0).astype(np.int16).tobytes())


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """Mix `noise` into `x` at a target SNR, measured on each signal's own RMS."""
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    off = int(rng.integers(0, max(1, len(noise) - len(x))))
    n = noise[off:off + len(x)].copy()
    sx, sn = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(n ** 2))
    if sn <= 0 or sx <= 0:
        return x
    n *= (sx / sn) * (10.0 ** (-snr_db / 20.0))
    return x + n


def make_babble(clean: List[np.ndarray], rng: np.random.Generator,
                n_talkers: int = 6, seconds: float = 30.0) -> np.ndarray:
    """Speech-shaped babble built from the corpus itself.

    Far more adversarial for an ASR model than white noise, and free -- white
    noise mostly measures the front end, whereas babble measures the thing that
    actually degrades a kitchen conversation.
    """
    out = np.zeros(int(SR * seconds))
    for _ in range(n_talkers):
        pos = 0
        while pos < len(out):
            u = clean[int(rng.integers(0, len(clean)))]
            end = min(len(out), pos + len(u))
            out[pos:end] += u[:end - pos]
            pos = end + int(rng.integers(0, SR // 2))
    return out / (np.max(np.abs(out)) + 1e-9)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def wer(ref: str, hyp: str) -> float:
    """Word error rate by Levenshtein distance over word tokens."""
    import re
    r = re.findall(r"[a-z']+", ref.lower())
    h = re.findall(r"[a-z']+", hyp.lower())
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / len(r)


def recovers_referent(hyp: str, spans: List[str], true_idx: int) -> bool:
    """Does the transcript still point at the right candidate?

    Scored the way the robot would have to: match the transcript against every
    candidate's distinguishing span and take the best. This is the metric the
    env actually cares about, and it is not a monotone function of WER.
    """
    import re
    stop = set("a an the of in on at by with from and or to is was that it one".split())

    def toks(s: str) -> set:
        return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in stop}

    h = toks(hyp)
    if not h:
        return False
    scores = []
    for sp in spans:
        t = toks(sp)
        scores.append(len(h & t) / max(1, len(t)))
    best = int(np.argmax(scores))
    return best == true_idx and scores[best] > 0


# ---------------------------------------------------------------------------
def cmd_synth(args) -> int:
    utts = utterances_from_scenarios(args.scenarios, args.n, args.seed)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    engine = args.tts
    if engine == "auto":
        engine = "sapi" if os.name == "nt" else "say"
    voices: List[Optional[str]] = [None]
    if engine == "sapi":
        v = sapi_voices()
        print(f"SAPI voices: {v or '(none found)'}")
        voices = v[:args.max_voices] or [None]
    elif engine == "say":
        voices = ["Daniel", "Samantha", "Karen"][:args.max_voices]

    clean_paths, clean_sig, meta = [], [], []
    for i, u in enumerate(utts):
        voice = voices[i % len(voices)]
        p = os.path.join(AUDIO_DIR, f"clean_{i:04d}.wav")
        ok = (sapi_synth(u["text"], p, voice) if engine == "sapi"
              else say_synth(u["text"], p, voice))
        if not ok:
            print(f"  TTS failed for {u['text']!r} (voice={voice})")
            continue
        x, sr = read_wav(p)
        x = resample(x, sr, SR)
        write_wav(p, x)
        clean_paths.append(p); clean_sig.append(x)
        meta.append({**u, "clean": p, "voice": voice, "index": i})
        if (i + 1) % 25 == 0:
            print(f"  synthesised {i + 1}/{len(utts)}", flush=True)

    if not clean_sig:
        print("no audio produced; check the TTS engine", file=sys.stderr)
        return 1

    babble = make_babble(clean_sig, rng)
    write_wav(os.path.join(AUDIO_DIR, "babble.wav"), babble)

    rows = []
    for m, x in zip(meta, clean_sig):
        for snr in args.snrs:
            p = os.path.join(AUDIO_DIR, f"n{snr:02d}_{m['index']:04d}.wav")
            write_wav(p, add_noise(x, babble, snr, rng))
            rows.append({**m, "snr_db": snr, "path": p})
        rows.append({**m, "snr_db": None, "path": m["clean"]})  # clean control

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"engine": engine, "voices": voices, "sr": SR,
                   "noise": "babble built from the corpus itself",
                   "snrs": list(args.snrs), "n_utterances": len(meta),
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {len(rows)} clips -> {AUDIO_DIR}")
    print(f"wrote manifest -> {MANIFEST}")
    return 0


def _load_whisper(args):
    """Load the model, falling back to CPU if the CUDA runtime is unusable.

    CTranslate2 needs `cublas64_12.dll` just as llama.cpp does, and on Windows
    it will not find the pip-installed copy on its own -- the same failure,
    reported as "Library cublas64_12.dll is not found or cannot be loaded".
    `_ensure_cuda_dlls()` is the fix that already works for the local tiers, so
    it is reused here rather than reimplemented.

    If CUDA still will not come up, fall back to CPU rather than abandoning the
    measurement: WER is a property of the model and the audio, not of the
    device, so the number is identical either way and only the wall clock
    changes. The device actually used is recorded in the manifest.
    """
    from faster_whisper import WhisperModel

    from cogniwatt.dialogue.backends import _ensure_cuda_dlls

    _ensure_cuda_dlls()
    want = args.device
    if want == "auto":
        want = "cuda"
    try:
        m = WhisperModel(args.whisper, device=want,
                         compute_type=args.compute_type)
        return m, want, args.compute_type
    except Exception as e:
        if want == "cpu":
            raise
        print(f"  CUDA unavailable ({type(e).__name__}: {str(e)[:90]}); "
              f"falling back to CPU -- WER is unaffected, only speed is")
        return WhisperModel(args.whisper, device="cpu",
                            compute_type="int8"), "cpu", "int8"


def cmd_transcribe(args) -> int:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("faster-whisper not installed:\n  pip install faster-whisper",
              file=sys.stderr)
        return 1
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)

    print(f"loading faster-whisper '{args.whisper}' on {args.device}...")
    model, used_device, used_compute = _load_whisper(args)
    args.device, args.compute_type = used_device, used_compute
    print(f"  running on {used_device} ({used_compute})")
    out = []
    for i, r in enumerate(man["rows"], 1):
        segs, _ = model.transcribe(r["path"], language="en", beam_size=1)
        hyp = " ".join(s.text for s in segs).strip()
        out.append({**{k: r[k] for k in
                       ("scenario_id", "k", "true_idx", "text", "snr_db",
                        "voice", "all_spans")},
                    "hyp": hyp,
                    "wer": wer(r["text"], hyp),
                    "recovered": recovers_referent(hyp, r["all_spans"], r["true_idx"])})
        if i % 50 == 0:
            print(f"  {i}/{len(man['rows'])}", flush=True)

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"whisper": args.whisper, "device": args.device,
                   "engine": man["engine"], "voices": man["voices"],
                   "rows": out}, f, indent=2)
    print(f"wrote {len(out)} transcripts -> {RESULTS}")
    return 0


def cmd_report(args) -> int:
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    rows = res["rows"]
    snrs = sorted({r["snr_db"] for r in rows if r["snr_db"] is not None}, reverse=True)

    print(f"whisper '{res['whisper']}'   TTS {res['engine']}   "
          f"voices {res['voices']}\n")
    print(f"{'SNR':>6}{'WER':>9}{'referent recovery':>20}{'n':>6}")
    curve = {}
    clean = [r for r in rows if r["snr_db"] is None]
    if clean:
        w = float(np.mean([r["wer"] for r in clean]))
        rec = float(np.mean([r["recovered"] for r in clean]))
        curve["clean"] = {"wer": w, "recovery": rec, "n": len(clean)}
        print(f"{'clean':>6}{w:>9.3f}{rec:>20.3f}{len(clean):>6}")
    for s in snrs:
        sub = [r for r in rows if r["snr_db"] == s]
        w = float(np.mean([r["wer"] for r in sub]))
        rec = float(np.mean([r["recovered"] for r in sub]))
        curve[str(s)] = {"wer": w, "recovery": rec, "n": len(sub)}
        print(f"{s:>6}{w:>9.3f}{rec:>20.3f}{len(sub):>6}")

    op = str(args.operating_snr)
    if op in curve:
        print(f"\nOPERATING POINT  {args.operating_snr} dB "
              f"(a typical living room)")
        print(f"  DEFAULT_ASR_WER = {curve[op]['wer']:.4f}")
        print(f"  referent recovery = {curve[op]['recovery']:.4f}")
        print(f"\n  Freeze with: python scripts/r2_freeze.py "
              f"--asr-wer {curve[op]['wer']:.4f}")
        # WER and referent loss are not the same quantity; say so with numbers.
        gap = (1 - curve[op]["recovery"]) - curve[op]["wer"]
        print(f"  referent-loss minus WER = {gap:+.4f} -- "
              + ("referent loss EXCEEDS WER, so quoting WER alone understates "
                 "the damage to the ask channel"
                 if gap > 0.02 else
                 "the two track closely here"))

    out = {"curve": curve, "operating_snr": args.operating_snr,
           "whisper": res["whisper"], "tts": res["engine"]}
    with open(os.path.join("data", "r2_asr_curve.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote data/r2_asr_curve.json")

    if args.figures and snrs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(snrs, [curve[str(s)]["wer"] for s in snrs], "o-", label="WER")
        ax.plot(snrs, [1 - curve[str(s)]["recovery"] for s in snrs], "s-",
                label="referent loss (1 - recovery)")
        ax.axvline(args.operating_snr, ls=":", c="k", lw=1)
        ax.annotate(f"operating point\n{args.operating_snr} dB",
                    (args.operating_snr, 0.5), fontsize=8, ha="left")
        ax.invert_xaxis()
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("error rate")
        ax.set_title("ASR degradation of the `ask` channel")
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(FIG_DIR, "r2_fig4_wer_vs_snr.png")
        fig.savefig(p, dpi=160); plt.close(fig)
        print(f"  figure -> {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=os.path.join("data", "r2_scenarios.jsonl"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snrs", type=int, nargs="+", default=list(SNRS))
    ap.add_argument("--tts", default="auto", choices=("auto", "sapi", "say", "piper"))
    ap.add_argument("--max-voices", type=int, default=3)
    ap.add_argument("--whisper", default="small",
                    help="'small' or 'base' -- the size that would plausibly run "
                         "on an edge device; recorded in the report")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--operating-snr", type=int, default=OPERATING_SNR)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    if args.synth:
        return cmd_synth(args)
    if args.transcribe:
        return cmd_transcribe(args)
    if args.report:
        return cmd_report(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
