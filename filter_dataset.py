"""Filter the synthesized dataset for training.

TTS-synthesized (or scraped) datasets often contain:
- Silence / very-short clips (model glitched)
- Cutoff clips that end mid-word
- Distorted clips with extreme peaks
- Clips with abnormal speaking rate (too fast / too slow)

This script reads the manifest, evaluates each WAV, and writes a cleaned
manifest at filtered/manifest.csv with paths into raw_audio/. Files
themselves don't move — we just decide which the trainer sees.

Heuristics:
  - Duration: drop < 1.0s or > 12.0s (sentences in our corpus are
    moderate length; outliers mean glitch).
  - Peak amplitude: drop < 0.05 (probably mostly silence) or rare clips
    that are mostly clipping.
  - Chars-per-second: drop < 5 or > 25 (real speech is ~10-18 cps).
    Catches mumbled-too-fast and stretched-too-slow cases.

Prints a summary so you can see how many survived and why.
"""
from __future__ import annotations
import csv
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

RAW = OUT = MANIFEST_IN = MANIFEST_OUT = None  # set from CLI args in main()


def wav_stats(path: Path) -> tuple[float, float, int]:
    """(duration_s, peak_abs, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if arr.size == 0:
        return 0.0, 0.0, sr
    return arr.size / sr, float(np.abs(arr).max()), sr


def main():
    global RAW, OUT, MANIFEST_IN, MANIFEST_OUT
    import argparse
    ap = argparse.ArgumentParser(description="Filter glitched clips out of a synthesized dataset.")
    ap.add_argument("--raw", type=Path, default=Path("raw_audio"),
                    help="dir containing the clips + manifest.csv (columns: file,text)")
    ap.add_argument("--out", type=Path, default=Path("filtered"))
    a = ap.parse_args()
    RAW, OUT = a.raw, a.out
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_IN = RAW / "manifest.csv"
    MANIFEST_OUT = OUT / "manifest.csv"
    if not MANIFEST_IN.exists():
        sys.exit(f"manifest not found at {MANIFEST_IN}. Expected columns: file,text.")
    rows = list(csv.DictReader(MANIFEST_IN.open(encoding="utf-8")))
    print(f"input: {len(rows)} clips")

    reasons: Counter = Counter()
    kept: list[dict] = []
    total_kept_dur = 0.0

    for row in rows:
        wav_path = RAW / row["filename"]
        if not wav_path.exists():
            reasons["missing_file"] += 1
            continue
        try:
            dur, peak, sr = wav_stats(wav_path)
        except Exception as e:
            reasons[f"read_error:{type(e).__name__}"] += 1
            continue

        if dur < 1.0:
            reasons["too_short"] += 1
            continue
        if dur > 12.0:
            reasons["too_long"] += 1
            continue
        if peak < 0.05:
            reasons["mostly_silent"] += 1
            continue

        text = row["text"].strip()
        cps = len(text) / dur if dur > 0 else 0
        if cps < 5.0:
            reasons["too_slow"] += 1
            continue
        if cps > 25.0:
            reasons["too_fast"] += 1
            continue

        kept.append({
            "idx": row["idx"],
            "filename": row["filename"],
            "duration_s": f"{dur:.3f}",
            "peak": f"{peak:.3f}",
            "cps": f"{cps:.2f}",
            "sample_rate": str(sr),
            "text": text,
        })
        total_kept_dur += dur

    # Write the filtered manifest.
    with MANIFEST_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "filename", "duration_s", "peak", "cps", "sample_rate", "text"])
        w.writeheader()
        w.writerows(kept)

    print(f"kept: {len(kept)} clips, {total_kept_dur/60:.1f} min audio")
    print("dropped reasons:")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    print(f"\nfiltered manifest written to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
