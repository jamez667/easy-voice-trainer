"""Convert filtered raw audio into the LJSpeech format Piper trains on.

Piper expects:
  dataset_dir/
    wavs/
      utt_001.wav     # mono, 22050 Hz, 16-bit PCM
      utt_002.wav
      ...
    metadata.csv      # pipe-separated: ID|text|text  (yes, text repeated)

We resample to 22.05 kHz, normalize loudness
to a consistent peak (~-3 dBFS), trim leading/trailing silence, and
write the metadata file. The resampling matches Piper's default
`--sample-rate 22050` so we don't have to retrain a vocoder.
"""
from __future__ import annotations
import csv
import sys
import wave
from pathlib import Path

import numpy as np

RAW = FILTERED_MANIFEST = OUT = WAVS_OUT = None  # set from CLI args in main()
METADATA = OUT / "metadata.csv"

TARGET_SR = 22050
TARGET_PEAK = 0.7079  # ~-3 dBFS, leaves headroom while keeping volume up


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resample. Good enough for TTS training data — Piper's
    model isn't sensitive to small high-freq artifacts from linear interp,
    and we want the script to not need scipy/librosa as deps."""
    if src_sr == dst_sr:
        return audio
    ratio = dst_sr / src_sr
    n_out = int(round(len(audio) * ratio))
    src_idx = np.linspace(0, len(audio) - 1, n_out)
    i0 = src_idx.astype(np.int64)
    i1 = np.clip(i0 + 1, 0, len(audio) - 1)
    frac = (src_idx - i0).astype(np.float32)
    return audio[i0] * (1 - frac) + audio[i1] * frac


def trim_silence(audio: np.ndarray, sr: int, thresh: float = 0.01, pad_ms: int = 80) -> np.ndarray:
    """Trim near-silence from start/end. `pad_ms` of natural quiet is kept
    on each side — abrupt cuts at speech onset make the model learn
    truncated phonemes."""
    abs_a = np.abs(audio)
    mask = abs_a > thresh
    if not mask.any():
        return audio
    start = max(0, mask.argmax() - int(sr * pad_ms / 1000))
    end = min(len(audio), len(audio) - mask[::-1].argmax() + int(sr * pad_ms / 1000))
    return audio[start:end]


def normalize_peak(audio: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return audio * (target_peak / peak)


def main():
    global RAW, FILTERED_MANIFEST, OUT, WAVS_OUT, METADATA
    import argparse
    ap = argparse.ArgumentParser(description="Convert filtered clips into Piper's LJSpeech layout.")
    ap.add_argument("--raw", type=Path, default=Path("raw_audio"))
    ap.add_argument("--manifest", type=Path, default=Path("filtered/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("piper_dataset"))
    a = ap.parse_args()
    RAW, FILTERED_MANIFEST, OUT = a.raw, a.manifest, a.out
    WAVS_OUT = OUT / "wavs"
    OUT.mkdir(parents=True, exist_ok=True)
    WAVS_OUT.mkdir(parents=True, exist_ok=True)
    METADATA = OUT / "metadata.csv"
    if not FILTERED_MANIFEST.exists():
        sys.exit(f"filtered manifest not found at {FILTERED_MANIFEST}. Run filter_dataset.py first.")
    rows = list(csv.DictReader(FILTERED_MANIFEST.open(encoding="utf-8")))
    print(f"converting {len(rows)} clips -> {OUT}")

    metadata_rows: list[tuple[str, str]] = []
    total_dur = 0.0
    fails = 0

    for i, row in enumerate(rows):
        src = RAW / row["filename"]
        utt_id = f"utt_{i:05d}"
        dst = WAVS_OUT / f"{utt_id}.wav"
        text = row["text"].strip()
        try:
            audio, sr = read_wav(src)
            audio = resample_linear(audio, sr, TARGET_SR)
            audio = trim_silence(audio, TARGET_SR)
            audio = normalize_peak(audio)
            write_wav(dst, audio, TARGET_SR)
            total_dur += len(audio) / TARGET_SR
            # Piper's metadata.csv format is: id|raw|normalized
            # We supply the same text for both — Piper normalizes during training.
            # piper1-gpl's default format: utt_filename|text  (two cols)
            metadata_rows.append((f"{utt_id}.wav", text))
        except Exception as e:
            print(f"  [{utt_id}] FAIL: {e}")
            fails += 1
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}")

    # Pipe-separated, NO header — Piper's preprocess expects raw rows.
    with METADATA.open("w", encoding="utf-8", newline="") as f:
        for row in metadata_rows:
            f.write("|".join(row) + "\n")
    print(f"\nwrote {len(metadata_rows)} utterances ({total_dur/60:.1f} min audio) to {OUT}")
    print(f"metadata: {METADATA}")
    print(f"wavs:     {WAVS_OUT}")
    if fails:
        print(f"  ({fails} failed)")


if __name__ == "__main__":
    main()
