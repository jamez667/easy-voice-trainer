# easy-voice-trainer

Fine-tune a [Piper](https://github.com/OHF-Voice/piper1-gpl) TTS voice on a
consumer GPU without losing nights of training to silent failure modes —
plus an objective evaluation suite so you know when to *stop* training.

Extracted from a real project that fine-tuned a custom voice on an RTX
3080 (10 GB) and hit every trap on the way. The tooling here encodes the
fixes:

- **Docker-only training.** Training piper natively on Windows is a
  parade of cross-platform breakage (torch 2.6 `weights_only` defaults,
  `PosixPath` pickles from Linux checkpoints, MSVC/cython issues). The
  image sidesteps all of it and was measured ~12× faster than a native
  venv on the same hardware.
- **A watchdog, not a bare `docker run`.** The deadliest failure is
  invisible: oversubscribe VRAM and WDDM silently pages CUDA into system
  RAM — GPU util reads 100%, no errors anywhere, throughput drops ~10×.
  One night of that produced 42 epochs instead of ~500. The watchdog
  detects the spill via GPU perf counters, kills the run, halves the
  batch size, and resumes from the latest checkpoint.
- **An eval suite, not vibes.** Fine-tunes converge much earlier than
  you think (ours: identity converged within ~20K steps; the next 300K
  steps were metric noise). `eval_voice.py` scores checkpoints on
  speaker similarity to a target sample, STT round-trip WER, DNSMOS,
  and pacing — and builds a side-by-side listening page for the final
  call your ears make.

## Quick start

```powershell
# 1. Build the training image (~20 GB, one-time)
docker build -t piper-train .

# 2. Lay out your training dir (default: ~\.easy-piper-training)
#    piper_dataset\wavs\*.wav         mono 22050 Hz 16-bit PCM
#    piper_dataset\metadata.csv       id|text|text  (LJSpeech style)
#    checkpoints\<base-or-resume>.ckpt
#    <voice>.onnx.json                written by piper on first run

# 3. Train — the watchdog launches and babysits the run
powershell -File watchdog.ps1 -VoiceName myvoice -MaxEpochs 8000

# 4. Export a checkpoint to ONNX
powershell -File export_checkpoint.ps1 -VoiceName myvoice

# 5. Evaluate checkpoints against each other (and a target sample)
python eval_voice.py exports\epoch=A.onnx exports\epoch=B.onnx `
    --target-wav target.wav --target-text "What the target sample says."
# -> eval\report.html with audio players + metrics
```

## The shared-memory trap (read this once)

On Windows/WSL2, CUDA allocations that exceed VRAM do not fail — the
driver pages them to system RAM and your training silently runs ~10×
slower while `nvidia-smi` shows a healthy-looking 100% utilization.
The tells:

- VRAM pinned at 100% of the card with **0 free** (healthy training
  should leave ~1.5 GB headroom)
- GPU power draw well below its normal training level
- epochs taking minutes instead of the ~1 min you measured before

The watchdog automates the response (kill → halve batch → resume), and
`nvidia-smi`'s "Prefer No Sysmem Fallback" CUDA policy turns the silent
crawl into a loud OOM if you'd rather fail fast. **Batch size is the
lever** — dataloader workers only affect CPU-side loading.

## Files

| File | What it does |
|---|---|
| `Dockerfile` | piper1-gpl + CUDA training image (pinned, reproducible, includes the torch `weights_only` shim old checkpoints need) |
| `watchdog.ps1` | launches training and auto-recovers from spill/stall/crash; decision logic is a pure function |
| `watchdog.Tests.ps1` | Pester 5 suite for every watchdog rule (no GPU/Docker needed) |
| `export_checkpoint.ps1` | checkpoint → ONNX via the same image |
| `eval_voice.py` | objective A/B: target-speaker similarity (resemblyzer), WER (faster-whisper), DNSMOS, rate/clipping/silence + HTML listening report |
| `filter_dataset.py` | drop glitched clips (silence, cutoffs, abnormal rate) before training |
| `prepare_dataset.py` | resample/normalize/trim raw clips into LJSpeech layout |

## Eval dependencies

```
pip install piper-tts faster-whisper librosa speechmos resemblyzer "setuptools<81" numpy
```

All CPU-only — you can evaluate while the GPU trains.

## Knowing when to stop

Run the eval across your checkpoint history. When speaker similarity to
the target flattens (ours sat at 0.90 ± 0.01 for 300K steps) and WER/MOS
stop moving, further training is burning electricity. The `report.html`
listening page is the final arbiter — metrics can't hear warmth.
