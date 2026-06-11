"""Objective A/B evaluation of trained Piper voices.

Synthesizes a fixed corpus of prosody-stressing sentences with each
given ONNX model, then scores every clip:

  - target_sim — resemblyzer speaker-embedding cosine similarity to a
                target voice sample (--target-wav). This is the metric
                that answers "does it sound like the target yet";
                everything else is hygiene. Same-speaker territory is
                roughly >0.80; two takes by the same voice land ~0.95.
  - WER       — faster-whisper round-trip vs the input text
                (intelligibility)
  - DNSMOS    — predicted MOS (ovrl/sig/bak) via speechmos
                (naturalness trend between checkpoints)
  - rate      — speaking rate in chars/sec (drift = pacing regression)
  - clipping  — % of samples at full scale (should be ~0)
  - silence   — fraction of low-energy frames (gaps/stutters show here)

Outputs (under --out, default ./eval):
  <model>/<nn>_<slug>.wav   — every clip, listenable
  results.json              — machine-readable scores
  report.html               — side-by-side A/B page with audio players

Usage (CPU-only, safe while the GPU trains):
  python eval_voice.py exports/epoch=A.onnx exports/epoch=B.onnx \\
      --target-wav target.wav --target-text "What the target says."

Bare names resolve against --exports-dir. Pass --corpus FILE (lines of
"category|sentence") to replace the built-in corpus. Numbers are for
TRENDS between checkpoints of the same voice — don't compare across
different voices/datasets. The final call is always your ears; the
report page is built for exactly that.

Deps: pip install piper-tts faster-whisper librosa speechmos resemblyzer
      "setuptools<81" numpy
"""
from __future__ import annotations
import argparse
import html
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

DEFAULT_CORPUS = [
    ("short ack",    "Got it."),
    ("question",     "Are you sure that's what you want to do?"),
    ("exclamation",  "That's fantastic news, congratulations!"),
    ("numbers",      "The meeting moved from 9:45 to 3:30, and we need 17 more units."),
    ("date",         "It's Thursday the eleventh of June, twenty twenty-six."),
    ("long",         "I was thinking about that thing you mentioned earlier, the one with "
                     "the Friday deadlines, and honestly I reckon we should push it a week."),
    ("names",        "Sarah and Mark are meeting Doctor Aleksandrov in Birmingham."),
    ("clusters",     "The sixth sick sheik's sixth sheep is sick."),
    ("hesitation",   "Hmm, let me think about that for a second."),
    ("empathy",      "Oh, that sounds really hard. Are you okay?"),
    ("technical",    "The container exposes port eight zero eight two for the synthesizer."),
    ("trailing",     "Well, that's one way of looking at it, I suppose."),
]

ARGS: argparse.Namespace
CORPUS: list[tuple[str, str]]

_voice_cache: dict = {}
_whisper = None
_HAS_DNSMOS = True
_spk_encoder = None
_target_embed = None


def synthesize(onnx_path: Path, text: str) -> tuple[np.ndarray, int]:
    voice = _voice_cache.get(onnx_path)
    if voice is None:
        from piper import PiperVoice
        voice = PiperVoice.load(str(onnx_path))
        _voice_cache[onnx_path] = voice
    chunks, sr = [], 22050
    for piece in voice.synthesize(text):
        if hasattr(piece, "audio_int16_array"):
            chunks.append(piece.audio_int16_array)
        elif hasattr(piece, "audio_int16_bytes"):
            chunks.append(np.frombuffer(piece.audio_int16_bytes, dtype=np.int16))
        if hasattr(piece, "sample_rate"):
            sr = piece.sample_rate
    return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16), sr


def target_similarity(audio_i16: np.ndarray, sr: int) -> float | None:
    """Cosine similarity of the clip's speaker embedding vs the target
    sample. Returns None when no --target-wav was given or resemblyzer
    is unavailable (the rest of the suite still runs)."""
    global _spk_encoder, _target_embed
    if not ARGS.target_wav:
        return None
    try:
        if _spk_encoder is None:
            from resemblyzer import VoiceEncoder, preprocess_wav
            _spk_encoder = VoiceEncoder("cpu")
            _target_embed = _spk_encoder.embed_utterance(preprocess_wav(str(ARGS.target_wav)))
        from resemblyzer import preprocess_wav
        f32 = audio_i16.astype(np.float32) / 32768.0
        emb = _spk_encoder.embed_utterance(preprocess_wav(f32, source_sr=sr))
        return round(float(np.dot(emb, _target_embed)), 3)
    except Exception as e:
        print(f"[eval] target similarity unavailable ({e})")
        return None


def transcribe(audio_i16: np.ndarray, sr: int) -> str:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(ARGS.whisper_model, device="cpu", compute_type="int8")
    f32 = audio_i16.astype(np.float32) / 32768.0
    if sr != 16000:
        import librosa
        f32 = librosa.resample(f32, orig_sr=sr, target_sr=16000)
    segments, _ = _whisper.transcribe(f32, language="en", beam_size=5)
    return " ".join(s.text for s in segments).strip()


def norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / len(r)


def mos(audio_i16: np.ndarray, sr: int) -> dict:
    global _HAS_DNSMOS
    if not _HAS_DNSMOS:
        return {}
    try:
        from speechmos import dnsmos
        import librosa
        f32 = audio_i16.astype(np.float64) / 32768.0
        if sr != 16000:
            f32 = librosa.resample(f32, orig_sr=sr, target_sr=16000)
        # Resampling ringing can overshoot ±1 by a hair; dnsmos rejects it.
        f32 = np.clip(f32, -1.0, 1.0)
        r = dnsmos.run(f32, sr=16000)
        return {k: round(float(v), 2) for k, v in r.items() if k.endswith("_mos")}
    except Exception as e:
        print(f"[eval] dnsmos unavailable ({e}); skipping MOS")
        _HAS_DNSMOS = False
        return {}


def clip_metrics(audio_i16: np.ndarray, sr: int, text: str) -> dict:
    f32 = audio_i16.astype(np.float32) / 32768.0
    dur = len(f32) / sr
    frame = max(1, int(sr * 0.03))
    n_frames = len(f32) // frame
    rms = np.sqrt(np.mean(
        f32[: n_frames * frame].reshape(-1, frame) ** 2, axis=1)) if n_frames else np.array([0.0])
    return {
        "duration_s": round(dur, 2),
        "chars_per_s": round(len(text) / dur, 1) if dur > 0 else 0,
        "clipping_pct": round(float(np.mean(np.abs(f32) > 0.999)) * 100, 3),
        "silence_ratio": round(float(np.mean(rms < 0.01)), 2),
        "peak": round(float(np.max(np.abs(f32))), 2),
    }


def write_wav(path: Path, audio_i16: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio_i16.tobytes())


def resolve_model(arg: str) -> Path:
    p = Path(arg)
    if p.suffix == ".onnx" and p.exists():
        return p
    for cand in (ARGS.exports_dir / f"{arg}.onnx", ARGS.exports_dir / arg):
        if cand.exists():
            return cand
    sys.exit(f"can't find model {arg!r} (looked in {ARGS.exports_dir})")


def evaluate(models: list[Path]) -> dict:
    out_dir = ARGS.out
    results: dict = {}
    for m in models:
        name = m.stem
        print(f"\n=== {name} ===")
        rows = []
        for idx, (cat, text) in enumerate(CORPUS):
            audio, sr = synthesize(m, text)
            slug = re.sub(r"[^a-z0-9]+", "-", cat)[:20]
            wav_path = out_dir / name / f"{idx:02d}_{slug}.wav"
            write_wav(wav_path, audio, sr)
            hyp = transcribe(audio, sr)
            sim = target_similarity(audio, sr)
            row = {
                "category": cat,
                "text": text,
                "transcript": hyp,
                "wer": round(wer(text, hyp), 3),
                **({"target_sim": sim} if sim is not None else {}),
                **clip_metrics(audio, sr, text),
                **mos(audio, sr),
                "wav": str(wav_path.relative_to(out_dir)).replace("\\", "/"),
            }
            rows.append(row)
            print(f"  [{cat:>11}] sim={row.get('target_sim', '-')} wer={row['wer']:.2f} "
                  f"mos={row.get('ovrl_mos', '-')} rate={row['chars_per_s']}c/s")
        sims = [r["target_sim"] for r in rows if "target_sim" in r]
        summary = {
            "mean_target_sim": round(float(np.mean(sims)), 3) if sims else None,
            "mean_wer": round(float(np.mean([r["wer"] for r in rows])), 3),
            "mean_ovrl_mos": round(float(np.mean([r["ovrl_mos"] for r in rows])), 2)
            if rows and "ovrl_mos" in rows[0] else None,
            "mean_chars_per_s": round(float(np.mean([r["chars_per_s"] for r in rows])), 1),
            "max_clipping_pct": max(r["clipping_pct"] for r in rows),
        }
        print(f"  summary: {summary}")
        results[name] = {"summary": summary, "clips": rows}
    return results


def write_report(results: dict):
    out_dir = ARGS.out
    names = list(results)
    h = ["<!doctype html><html><head><meta charset='utf-8'><title>Voice eval</title>",
         "<style>body{font-family:sans-serif;background:#15161a;color:#ececec;padding:24px}",
         "table{border-collapse:collapse;width:100%}td,th{border:1px solid #333;padding:8px;",
         "font-size:13px;vertical-align:top}th{background:#22242a}audio{width:230px}",
         ".best{color:#7dd87d;font-weight:bold}</style></head><body>",
         "<h1>Voice checkpoint A/B</h1>",
         "<p>Numbers are trends, not truth — use the players. Lower WER and "
         "higher MOS/similarity are better; rate and silence should stay stable.</p>"]
    if ARGS.target_wav:
        import shutil
        shutil.copy(ARGS.target_wav, out_dir / "target_reference.wav")
        h.append("<h2>Target voice (reference)</h2>"
                 "<audio controls preload='none' src='target_reference.wav'></audio>")
        if ARGS.target_text:
            h.append(f"<p><i>{html.escape(ARGS.target_text)}</i></p>")
    h.append("<table><tr><th>model</th><th>target sim</th><th>mean WER</th><th>mean MOS</th>"
             "<th>rate c/s</th><th>max clip %</th></tr>")
    sims = [results[n]["summary"].get("mean_target_sim") for n in names]
    best_sim = max((s for s in sims if s is not None), default=None)
    for n in names:
        s = results[n]["summary"]
        sim = s.get("mean_target_sim")
        cls = " class='best'" if sim is not None and sim == best_sim else ""
        h.append(f"<tr><td>{html.escape(n)}</td>"
                 f"<td{cls}>{sim if sim is not None else '-'}</td>"
                 f"<td>{s['mean_wer']}</td>"
                 f"<td>{s['mean_ovrl_mos'] if s['mean_ovrl_mos'] is not None else '-'}</td>"
                 f"<td>{s['mean_chars_per_s']}</td><td>{s['max_clipping_pct']}</td></tr>")
    h.append("</table>")
    h.append("<table><tr><th>sentence</th>" +
             "".join(f"<th>{html.escape(n)}</th>" for n in names) + "</tr>")
    for idx, (cat, text) in enumerate(CORPUS):
        h.append(f"<tr><td><b>{html.escape(cat)}</b><br>{html.escape(text)}</td>")
        for n in names:
            r = results[n]["clips"][idx]
            mos_s = f" · mos {r['ovrl_mos']}" if "ovrl_mos" in r else ""
            sim_s = f"sim {r['target_sim']} · " if "target_sim" in r else ""
            h.append(f"<td><audio controls preload='none' src='{html.escape(r['wav'])}'></audio>"
                     f"<br>{sim_s}wer {r['wer']}{mos_s} · {r['chars_per_s']} c/s"
                     f"<br><i>{html.escape(r['transcript'])}</i></td>")
        h.append("</tr>")
    h.append("</table></body></html>")
    (out_dir / "report.html").write_text("\n".join(h), encoding="utf-8")


def main():
    global ARGS, CORPUS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="ONNX paths or bare names in --exports-dir")
    ap.add_argument("--exports-dir", type=Path, default=Path("exports"))
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--target-wav", type=Path, default=None,
                    help="reference sample of the voice being chased (enables target_sim)")
    ap.add_argument("--target-text", default=None,
                    help="exact transcript of --target-wav; added to the corpus as a "
                         "'target' row for a same-text comparison")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="file of 'category|sentence' lines replacing the built-in corpus")
    ap.add_argument("--whisper-model", default="small")
    ARGS = ap.parse_args()

    if ARGS.target_wav and not ARGS.target_wav.exists():
        sys.exit(f"target wav not found: {ARGS.target_wav}")
    CORPUS = list(DEFAULT_CORPUS)
    if ARGS.corpus:
        CORPUS = []
        for line in ARGS.corpus.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and "|" in line:
                cat, text = line.split("|", 1)
                CORPUS.append((cat.strip(), text.strip()))
        if not CORPUS:
            sys.exit("corpus file had no 'category|sentence' lines")
    if ARGS.target_text:
        CORPUS.insert(0, ("target", ARGS.target_text))

    models = [resolve_model(a) for a in ARGS.models]
    ARGS.out.mkdir(parents=True, exist_ok=True)
    results = evaluate(models)
    (ARGS.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(f"\nreport: {ARGS.out / 'report.html'}")
    print(f"json:   {ARGS.out / 'results.json'}")


if __name__ == "__main__":
    main()
