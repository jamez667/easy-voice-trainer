#!/usr/bin/env bash
# Export a Lightning checkpoint to ONNX (Linux). Latest checkpoint by
# default, or pass a filename from $TRAINING_DIR/checkpoints.
#
#   ./export.sh myvoice
#   ./export.sh myvoice "epoch=8000-step=X.ckpt"
set -euo pipefail

VOICE_NAME="${1:?usage: ./export.sh <voice-name> [ckpt-filename]}"
TRAINING_DIR="${TRAINING_DIR:-$HOME/.easy-voice-trainer}"
IMAGE="${IMAGE:-piper-train}"

if [ -n "${2:-}" ]; then
    ckpt="$TRAINING_DIR/checkpoints/$2"
else
    ckpt=$(ls -t "$TRAINING_DIR/checkpoints/"*.ckpt 2>/dev/null | grep -v '\.cleaned\.ckpt$' | head -1)
fi
[ -f "$ckpt" ] || { echo "no checkpoint found"; exit 1; }
name=$(basename "$ckpt" .ckpt)
mkdir -p "$TRAINING_DIR/exports"

docker run --rm -v "$TRAINING_DIR:/training" --entrypoint python "$IMAGE" \
  -m piper.train.export_onnx \
  --checkpoint "/training/checkpoints/$(basename "$ckpt")" \
  --output-file "/training/exports/$name.onnx"

cfg="$TRAINING_DIR/$VOICE_NAME.onnx.json"
if [ -f "$cfg" ]; then
    cp "$cfg" "$TRAINING_DIR/exports/$name.onnx.json"
else
    echo "warning: no $cfg — piper inference needs the .onnx.json next to the .onnx"
fi
echo "exported: $TRAINING_DIR/exports/$name.onnx"
