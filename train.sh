#!/usr/bin/env bash
# Launch Piper fine-tuning on Linux (NVIDIA GPU + nvidia-container-toolkit).
#
#   ./train.sh myvoice            # to epoch 8000
#   MAX_EPOCHS=9000 ./train.sh myvoice
#
# Resumes from the newest non-cleaned checkpoint in
# $TRAINING_DIR/checkpoints and restarts on crash (capped). The Windows
# watchdog's shared-memory spill guard isn't needed here: WDDM paging
# is a Windows/WSL2 behaviour — on native Linux an oversubscribed batch
# fails fast with a real CUDA OOM. If you see one, halve BATCH_SIZE.
set -euo pipefail

VOICE_NAME="${1:?usage: ./train.sh <voice-name>}"
TRAINING_DIR="${TRAINING_DIR:-$HOME/.easy-voice-trainer}"
IMAGE="${IMAGE:-piper-train}"
CONTAINER="${CONTAINER:-piper-train-run}"
MAX_EPOCHS="${MAX_EPOCHS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"

latest_ckpt() {
    ls -t "$TRAINING_DIR/checkpoints/"*.ckpt 2>/dev/null | grep -v '\.cleaned\.ckpt$' | head -1
}

restarts=0
while true; do
    ckpt=$(latest_ckpt)
    [ -n "$ckpt" ] || { echo "no resume checkpoint in $TRAINING_DIR/checkpoints"; exit 1; }
    echo "launching: batch=$BATCH_SIZE, resume from $(basename "$ckpt"), max_epochs=$MAX_EPOCHS"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run --name "$CONTAINER" --gpus all --shm-size=8g \
      -e PYTHONUNBUFFERED=1 \
      -v "$TRAINING_DIR:/training" \
      "$IMAGE" fit \
        --data.voice_name "$VOICE_NAME" \
        --data.csv_path /training/piper_dataset/metadata.csv \
        --data.audio_dir /training/piper_dataset/wavs \
        --model.sample_rate 22050 \
        --data.espeak_voice en-us \
        --data.cache_dir /training/piper_cache \
        --data.config_path "/training/$VOICE_NAME.onnx.json" \
        --data.batch_size "$BATCH_SIZE" \
        --data.num_workers 2 \
        --trainer.accelerator gpu \
        --trainer.devices 1 \
        --trainer.precision 32 \
        --trainer.max_epochs "$MAX_EPOCHS" \
        --trainer.log_every_n_steps 10 \
        --trainer.callbacks+="lightning.pytorch.callbacks.ModelCheckpoint" \
        --trainer.callbacks.every_n_epochs=5 \
        --trainer.callbacks.save_top_k=-1 \
        --trainer.callbacks.dirpath="/training/checkpoints" \
        --ckpt_path "/training/checkpoints/$(basename "$ckpt")" \
      && { echo "training completed"; exit 0; }

    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        echo "restart cap ($MAX_RESTARTS) hit — giving up"
        exit 1
    fi
    if docker logs --tail 50 "$CONTAINER" 2>&1 | grep -qE "CUDA out of memory|OutOfMemoryError"; then
        if [ "$BATCH_SIZE" -gt 1 ]; then
            BATCH_SIZE=$((BATCH_SIZE / 2))
            echo "CUDA OOM — halving batch to $BATCH_SIZE"
        fi
    fi
    echo "container exited abnormally — relaunching ($restarts/$MAX_RESTARTS)"
    sleep 5
done
