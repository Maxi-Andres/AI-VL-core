#!/usr/bin/env bash
# Train the "Bender" wake word model in the training container (Dockerfile), step by step.
#
#   ./train.sh build       build the image (aivl-wakeword:1)
#   ./train.sh prepare     background audio: room echoes, AudioSet, the G1's own mic (prepare.py)
#   ./train.sh generate    synthetic "Bender"s and near-misses with Piper (openWakeWord train.py)
#   ./train.sh augment     mix them with echo and noise, and compute their features
#   ./train.sh train       train; the model lands in ~/wakeword/models/bender.onnx
#
# Everything big lives outside the repo: ~/wakeword (sources, data, models) is /work in the
# container, and the recordings of the G1's mic (~/.pilot/wakeword) are /recorded, read-only.
# The GPU is shared with the VLM server, which takes ~7 of its 8 GB: stop the app before
# `generate` and `train` (they need the GPU), or they fail for lack of memory.
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: Piper's voice (en_US-libritts_r-medium.pt, rhasspy's own
# GitHub release) is a whole pickled model, which PyTorch >= 2.6 refuses by default.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WAKEWORD_WORK:-$HOME/wakeword}"
RECORDED="${WAKEWORD_RECORDED:-$HOME/.pilot/wakeword}"
IMAGE=aivl-wakeword:1

run() {
    docker run --rm --gpus all --shm-size=8g \
        -u "$(id -u):$(id -g)" -e HOME=/work/home -e HF_HOME=/work/hf \
        -e PYTHONPATH=/work/src/openWakeWord \
        -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        -v "$WORK:/work" -v "$RECORDED:/recorded:ro" -v "$HERE:/tools:ro" \
        -w /work "$IMAGE" "$@"
}

oww() {
    run python /work/src/openWakeWord/openwakeword/train.py --training_config /tools/bender.yml "$@"
}

mkdir -p "$WORK/home" "$WORK/hf" "$WORK/models"
case "${1:-}" in
    build)    docker build -t "$IMAGE" "$HERE" ;;
    prepare)  run python /tools/prepare.py ;;
    generate) oww --generate_clips ;;
    augment)  oww --augment_clips ;;
    train)    oww --train_model ;;
    shell)    run bash ;;
    *) sed -n '2,15p' "$0"; exit 2 ;;
esac
