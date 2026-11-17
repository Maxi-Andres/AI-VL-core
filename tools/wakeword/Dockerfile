# The wake word's training box: openWakeWord's automatic training (synthetic "Bender"s from Piper,
# augmented with room echo and noise, against ~2000 h of negative audio), on the RTX 5060.
# CUDA 12.8 because the 5060 (Blackwell, sm_120) needs it; Python 3.11 because piper-phonemize
# has no wheel for newer ones. Built and run by train.sh; nothing here is installed on the host.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# openWakeWord's training notebook (notebooks/automatic_model_training.ipynb), less TensorFlow:
# it only converts the model to .tflite for phones, and ONNX is what runs here.
RUN pip install --no-cache-dir "numpy<2" "scipy<1.12" \
    piper-phonemize==1.1.0 webrtcvad mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 \
    speechbrain==0.5.16 audiomentations==0.33.0 torch-audiomentations==0.11.0 acoustics==0.2.6 \
    pronouncing==0.2.0 datasets==2.14.6 deep-phonemizer==0.0.19 onnx onnxruntime pyyaml tqdm \
    soundfile scikit-learn requests "pyarrow<15"

WORKDIR /work
