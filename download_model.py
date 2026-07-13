#!/usr/bin/env python3
# vim:set et sts=4 sw=4:
#
# ibus-stt - pre-download an ASR model before configuring the engine, so the
# first dictation doesn't stall on a multi-GB download.
#
#   python download_model.py --parakeet
#   python download_model.py --whisper                       # default ggml turbo
#   python download_model.py --whisper --whisper-model ggml-base.en.bin
#
# Run it with the engine's venv python so the download lands where the engine
# looks:  venv/bin/python IBus-Speech-To-Text/download_model.py --parakeet

import argparse
import sys

PARAKEET_MODEL        = "nemo-parakeet-tdt-0.6b-v3"   # multilingual, CC-BY-4.0
WHISPER_REPO          = "ggerganov/whisper.cpp"
DEFAULT_WHISPER_MODEL = "ggml-large-v3-turbo-q5_0.bin"


def download_parakeet():
    try:
        import onnx_asr
    except ImportError:
        sys.exit("onnx-asr not installed -> pip install onnx-asr huggingface_hub")
    print(f"Downloading Parakeet '{PARAKEET_MODEL}' (~2.4 GB) to the HF cache ...")
    # Load on CPU just to fetch + verify the files. A download needs no GPU, and
    # forcing CPU avoids noisy TensorRT/CUDA provider-init errors during install
    # (onnxruntime-gpu lists TensorRT first, which most machines don't have).
    onnx_asr.load_model(PARAKEET_MODEL, providers=["CPUExecutionProvider"])
    print("Parakeet model ready.")


def download_whisper(filename, models_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed -> pip install huggingface_hub")
    print(f"Downloading Whisper '{filename}' from {WHISPER_REPO} -> {models_dir} ...")
    path = hf_hub_download(repo_id=WHISPER_REPO, filename=filename,
                           local_dir=models_dir)
    print(f"Whisper model at {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Pre-download an ibus-stt ASR model.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--parakeet", action="store_true",
                       help="Download the Parakeet (onnx-asr) model")
    group.add_argument("--whisper", action="store_true",
                       help="Download a whisper.cpp ggml model")
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                    help=f"ggml filename in {WHISPER_REPO} "
                         f"(default: {DEFAULT_WHISPER_MODEL})")
    ap.add_argument("--models-dir", default="models",
                    help="Destination for whisper ggml files (default: ./models)")
    args = ap.parse_args()

    if args.parakeet:
        download_parakeet()
    else:
        download_whisper(args.whisper_model, args.models_dir)


if __name__ == "__main__":
    main()
