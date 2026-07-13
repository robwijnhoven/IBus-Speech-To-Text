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
import glob
import os
import sys


def _ensure_cuda_libpath():
    """Put the nvidia-*-cuNN pip-wheel lib dirs on LD_LIBRARY_PATH, then re-exec.

    onnxruntime-gpu links libcudart.so.NN which lives under the venv's
    site-packages/nvidia/*/lib wheels, NOT on the default loader path. Its own
    preload_dlls() can't rescue this: `import onnxruntime` raises at import time
    (before any call is possible). LD_LIBRARY_PATH is read only at exec, so we
    set it and re-exec ourselves once. No-op on CPU-only installs (no wheels).
    """
    if os.environ.get("_STT_CUDA_LIBPATH_SET"):
        return  # already re-exec'd once; avoid a loop
    sp = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                      "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages")
    libdirs = sorted({os.path.dirname(p)
                      for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib", "*.so*"))})
    if not libdirs:
        return  # CPU-only install, nothing to add
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libdirs + ([ld] if ld else []))
    os.environ["_STT_CUDA_LIBPATH_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libpath()

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
