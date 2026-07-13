#!/usr/bin/env bash
#
# ibus-stt one-shot setup: pick a backend + CPU/GPU, install the matching Python
# deps (incl. CUDA wheels for GPU), download the model, and select the backend.
#
# Run it from anywhere:  ./install.sh   (or:  bash install.sh)
#
# Overridable via env: PROJECT_ROOT, VENV, MODELS_DIR.
#   VENV=/opt/ibus-stt/venv ./install.sh
#
# NB: this installs deps + model. Building/installing the engine itself (meson,
# whisper.cpp, pywhispercpp) is separate -- see INSTALL.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$REPO_DIR")}"
VENV="${VENV:-$PROJECT_ROOT/venv}"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
SCHEMA="org.freedesktop.ibus.engine.stt"

echo "ibus-stt setup"
echo "  repo:   $REPO_DIR"
echo "  venv:   $VENV"
echo "  models: $MODELS_DIR"
echo

# --- venv (create if missing; system site-packages for gi/GStreamer) ---------
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "No venv at $VENV -- creating one (--system-site-packages for gi/GStreamer)."
    python3 -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# --- backend choice ----------------------------------------------------------
echo "Which speech-recognition backend?"
echo "  1) Parakeet  [recommended] - fastest (~18x realtime on CPU), does NOT"
echo "               hallucinate on noise, multilingual (EN + Dutch + 23 more)."
echo "               No live word-by-word streaming: text appears when you pause."
echo "  2) Whisper   - very accurate + live streaming pre-edit, BUT hallucinates"
echo "               on background/fan noise (invents text, stray CJK). GPU advised."
read -rp "Backend [1]: " b; b="${b:-1}"
case "$b" in
    1) BACKEND=parakeet ;;
    2) BACKEND=whisper ;;
    *) echo "invalid choice"; exit 1 ;;
esac

# --- compute device ----------------------------------------------------------
echo
echo "Run on CPU or GPU?"
echo "  1) CPU  [safe default] - works everywhere. Parakeet is ~18x realtime on CPU."
echo "  2) GPU  - NVIDIA CUDA; faster decode (mainly helps long utterances)."
echo "            Needs an NVIDIA GPU + CUDA driver; installs onnxruntime-gpu + cuDNN."
read -rp "Device [1]: " d; d="${d:-1}"
case "$d" in
    1) DEVICE=cpu ;;
    2) DEVICE=gpu ;;
    *) echo "invalid choice"; exit 1 ;;
esac

echo
echo ">> backend=$BACKEND  device=$DEVICE"
echo

# --- dependencies + model ----------------------------------------------------
if [[ "$BACKEND" == parakeet ]]; then
    echo "Installing Parakeet deps ..."
    "$PIP" install --upgrade onnx-asr huggingface_hub
    if [[ "$DEVICE" == gpu ]]; then
        echo "Swapping onnxruntime -> onnxruntime-gpu (+ matching CUDA/cuDNN wheels) ..."
        "$PIP" uninstall -y onnxruntime || true
        "$PIP" install --upgrade onnxruntime-gpu
        # onnxruntime-gpu needs cuDNN 9 + CUDA wheels matching ITS CUDA major.
        # CUDA 13 dropped the -cu13 suffix for the core libs: the CUDA-13 wheels
        # are the UNSUFFIXED nvidia-cublas / nvidia-cuda-runtime (the -cu13 ones
        # are deprecated stubs that fail to build); only cuDNN keeps its suffix
        # (nvidia-cudnn-cu13), and it pulls cublas+nvrtc as deps. CUDA-12 machines
        # still use the -cu12 names. The backend calls onnxruntime.preload_dlls()
        # so these load without a manual LD_LIBRARY_PATH.
        "$PIP" install --upgrade nvidia-cudnn-cu13 nvidia-cuda-runtime \
            || "$PIP" install --upgrade nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
            || echo "WARN: CUDA/cuDNN wheels not installed -- GPU will fall back to CPU."
        echo "After 'ibus restart', check the log: providers should list CUDAExecutionProvider."
        echo "If it shows only CPU, cuDNN wasn't found -- Parakeet still runs fine on CPU."
    else
        "$PIP" install --upgrade onnxruntime
    fi
    echo "Downloading Parakeet model (~2.4 GB, cached in ~/.cache/huggingface) ..."
    "$PY" "$REPO_DIR/download_model.py" --parakeet
else
    echo "Downloading Whisper model into $MODELS_DIR ..."
    "$PIP" install --upgrade huggingface_hub
    "$PY" "$REPO_DIR/download_model.py" --whisper --models-dir "$MODELS_DIR"
    if [[ "$DEVICE" == gpu ]]; then
        echo
        echo "NOTE: whisper.cpp GPU is a BUILD flag (-DGGML_CUDA=ON / -DGGML_HIP=ON),"
        echo "      not a pip package. Rebuild pywhispercpp with GPU support (INSTALL.md)."
    fi
fi

# --- select the backend in gsettings (if the engine/schema is installed) -----
echo
if command -v gsettings >/dev/null 2>&1 && gsettings writable "$SCHEMA" backend >/dev/null 2>&1; then
    gsettings set "$SCHEMA" backend "$BACKEND"
    echo "Selected backend '$BACKEND' in $SCHEMA."
else
    echo "NOTE: gschema '$SCHEMA' not found -- install the engine first (INSTALL.md),"
    echo "      then run: gsettings set $SCHEMA backend $BACKEND"
fi

echo
echo "Done. Apply it with:  ibus restart"
if [[ "$DEVICE" == gpu && "$BACKEND" == parakeet ]]; then
    echo "If the engine log shows a CPU fallback, cuDNN isn't on the library path"
    echo "(export LD_LIBRARY_PATH to the venv's nvidia/cudnn/lib) -- it still runs on CPU."
fi
