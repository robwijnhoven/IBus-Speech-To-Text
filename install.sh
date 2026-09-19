#!/usr/bin/env bash
#
# ibus-stt one-shot setup: pick a backend + CPU/GPU, install the matching Python
# deps (incl. CUDA wheels for GPU), download the model, and select the backend.
#
# Run it from anywhere:  ./install.sh   (or:  bash install.sh)
#
# venv + models default to inside this repo dir; override via env:
#   VENV=/opt/ibus-stt/venv MODELS_DIR=/opt/ibus-stt/models ./install.sh
#
# NB: this installs deps + model. Building/installing the engine itself (meson,
# whisper.cpp, pywhispercpp) is separate -- see INSTALL.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# venv + models default to inside this repo dir (self-contained). Both are
# untracked/gitignored -- only the engine source under IBus-Speech-To-Text/ is
# versioned. Override with VENV= / MODELS_DIR= to place them elsewhere.
VENV="${VENV:-$REPO_DIR/venv}"
MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"
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
# ponytail: call pip via `python -m pip`, not the bin/pip wrapper. Console-script
# shebangs hardcode the venv's original abspath, so a moved project (or a venv
# built under a different path) leaves bin/pip pointing at a dead python ->
# "required file not found". The python symlink survives; -m pip rides it.
PIP=("$PY" -m pip)

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
echo "  2) GPU  - ~5-6x faster decode. Auto-detects vendor: AMD/ROCm (onnxruntime-rocm)"
echo "            or NVIDIA/CUDA (onnxruntime-gpu). Needs the matching driver stack."
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
    "${PIP[@]}" install --upgrade onnx-asr huggingface_hub
    if [[ "$DEVICE" == gpu ]]; then
        "${PIP[@]}" uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm || true
        # Vendor auto-detect: an AMD GPU + an /opt/rocm install -> ROCm; else CUDA.
        ROCM_DIR="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1)"
        if lspci 2>/dev/null | grep -iE 'VGA|3D|Display' | grep -iqE 'AMD|ATI' \
           && [[ -n "$ROCM_DIR" ]]; then
            ROCM_VER="$(basename "$ROCM_DIR" | sed 's/rocm-//;s/\.[0-9]*$//')"  # 6.4.0 -> 6.4
            echo "Detected AMD GPU + ROCm $ROCM_VER -> installing onnxruntime-rocm."
            # Use AMD's ROCm-matched wheel index (repo.radeon.com), NOT pypi: the
            # pypi onnxruntime-rocm links libhipblas.so.3 (ROCm 6.5+), but a 6.4
            # box has .so.2 -> the ROCm EP fails to load and it silently drops to
            # CPU. The rocm-rel-<ver> wheel links the .so majors that ver ships.
            "${PIP[@]}" install --upgrade onnxruntime-rocm \
                --index-url "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VER}/" \
                --extra-index-url "https://pypi.org/simple/" \
              || echo "WARN: onnxruntime-rocm wheel not found for ROCm $ROCM_VER -- GPU will fall back to CPU."
            # The ROCm EP also dlopens MIOpen + hipFFT, which the base ROCm
            # install often omits. Install them (needs sudo); non-fatal if it
            # can't -- the EP just won't load and Parakeet runs on CPU.
            if ! ldconfig -p | grep -q 'libMIOpen.so.1'; then
                echo "Installing ROCm EP runtime libs (miopen-hip hipfft) -- needs sudo ..."
                sudo apt-get install -y miopen-hip hipfft \
                  || echo "WARN: could not install miopen-hip/hipfft -- run it manually, else GPU falls back to CPU."
            fi
            echo "The launcher must export HSA_OVERRIDE_GFX_VERSION (gfx1100 -> 11.0.0)"
            echo "and LD_LIBRARY_PATH=/opt/rocm/lib for the ROCm EP to bind (see INSTALL.md)."
        else
            echo "Installing onnxruntime-gpu (NVIDIA CUDA) ..."
            "${PIP[@]}" install --upgrade onnxruntime-gpu
            # onnxruntime-gpu needs cuDNN 9 + CUDA wheels matching ITS CUDA major.
            # CUDA 13 dropped the -cu13 suffix for the core libs: the CUDA-13
            # wheels are the UNSUFFIXED nvidia-cublas / nvidia-cuda-runtime (the
            # -cu13 ones are deprecated stubs that fail to build); only cuDNN
            # keeps its suffix (nvidia-cudnn-cu13), pulling cublas+nvrtc as deps.
            # CUDA-12 machines still use the -cu12 names. download_model.py and
            # the backend put these wheel dirs on LD_LIBRARY_PATH so they load.
            "${PIP[@]}" install --upgrade nvidia-cudnn-cu13 nvidia-cuda-runtime \
                || "${PIP[@]}" install --upgrade nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
                || echo "WARN: CUDA/cuDNN wheels not installed -- GPU will fall back to CPU."
        fi
        echo "After 'ibus restart', check the log: providers should list the GPU EP"
        echo "(ROCMExecutionProvider / CUDAExecutionProvider). CPU-only = EP didn't load."
    else
        "${PIP[@]}" install --upgrade onnxruntime
    fi
    echo "Downloading Parakeet model (~2.4 GB, cached in ~/.cache/huggingface) ..."
    "$PY" "$REPO_DIR/download_model.py" --parakeet
else
    echo "Downloading Whisper model into $MODELS_DIR ..."
    "${PIP[@]}" install --upgrade huggingface_hub
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
    echo "If the engine log shows a CPU fallback, the GPU EP's libs aren't on the"
    echo "library path -- ROCm: LD_LIBRARY_PATH=/opt/rocm/lib + MIOpen/hipFFT +"
    echo "HSA_OVERRIDE_GFX_VERSION; CUDA: the venv's nvidia/*/lib. Still runs on CPU."
fi
