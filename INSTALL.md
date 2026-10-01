# Installation guide

Setup for the IBus Speech-To-Text engine on Ubuntu/GNOME (X11 or Wayland).
Text is inserted through IBus itself (`commit_text`), so no clipboard, `ydotool`
or `xdotool` is involved and the display server doesn't matter.

Tested on Ubuntu 24.04 with a recent high-end AMD GPU (ROCm 6.4) and with an
NVIDIA laptop GPU (CUDA). CPU-only works too, just slower.

## Prerequisites

- Ubuntu 22.04+ with GNOME
- PipeWire (default on Ubuntu 22.10+) or PulseAudio
- Python 3.12+
- A microphone
- Optional GPU: AMD with ROCm 6.x at `/opt/rocm`, or NVIDIA with a working driver

## 1. System packages

```bash
sudo apt install \
    ibus python3-gi python3-venv \
    gir1.2-ibus-1.0 gir1.2-gstreamer-1.0 gir1.2-adw-1 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-pulseaudio \
    meson ninja-build gettext libglib2.0-dev-bin libibus-1.0-dev libadwaita-1-dev
```

## 2. Get the source

```bash
git clone -b sst-local https://github.com/robwijnhoven/IBus-Speech-To-Text.git
cd IBus-Speech-To-Text
```

`sst-local` carries the Parakeet/Ultra backends, Silero VAD, the live preview
and the latency tuning described in the README.

## 3. Build and install the engine

```bash
meson setup builddir --prefix=/usr
sudo meson install -C builddir
```

This installs the engine to `/usr/share/ibus-stt/`, the IBus component, the
GSettings schema and a launcher at `/usr/libexec/ibus-engine-stt`. Step 5
replaces that launcher with one that uses the engine's own venv.

## 4. Silero VAD model

Silero VAD separates speech from background noise (typing, fans, traffic) far
better than the energy-based fallback.

```bash
sudo install -d -o "$USER" /opt/ibus-stt/models
python3 -c "import urllib.request; urllib.request.urlretrieve(
    'https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx',
    '/opt/ibus-stt/models/silero_vad.onnx')"
```

Use the **v4** model (`v4.0` tag). The v5 model on `master` breaks with
onnxruntime 1.26+ (LSTM state shape errors). The engine looks in
`/opt/ibus-stt/models/` first, then `/usr/share/ibus-stt/models/`, or wherever
`STT_SILERO_VAD_MODEL` points. Without a model it falls back to energy-based VAD.

## 5. Backend, dependencies, model and launcher

```bash
bash install.sh
```

It asks two questions, then does the rest:

- **Backend.** *Ultra* (recommended) is moondream's post-trained Parakeet TDT
  0.6B: 25 languages, lower error rate than plain Parakeet, especially with
  background noise. *Parakeet* is the original v3 model. *Whisper* needs the
  extra build in [Optional: Whisper backend](#optional-whisper-backend).
- **CPU or GPU.** For GPU it detects the vendor and installs the matching
  onnxruntime build (ROCm from AMD's wheel index, or CUDA). With Ultra on NVIDIA
  it also installs moondream's Photon runtime (bf16, least VRAM). Photon is
  CUDA-only, so on AMD and CPU Ultra runs a pinned ONNX export of the same
  weights instead.

Then it creates a venv in `/opt/ibus-stt/venv`, downloads the model, writes the
launcher and a writable `/var/log/ibus-stt.log`, and selects the backend in
GSettings. Runtime files live in `/opt` so moving or deleting the source checkout
can't break a running engine. Set `VENV=` or `MODELS_DIR=` to put them elsewhere.

**AMD GPU selection.** The script lists your ROCm GPUs and asks which index to
use. That number goes into the launcher as `ROCR_VISIBLE_DEVICES`. It must be
the **index** from that list, not the GUID that `rocm-smi` shows: a GUID matches
no device, ROCm then sees no GPU at all, and the engine silently runs on CPU.
Pick a discrete card. Integrated Radeon GPUs such as `gfx90c` have no ROCm 6.4
kernels and crash on the first decode. Don't use `HSA_OVERRIDE_GFX_VERSION`: it
is a global override that disables every GPU of a different architecture.

Re-run `install.sh` after any later `meson install`, which puts back its own
launcher.

## 6. Select the input method

Add **Speech To Text** under *Settings > Keyboard > Input Sources* and switch to
it (usually Super+Space), or run:

```bash
ibus engine stt
```

## Verify it works

```bash
tail -f /var/log/ibus-stt.log
```

At startup you should see the VAD and the model come up:

```
INFO: Silero VAD loaded from /opt/ibus-stt/models/silero_vad.onnx
INFO: STTVad ready -- backend=silero | chunk=512 samples | threshold=0.50 | silence=300ms | ...
INFO: Parakeet model ready (ROCm/fp32)
```

`ROCm/...` or `CUDA/...` means the GPU is in use. `CPU/...` means it fell back.
While you speak there is one line per decode:

```
DEBUG: VAD: speech onset (confidence=0.843)
INFO: decode: 1.54s audio -> 50 ms (RTF=0.032) [ROCm/fp32]
```

The log never contains what you dictated, only lengths such as `<105 chars>`.
To see the text while debugging, add `export STT_LOG_TEXT=1` to the launcher.

## Live preview

While you speak, the text so far appears underlined in the input field (IBus
preedit) and keeps correcting itself. When you pause, the final text replaces it.
Only the final decode is committed, so early wrong guesses never reach the
document.

Some apps don't render IBus preedit (several Electron and Flatpak apps). They
still receive the final text; you just don't see the preview. To turn it off:

```bash
gsettings set org.freedesktop.ibus.engine.stt preedit-text false
```

## Optional: Whisper backend

Whisper (whisper.cpp) needs two extra builds. Clone them next to the engine:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
git clone --recursive https://github.com/absadiki/pywhispercpp.git
```

Build whisper.cpp for CPU:

```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

or for an AMD GPU, setting `AMDGPU_TARGETS` to your card's architecture (see
`rocminfo | grep gfx`; RDNA3 cards are `gfx1100`):

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
    -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Then build the bindings into the engine's venv:

```bash
cd ../pywhispercpp
/opt/ibus-stt/venv/bin/python -m pip install -e . --no-build-isolation
```

Pick Whisper in `install.sh`; it downloads a GGML model and adds the
`LD_LIBRARY_PATH` the launcher needs. (`_pywhispercpp.so` has its build
directory baked in as RUNPATH, so without it `libwhisper.so.1` is not found and
the engine reports "no valid model".) Whisper can invent text on background
noise; if that happens, raise `MIN_SEGMENT_PROB` in `sttgstwhisper.py` or add
the recurring phrase to `_HALLUCINATIONS`.

## Troubleshooting

### No text appears

1. Check `ibus engine` prints `stt`.
2. Check the log for `speech onset` lines when you speak. If there are none, the
   VAD isn't hearing you: see audio capture below.
3. If there are decode lines but nothing is typed, check that the app has focus
   and that a held modifier key isn't deferring the commit.

### The GPU isn't used

A `ROCm pre-flight found no usable GPU device` warning means the ROCm runtime
sees no GPU. Check `rocminfo` lists your card, and that `ROCR_VISIBLE_DEVICES`
in `/usr/libexec/ibus-engine-stt` is an index from that list. The engine runs on
CPU rather than crash, so dictation still works, just slower.

### Audio capture

```bash
# Restart PipeWire
systemctl --user restart pipewire pipewire-pulse wireplumber

# Record 1 second; the file must not be empty (~32 KB)
gst-launch-1.0 pulsesrc num-buffers=100 ! \
    audio/x-raw,format=S16LE,rate=16000,channels=1 ! \
    filesink location=/tmp/test.raw

# List audio sources
pactl list sources short
```

A USB microphone can come back from a USB switch or a suspend enumerated but
delivering no audio. Replugging it fixes that; restart the engine afterwards.
