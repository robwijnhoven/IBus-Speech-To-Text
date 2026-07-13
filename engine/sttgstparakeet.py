# vim:set et sts=4 sw=4:
#
# ibus-stt - Speech To Text engine for IBus
# Copyright (C) 2022 Philippe Rouquier <bonfire-app@wanadoo.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Parakeet speech-to-text backend.

A deliberately thin sibling of the Whisper backend. It reuses the same capture
graph (pulsesrc -> webrtcdsp -> appsink @ 16 kHz mono) and the same Silero VAD
segmentation, then decodes each finalised VAD segment with NVIDIA Parakeet-TDT
via onnx-asr on CPU.

Parakeet is a Token-and-Duration Transducer: at every step it may emit a blank,
so on silence/noise it produces an empty string rather than inventing text. That
removes -- at the source -- the failure modes the Whisper backend spends ~400
lines fighting (silence hallucination, repetition loops, streamed+tail seam
duplication). So there is intentionally NO streaming/promotion/tail-decode,
_collapse_repetitions, or _HALLUCINATIONS machinery here: one VAD segment in,
one string out.
"""

import logging
import queue
import threading
import time

import numpy as np

from gi.repository import GLib
from gi.repository import Gio
from gi.repository import Gst

from sttgstbase import STTGstBase

try:
    import onnx_asr
    ONNX_ASR_AVAILABLE = True
except ImportError:
    ONNX_ASR_AVAILABLE = False

try:
    from sttvad import STTVad
    VAD_MODULE_OK = True
except Exception:
    VAD_MODULE_OK = False

LOG_MSG = logging.getLogger()

SAMPLE_RATE = 16000
# Parakeet-TDT 0.6B v3: multilingual (EN + Dutch + 23 more European langs),
# CC-BY-4.0. onnx-asr downloads it to the HF cache (~/.cache/huggingface) on
# first load. ponytail: model id hard-coded -- there is only one, no per-locale
# file to pick, so no model-chooser plumbing exists for this backend.
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"
# None = fp32 weights (~2.4GB VRAM on GPU, RTF ~0.012 on this box). "int8" = ~4x
# smaller weights, but MEASURED ~3-5x SLOWER on the onnxruntime CUDA EP (16s audio:
# 185ms fp32 -> 907ms int8, ~CPU speed) because the CUDA EP dequantizes instead of
# using int8 tensor cores (those need TensorRT). So int8-on-GPU is pointless here:
# CPU-like speed while still burning GPU VRAM. Kept fp32. ponytail: for less VRAM
# without losing GPU speed you'd need fp16 (offline conversion); or just run on CPU
# (same speed as int8-on-GPU, zero VRAM).
QUANTIZATION = None
# Ignore VAD segments shorter than this -- nothing useful in a sub-200ms blip.
MIN_SEGMENT_S = 0.2


class STTGstParakeet(STTGstBase):
    __gtype_name__ = 'STTGstParakeet'

    # Same capture graph as the Whisper backend; only the appsink name differs.
    _pipeline_def = "pulsesrc name=stt_audio_src blocksize=3200 buffer-time=9223372036854775807 ! " \
                    "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                    "webrtcdsp noise-suppression-level=3 echo-cancel=false ! " \
                    "queue ! " \
                    "appsink name=ParakeetSink emit-signals=true sync=false"
    _pipeline_def_alt = "pulsesrc name=stt_audio_src blocksize=3200 buffer-time=9223372036854775807 ! " \
                        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                        "queue ! " \
                        "appsink name=ParakeetSink emit-signals=true sync=false"

    def __init__(self, current_locale=None):
        plugin = Gst.Registry.get().find_plugin("webrtcdsp")
        if plugin is not None:
            super().__init__(pipeline_definition=STTGstParakeet._pipeline_def)
            LOG_MSG.debug("using Webrtcdsp plugin")
        else:
            super().__init__(pipeline_definition=STTGstParakeet._pipeline_def_alt)
            LOG_MSG.debug("not using Webrtcdsp plugin")

        if self.pipeline is None:
            LOG_MSG.error("pipeline was not created")
            return

        self._appsink = self.pipeline.get_by_name("ParakeetSink")
        if self._appsink is None:
            LOG_MSG.error("no appsink element!")
            return
        self._appsink.connect("new-sample", self._on_new_sample)

        # Live mic level (peak-decay meter) surfaced to the IBus widget.
        self._audio_level = 0.0

        self._settings = Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        saved_device = self._settings.get_string("audio-device")
        if saved_device:
            self._apply_audio_device(saved_device)

        # The onnx-asr model is loaded lazily on the worker thread (first
        # segment) so neither building the pipeline nor enabling the engine
        # blocks the IBus main loop on a model load / first-run download.
        self._asr = None
        self._asr_load_failed = False
        self._provider_label = "?"   # set on model load; shown in timing logs

        # Capture always runs (for the meter); this gates transcription so
        # "recognition off" means "monitoring only", matching the Whisper backend.
        self._recognizing = False

        if VAD_MODULE_OK:
            # Lower threshold than the Whisper backend's 0.75: Parakeet emits
            # blank on any noise that leaks past the gate, so an aggressive gate
            # (which risks clipping quiet speech onsets) buys nothing here. 0.5
            # is the Silero default. ponytail: raise toward 0.6 only if idle
            # noise ever produces stray words (it shouldn't, given the model).
            self._vad = STTVad(
                speech_threshold=0.5,
                # 800ms: user prefers longer pauses (can pause mid-thought
                # without the utterance finalizing). Parakeet decodes in ~tens of
                # ms (faster still on GPU), so total latency after you stop is
                # basically this window. ponytail: lower toward 500/400 if you
                # want snappier finalize at the cost of shorter tolerated pauses.
                silence_duration_ms=800,
                speech_pad_ms=200,
                min_speech_duration_ms=300,
                max_speech_duration_s=30.0,
                freq_thold=100.0,
            )
            LOG_MSG.info("VAD active: %s", self._vad.backend_name)
        else:
            self._vad = None
            LOG_MSG.warning("VAD not available -- Parakeet backend requires VAD")

        self._process_queue = queue.Queue()
        self._process_thread = None
        self._stop_processing = False

    def __del__(self):
        LOG_MSG.info("Parakeet __del__")
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        super().__del__()

    def destroy(self):
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        self._appsink = None
        self._asr = None
        self._vad = None
        LOG_MSG.info("Parakeet.destroy() called")
        super().destroy()

    # --- audio capture -----------------------------------------------------

    def _on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio_data = np.frombuffer(map_info.data, dtype=np.int16)
        buf.unmap(map_info)
        audio_float = audio_data.astype(np.float32) / 32768.0

        if len(audio_float):
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            # Fast attack, slow release so the meter is readable at ~1 Hz.
            self._audio_level = rms if rms > self._audio_level else self._audio_level * 0.8

        if not self._recognizing or self._vad is None:
            return Gst.FlowReturn.OK

        for segment in self._vad.process(audio_float):
            LOG_MSG.debug("VAD segment ready: %.2f s", len(segment) / SAMPLE_RATE)
            self._enqueue(segment)

        return Gst.FlowReturn.OK

    def _enqueue(self, audio_float32):
        self._process_queue.put(audio_float32)
        if self._process_thread is None or not self._process_thread.is_alive():
            self._process_thread = threading.Thread(
                target=self._process_worker, daemon=True)
            self._process_thread.start()

    def _ensure_model(self):
        """Lazy-load the Parakeet model on the worker thread. Returns True once
        the model is usable; caches a failure so we don't retry every segment."""
        if self._asr is not None:
            return True
        if self._asr_load_failed:
            return False
        if not ONNX_ASR_AVAILABLE:
            LOG_MSG.error("onnx_asr not available -- "
                          "run: pip install onnx-asr huggingface_hub")
            self._asr_load_failed = True
            return False
        try:
            # Prefer CUDA when an onnxruntime-gpu build exposes it; always keep
            # CPU as fallback so a plain onnxruntime install just runs on CPU.
            # Install onnxruntime-gpu (+ cuDNN) to light up the GPU -- no code
            # change needed here.
            import onnxruntime as _rt
            avail = _rt.get_available_providers()
            use_cuda = "CUDAExecutionProvider" in avail
            if use_cuda and hasattr(_rt, "preload_dlls"):
                # Load CUDA/cuDNN from the nvidia-*-cuNN pip wheels so the CUDA EP
                # finds libcudnn.so.9 without a manual LD_LIBRARY_PATH. No-op if
                # the wheels aren't installed -- load then falls back to CPU below.
                try:
                    _rt.preload_dlls()
                except Exception as e:
                    LOG_MSG.debug("onnxruntime.preload_dlls() failed: %s", e)
            providers = (["CUDAExecutionProvider"] if use_cuda else []) \
                        + ["CPUExecutionProvider"]
            _qlabel = QUANTIZATION or "fp32"
            LOG_MSG.info("Loading Parakeet model: %s (providers=%s, quant=%s; "
                         "first run downloads it to the HF cache)",
                         PARAKEET_MODEL, providers, _qlabel)
            try:
                self._asr = onnx_asr.load_model(
                    PARAKEET_MODEL, quantization=QUANTIZATION, providers=providers)
                self._provider_label = f"{'CUDA' if use_cuda else 'CPU'}/{_qlabel}"
            except Exception as gpu_err:
                # A half-configured GPU (e.g. onnxruntime-gpu without cuDNN) can
                # throw at session creation. Don't let that kill the backend --
                # fall back to CPU, which always works.
                if len(providers) > 1:
                    LOG_MSG.warning("Parakeet load with %s failed (%s); "
                                    "retrying CPU-only", providers, gpu_err)
                    self._asr = onnx_asr.load_model(
                        PARAKEET_MODEL, quantization=QUANTIZATION,
                        providers=["CPUExecutionProvider"])
                    self._provider_label = f"CPU/{_qlabel} (fallback)"
                else:
                    raise
            LOG_MSG.info("Parakeet model ready (%s)", self._provider_label)
            return True
        except Exception as e:
            LOG_MSG.error("Failed to load Parakeet model: %s", e, exc_info=True)
            self._asr_load_failed = True
            return False

    def _process_worker(self):
        while not self._stop_processing:
            try:
                audio = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if len(audio) < int(MIN_SEGMENT_S * SAMPLE_RATE):
                    continue
                if not self._ensure_model():
                    continue
                # onnx-asr accepts a float32 mono waveform at the model's 16 kHz;
                # the VAD hands us exactly that. Transducer -> "" on silence/noise.
                # Time the decode so CPU vs GPU is measurable: this is the compute
                # from "VAD says you stopped" to "text ready" (total felt latency =
                # silence_duration_ms + this). RTF = decode / audio length.
                audio_s = len(audio) / SAMPLE_RATE
                _t0 = time.perf_counter()
                text = (self._asr.recognize(audio) or "").strip()
                decode_ms = (time.perf_counter() - _t0) * 1000.0
                rtf = (decode_ms / 1000.0 / audio_s) if audio_s else 0.0
                LOG_MSG.info("decode: %.2fs audio -> %.0f ms (RTF=%.3f) [%s]",
                             audio_s, decode_ms, rtf, self._provider_label)
                if text:
                    if text[-1] not in '.?!':
                        text += '.'
                    LOG_MSG.info("Parakeet transcription result: '%s'", text)
                    GLib.idle_add(self._emit_text, text)
                else:
                    LOG_MSG.debug("Parakeet returned empty (silence/noise)")
            except Exception as e:
                LOG_MSG.error("Parakeet transcription error: %s", e, exc_info=True)
            finally:
                self._process_queue.task_done()

    def _emit_text(self, text):
        self.emit("text", text)
        return False

    # --- public surface expected by sttengine.py ---------------------------

    def get_final_results(self, wait=True):
        """Flush the in-progress utterance to a final result. wait=True blocks
        until the worker drains (used off the main thread); the key handler
        calls wait=False so it never stalls the IBus main loop."""
        if self._vad is not None:
            remaining = self._vad.flush()
            if remaining is not None:
                self._enqueue(remaining)
        if wait:
            self._process_queue.join()

    def get_results(self):
        pass

    def set_use_partial_results(self, active):
        # Parakeet decodes per finalised VAD segment; there is no live partial
        # stream. Accepted as a no-op so the engine's calls stay harmless.
        pass

    def set_alternatives_num(self, num):
        pass

    def is_recognizing(self):
        return self._recognizing

    def set_recognizing(self, active):
        active = bool(active)
        if active == self._recognizing:
            return
        self._recognizing = active
        if active:
            LOG_MSG.info("recognition on")
        else:
            LOG_MSG.info("recognition off")
            self.get_final_results()
            if self._vad is not None:
                self._vad._in_speech = False

    # --- status surfaced to the IBus widget --------------------------------

    def get_audio_level(self):
        if not self.is_running():
            return 0.0
        return self._audio_level

    def get_vad_status(self):
        if self._vad is None:
            return ("", False)
        return (self._vad.backend_name,
                bool(self._recognizing and self._vad._in_speech))

    def get_model_name(self):
        return PARAKEET_MODEL

    # --- capture device selection (mirrors the Whisper backend) ------------

    @staticmethod
    def list_audio_sources():
        sources = []
        try:
            monitor = Gst.DeviceMonitor.new()
            monitor.add_filter("Audio/Source", None)
            monitor.start()
            for dev in monitor.get_devices() or []:
                props = dev.get_properties()
                node_name = props.get_string("node.name") if props else None
                if not node_name or node_name.endswith(".monitor"):
                    continue
                desc = dev.get_display_name() or node_name
                sources.append((node_name, desc))
            monitor.stop()
        except Exception as e:
            LOG_MSG.warning("could not enumerate audio sources: %s", e)
        return sources

    def get_audio_device(self):
        return self._settings.get_string("audio-device")

    def set_audio_device(self, device):
        device = device or ""
        self._settings.set_string("audio-device", device)
        self._apply_audio_device(device)

    def _apply_audio_device(self, device):
        if self.pipeline is None:
            return
        src = self.pipeline.get_by_name("stt_audio_src")
        if src is None:
            LOG_MSG.warning("no audio source element to set device on")
            return

        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        # None tells pulsesrc to follow the system default source.
        src.set_property("device", device if device else None)
        LOG_MSG.info("audio capture device set to %s", device or "(system default)")

        if state >= Gst.State.READY:
            self.pipeline.set_state(state)

    def _stop_real(self):
        self.get_final_results()
        return super()._stop_real()


if __name__ == "__main__":
    # Surface check: sttengine.py calls these on whatever backend is active.
    # This fails loudly here if a rename ever drops one, instead of crashing
    # IBus at runtime when the user switches to Parakeet. No audio/model needed.
    required = [
        "get_audio_device", "get_audio_level", "get_final_results",
        "get_model_name", "get_results", "get_vad_status", "has_model",
        "is_recognizing", "is_running", "list_audio_sources", "release",
        "run", "set_alternatives_num", "set_audio_device", "set_recognizing",
        "set_use_partial_results", "stop",
    ]
    missing = [m for m in required if not hasattr(STTGstParakeet, m)]
    assert not missing, f"missing engine-facing methods: {missing}"
    print("sttgstparakeet self-check: OK")
