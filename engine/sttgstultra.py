# vim:set et sts=4 sw=4:
#
# ibus-stt - Speech To Text engine for IBus
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Parakeet Ultra backend (moondream/parakeet-ultra via the Photon runtime).

Ultra is a post-trained parakeet-tdt-0.6b-v3: same 25 languages, same
architecture, lower WER everywhere (notably in background noise), and bf16
weights (~1.26GB) instead of our fp32 ONNX (~2.4GB) -- which is what lets it
fit the 4GB laptop GPU without OOMing.

Everything except model load + decode is inherited from STTGstParakeet:
capture graph, Silero VAD segmentation, live partial previews, and the
"commit the last partial if the final is empty/failed" guard.

PRIVACY: Photon ships usage telemetry (hostname, GPU, request counts every 60s
to api.moondream.ai) and a startup thread that pings huggingface.co for
config.json. Both are disabled in _photon_private() before the engine is
created, and HF is forced offline once the weights are cached. The
__main__ self-check proves it: it loads and decodes with every outbound socket
connect raising.
"""

import logging
import os

from sttgstparakeet import STTGstParakeet, SAMPLE_RATE

LOG_MSG = logging.getLogger()

ULTRA_MODEL = "moondream/parakeet-ultra"
# Photon's single-pass batch defaults to 64 on small GPUs; we decode one
# utterance at a time, so reserve for one.
BATCH_CAPACITY = 1
# CUDA graphs cut per-decode launch overhead but cost VRAM per captured shape.
# ponytail: set False if VRAM gets tight on the 4GB card.
CUDA_GRAPHS = True


def _weights_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(ULTRA_MODEL, "model.safetensors"), str)
    except Exception:
        return False


def _photon_private():
    """Import Photon with every phone-home path disabled. Returns md.photon."""
    if _weights_cached():
        # No network at all once the weights are local: HF reads the cache only.
        # The env var alone is too late -- huggingface_hub reads it at import,
        # and onnx_asr/our cache check import it first -- so flip the live flag
        # that is_offline_mode() consults on every request.
        os.environ["HF_HUB_OFFLINE"] = "1"
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_OFFLINE = True
    import kestrel.photon as kphoton
    import kestrel.model_download as kdownload

    async def _no_flush(self, *, rotate=True):
        return "anonymous"   # what the server says for key-less use; nothing sent

    kphoton.PhotonReporter._flush_window = _no_flush
    kphoton.PhotonReporter.start = lambda self: None
    kdownload.probe_supported_model_configs = lambda *a, **k: None
    import moondream as md
    return md.photon


class STTGstUltra(STTGstParakeet):
    __gtype_name__ = 'STTGstUltra'

    def _ensure_model(self):
        if self._asr is not None:
            return True
        if self._asr_load_failed:
            return False
        self._model_state = "loading"
        try:
            photon = _photon_private()
            LOG_MSG.info("Loading %s via Photon (first run downloads ~1.26GB)",
                         ULTRA_MODEL)
            self._asr = photon(ULTRA_MODEL, device="cuda",
                               single_pass_batch_capacity=BATCH_CAPACITY,
                               enable_cuda_graphs=CUDA_GRAPHS)
            self._provider_label = "Photon/CUDA bf16"
            LOG_MSG.info("Parakeet Ultra ready")
            self._model_state = "ready"
            return True
        except Exception as e:
            LOG_MSG.error("Failed to load Parakeet Ultra: %s", e, exc_info=True)
            self._asr_load_failed = True
            self._model_state = "failed"
            return False

    def _decode(self, audio, source):
        try:
            out = self._asr.transcribe(audio=audio, sample_rate=SAMPLE_RATE)
            return (out.get("text") or "").strip()
        except Exception as e:
            if source == 'partial':
                raise
            # "" -> the worker commits the last partial instead of losing it.
            LOG_MSG.error("Ultra decode failed: %s", e, exc_info=True)
            return ""

    def get_model_name(self):
        return ULTRA_MODEL if self._asr is not None else None

    def destroy(self):
        if self._asr is not None:
            self._asr.close()
        super().destroy()


if __name__ == "__main__":
    # Privacy + decode check: load and transcribe with ALL outbound network
    # blocked. Any telemetry/probe/download attempt raises and fails loudly.
    # Needs the weights cached (download_model.py --ultra) and a free GPU.
    import socket
    import sys
    import numpy as np

    attempts = []
    _real_connect = socket.socket.connect

    def _guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host in ("127.0.0.1", "::1", "localhost") or self.family == socket.AF_UNIX:
            return _real_connect(self, addr)
        attempts.append(addr)
        raise OSError(f"network blocked by self-check: {addr}")

    socket.socket.connect = _guard
    _real_gai = socket.getaddrinfo

    def _gai_guard(host, *a, **k):
        if host not in (None, "127.0.0.1", "::1", "localhost"):
            attempts.append(("dns", host))
            raise OSError(f"DNS blocked by self-check: {host}")
        return _real_gai(host, *a, **k)

    socket.getaddrinfo = _gai_guard
    assert _weights_cached(), "weights not cached -- run download_model.py --ultra"
    client = _photon_private()(ULTRA_MODEL, device="cuda",
                               single_pass_batch_capacity=BATCH_CAPACITY,
                               enable_cuda_graphs=CUDA_GRAPHS)
    if len(sys.argv) > 1:   # any audio file; Photon decodes + resamples it
        out = client.transcribe(audio=sys.argv[1])
    else:
        out = client.transcribe(audio=np.zeros(SAMPLE_RATE * 2, np.float32),
                                sample_rate=SAMPLE_RATE)
    print("text:", repr(out["text"]))
    import time; time.sleep(2)   # give any background thread time to try
    client.close()
    assert not attempts, f"network attempted: {attempts}"
    print("sttgstultra self-check: OK (no network)")
