import logging
import threading
import queue
import numpy as np
import re
import os

from pathlib import Path
from gi.repository import Gst, GLib, Gio
from sttutils import *
from sttgstbase import STTGstBase
from sttcurrentlocale import stt_current_locale
from sttwhispermodel import STTWhisperModel

LOG_MSG = logging.getLogger()
SPECIAL_PATTERN = re.compile(r'^(?:\[[^\]]+\]|\([^)]+\)|\*[^*]+\*)$',re.IGNORECASE)
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks.", "thanks",
    "you", "you.", "bye.", "bye",
    "all right.", "all right", "alright.", "alright",
    "okay.", "okay", "oh.", "oh",
    "hmm.", "hmm", "um.", "um", "uh.", "uh",
    "yes.", "yes", "no.", "no",
    "so.", "so", "well.", "well",
    "i'm sorry.", "sorry.", "sorry",
    "good.", "good", "right.", "right",
    "hello.", "hello", "hi.", "hi",
    "yeah.", "yeah", "yep.", "yep",
}


try:
    from pywhispercpp.model import Model
    WHISPER_AVAILABLE = True
except ImportError:
    LOG_MSG.warning("pywhispercpp not available. Install with: pip install pywhispercpp")
    WHISPER_AVAILABLE = False
try:
    from sttvad import STTVad
    VAD_MODULE_OK = True
except ImportError:
    LOG_MSG.warning("sttvad module not found - VAD disabled")
    VAD_MODULE_OK = False

WINDOW_SECONDS      = 5.0
MIN_SAMPLES         = 1024
SAMPLE_RATE         = 16000
AUDIO_CTX           = 768
# 0.2 is the whisper.cpp default. A value <= 0 disables temperature fallback,
# which is whisper's ONLY built-in escape from greedy-decoding repetition loops
# (the "same phrase pasted 15-50x" bug). The entropy/logprob thresholds below
# only DETECT a degenerate decode; fallback to a higher temperature is what
# actually breaks the loop. Keep this strictly positive.
TEMPERATURE_INC     = 0.2
MAX_TOKENS          = 256
_cpu_count = os.cpu_count() or 1
N_THREADS           = max(4, _cpu_count // 2 )
MAX_RING_QUEUE_DEPTH = 2
# Streaming partials use LocalAgreement-2: each tick re-decodes the recent audio
# and only COMMITS the words that two consecutive decodes agree on (longest
# common prefix of the still-unconfirmed tail). Committed words are never
# re-emitted, so the preview is stable and the finalize promotion sees the full
# accurate text. Validated byte-exact against a full decode on jfk.wav.
# 2.5s tick (was 0.5): re-decoding the whole buffer every 0.5s was O(n^2) GPU
# waste and pinned the worker; 2.5s keeps load low while staying responsive.
PARTIAL_INTERVAL_S  = 2.5
# Cap the per-tick decode to the most recent N seconds so decode time stays
# bounded (and below whisper's ~15s cost cliff) no matter how long you talk.
# Must exceed PARTIAL_INTERVAL_S by enough to overlap the previous decode for
# the agreement comparison; 12s gives ample overlap.
PARTIAL_DECODE_CAP_S = 12.0
MIN_SEGMENT_PROB    = 0.35


def _word_key(w):
    """Normalise a word for agreement comparison (case/punctuation-insensitive)."""
    return w.strip(".,!?;:…—-").lower()


def _common_prefix_len(a, b):
    """Length of the longest common (normalised) word prefix of a and b."""
    n = 0
    for x, y in zip(a, b):
        if _word_key(x) == _word_key(y):
            n += 1
        else:
            break
    return n


def _new_tail_after_agreed(decode_words, agreed_words, max_overlap=40):
    """Return the part of `decode_words` that comes AFTER the already-agreed
    prefix, aligning by word-overlap rather than by index.

    The decode may cover only a recent SUFFIX of the utterance (the partial
    decode is capped to bound cost), so the agreed prefix is not necessarily a
    positional prefix of decode_words. We find the largest k where the last k
    agreed words equal the first k decode words, and return decode_words[k:].
    Falls back to index-slicing when there is no overlap (e.g. the very first
    decode, agreed empty). Prevents both dropping text and re-emitting already
    -committed words on long, capped utterances.
    """
    if not agreed_words:
        return decode_words
    a_keys = [_word_key(w) for w in agreed_words]
    d_keys = [_word_key(w) for w in decode_words]
    limit = min(len(a_keys), len(d_keys), max_overlap)
    # Largest k: last k of agreed == first k of decode.
    for k in range(limit, 0, -1):
        if a_keys[-k:] == d_keys[:k]:
            return decode_words[k:]
    # No overlap found. If the decode still contains the whole agreed prefix
    # positionally (uncapped case), slice it off; else treat all as new.
    if len(decode_words) >= len(agreed_words) and \
            d_keys[:len(a_keys)] == a_keys:
        return decode_words[len(agreed_words):]
    return decode_words
# Option 1: promote the last partial as the final result (skip the redundant
# full re-decode) when that partial already covered at least this fraction of
# the finalized segment AND the uncovered tail is shorter than
# PROMOTE_MAX_TAIL_S. Otherwise fall back to a full decode so trailing words
# spoken just before the pause are never dropped.
PROMOTE_MIN_COVERAGE = 0.90
PROMOTE_MAX_TAIL_S   = 0.6
# When the tail-decode path fires, feed whisper this many trailing words of the
# already-streamed text as initial_prompt so the short tail slice is decoded as
# a mid-sentence continuation, not a freshly-capitalised standalone sentence
# (else a trailing word like "better" comes back as "Better."). ~20 words ≈ one
# clause of context -- enough to disambiguate continue-vs-new-sentence without
# risking whisper echoing a long prompt back into the output. Fed across any
# sentence boundary on purpose: 56% of partials already end in '.', and cutting
# at the boundary would starve exactly the case this fixes ("...stands out." +
# "better" must still continue, not capitalise).
TAIL_PROMPT_WORDS   = 20
# Output guard against runaway repetition loops that slip past whisper's own
# fallback. A phrase repeated more than this many times in a row is trimmed
# back to this many copies. Raise if it ever clips legitimate speech.
MAX_PHRASE_REPEAT   = 2


def _collapse_repetitions(text, keep=MAX_PHRASE_REPEAT, max_ngram=10):
    """Trim runaway Whisper repetition loops.

    A phrase of up to ``max_ngram`` words that repeats more than ``keep`` times
    in a row is reduced to ``keep`` copies. Words are compared
    case-insensitively and ignoring surrounding punctuation, but the original
    tokens are preserved in the output. Ordinary short repeats survive.
    """
    words = text.split()
    n_words = len(words)
    if n_words <= keep:
        return text

    def _norm(w):
        return w.strip(".,!?;:…—-").lower()

    keys = [_norm(w) for w in words]
    out = []
    i = 0
    collapsed_any = False
    while i < n_words:
        matched = False
        # Try the shortest phrase first: a single-word run ("hi hi hi ...")
        # collapses to one word, while a phrase-level loop ("the cat sat
        # the cat sat ...") falls through to the n that actually repeats.
        upper = min(max_ngram, (n_words - i) // 2)
        for n in range(1, upper + 1):
            gram = keys[i:i + n]
            reps = 1
            j = i + n
            while j + n <= n_words and keys[j:j + n] == gram:
                reps += 1
                j += n
            if reps > keep:
                out.extend(words[i:i + n * keep])
                i = j
                matched = True
                collapsed_any = True
                break
        if not matched:
            out.append(words[i])
            i += 1

    if collapsed_any:
        LOG_MSG.warning("Collapsed repetition loop: %d words -> %d words",
                        n_words, len(out))
        return ' '.join(out)
    return text


class STTGstWhisper(STTGstBase):
    __gtype_name__ = 'STTGstWhisper'
    _pipeline_def = "pulsesrc name=stt_audio_src blocksize=3200 buffer-time=9223372036854775807 ! " \
                    "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                    "webrtcdsp noise-suppression-level=3 echo-cancel=false ! " \
                    "queue ! " \
                    "appsink name=WhisperSink emit-signals=true sync=false"

    _pipeline_def_alt = "pulsesrc name=stt_audio_src blocksize=3200 buffer-time=9223372036854775807 ! " \
                        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                        "queue ! " \
                        "appsink name=WhisperSink emit-signals=true sync=false"

    def __init__(self, current_locale=None):
        plugin = Gst.Registry.get().find_plugin("webrtcdsp")
        if plugin is not None:
            super().__init__(pipeline_definition=STTGstWhisper._pipeline_def)
            LOG_MSG.debug("using Webrtcdsp plugin")
        else:
            super().__init__(pipeline_definition=STTGstWhisper._pipeline_def_alt)
            LOG_MSG.debug("not using Webrtcdsp plugin")

        if self.pipeline is None:
            LOG_MSG.error("pipeline was not created")
            return

        self._appsink = self.pipeline.get_by_name("WhisperSink")
        if self._appsink is None:
            LOG_MSG.error("no appsink element!")
            return

        self._appsink.connect("new-sample", self._on_new_sample)

        # Live mic level (peak-decay meter), surfaced to the IBus widget.
        self._audio_level = 0.0

        # Apply a saved capture device, if the user pinned one in the widget.
        self._settings = Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        saved_device = self._settings.get_string("audio-device")
        if saved_device:
            self._apply_audio_device(saved_device)

        if current_locale is None:
            self._current_locale = stt_current_locale()
        else:
            self._current_locale = current_locale

        self._locale_id = self._current_locale.connect("changed", self._locale_changed)

        self._model_id = 0
        self._model    = None
        self._whisper  = None
        self._vad      = None
        self._set_model()
        self._ring_buffer     = np.array([], dtype=np.float32)
        self._ring_lock       = threading.Lock()
        self._chunk_buffer    = []
        self._chunk_samples   = 0

        if VAD_MODULE_OK:
            self._vad = STTVad(
                speech_threshold=0.5,
                # 800ms: the user's chosen optimum between snappy finalize and
                # stitching short mid-thought pauses. This silence window is the
                # only felt delay now that injection is instant (ydotoold +
                # zeroed ydotool key-delay/key-hold). Was 300 -> 1200 -> 1000 ->
                # 800.
                silence_duration_ms=800,
                speech_pad_ms=200,
                min_speech_duration_ms=300,
                # Raised 15 -> 30: a long uninterrupted sentence used to be
                # force-cut mid-thought at 15s. Decode is ~75x realtime on this
                # GPU (30s -> ~400ms), so the larger cap costs little.
                max_speech_duration_s=30.0,
                freq_thold=100.0,
            )
            LOG_MSG.info("VAD active: %s", self._vad.backend_name)
        else:
            self._vad = None
            LOG_MSG.warning("VAD not available")

        self._process_queue   = queue.Queue()
        self._process_thread  = None
        self._stop_processing = False
        self._use_partial_results = False
        self._partial_timer_id = 0
        self._last_partial_samples = 0
        # Option 1 ("promote last partial"): the most recent partial decode's
        # text and the sample count it was decoded from. When a segment
        # finalizes and the last partial already covered nearly all of it, we
        # emit the partial as the final result and skip the redundant
        # full-segment re-decode. Guarded by _partial_lock since the worker
        # thread writes these and the (GLib main-thread) finalize path reads.
        self._last_partial_text = None
        self._last_partial_text_samples = 0
        # LocalAgreement-2 streaming state (worker thread only). _agreed_words is
        # the confirmed word prefix of the current utterance; _prev_tail is the
        # previous decode's still-unconfirmed tail, used to find what the next
        # decode agrees with. Both reset on speech onset (_start_partial_timer).
        self._agreed_words = []
        self._prev_tail = []
        self._partial_lock = threading.Lock()
        # Whether to actually transcribe. The pipeline keeps capturing (so the
        # mic meter stays live) whenever the engine is enabled; this flag gates
        # the VAD/Whisper path so "recognition off" means "monitoring only".
        self._recognizing = False

    def __del__(self):
        LOG_MSG.info("Whisper __del__")
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        super().__del__()

    def destroy(self):
        self._stop_partial_timer()
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)

        self._current_locale.disconnect(self._locale_id)
        self._locale_id = 0

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._appsink = None
        self._whisper = None
        self._vad     = None
        LOG_MSG.info("Whisper.destroy() called")
        super().destroy()

    def _build_lang_code(self):
        if not self._current_locale or not self._current_locale.locale:
            return None
        code = self._current_locale.locale[:2].lower()
        return code if code.isalpha() else None

    def _load_whisper_model(self, model_path):
        """Load Whisper model using pywhispercpp"""
        if not WHISPER_AVAILABLE:
            LOG_MSG.error("pywhispercpp not available")
            return False

        try:
            LOG_MSG.info("Loading Whisper model: %s", model_path)
            lang_code = self._build_lang_code()

            kwargs = dict(
                print_realtime=False,
                print_progress=False,
                single_segment=True,
                no_context=True,
                audio_ctx=AUDIO_CTX,
                temperature_inc=TEMPERATURE_INC,
                max_tokens=MAX_TOKENS,
                n_threads=N_THREADS,
                # Raised from 0.5/2.4 so a looping/low-quality decode trips the
                # temperature fallback (re-decode at higher temp) more readily.
                # Repetition loops have LOW entropy, so they slip past a tight
                # entropy_thold; a higher value catches more of them.
                no_speech_thold=0.6,
                logprob_thold=-1.0,
                entropy_thold=2.8,
                suppress_blank=True,
            )

            if lang_code:
                kwargs["language"] = lang_code

            self._whisper = Model(model_path, **kwargs)
            return True

        except Exception as e:
            LOG_MSG.error("Failed to load Whisper model: %s", e)
            self._whisper = None
            return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model path does not exist (%s - %s)",
                        self._model.get_name() if self._model else "None",
                        self._model.get_path() if self._model else "None")
            self._whisper = None
            self.emit("model-changed")
            return

        new_model_path = self._model.get_path()
        LOG_MSG.debug("model ready %s", new_model_path)

        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        success = self._load_whisper_model(new_model_path)

        if state >= Gst.State.READY:
            self.pipeline.set_state(state)

        if success:
            if self._vad is not None:
                self._vad.reset()
            self.emit("model-changed")

    def _model_changed(self, model):
        self._set_model_path()

    def _set_model(self):
        if (self._model is not None and
                self._model.get_locale() == self._current_locale.locale):
            return

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._model = STTWhisperModel(locale_str=self._current_locale.locale)
        self._model_id = self._model.connect("changed", self._model_changed)
        self._set_model_path()

    def _locale_changed(self, locale):
        self._set_model()

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

        # Capture always runs (for the meter); only transcribe when recognising.
        if not self._recognizing:
            return Gst.FlowReturn.OK

        if self._vad is not None:
            was_in_speech = self._vad._in_speech
            segments = self._vad.process(audio_float)
            for segment in segments:
                LOG_MSG.debug("VAD segment ready: %.2f s", len(segment) / SAMPLE_RATE)
                self._stop_partial_timer()
                self._enqueue_for_transcription(segment, source='vad')

            if self._use_partial_results and self._vad._in_speech and self._partial_timer_id == 0:
                self._start_partial_timer()
            elif not self._vad._in_speech and self._partial_timer_id != 0:
                self._stop_partial_timer()
        else:
            self._chunk_buffer.append(audio_data)
            self._chunk_samples += len(audio_data)
            window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
            if self._chunk_samples >= window_samples:
                self._flush_chunks_to_ring()
                self._dispatch_to_worker()

        return Gst.FlowReturn.OK

    def _flush_chunks_to_ring(self):
        if not self._chunk_buffer:
            return

        new_samples = np.concatenate(self._chunk_buffer).astype(np.float32) / 32768.0
        self._chunk_buffer.clear()
        self._chunk_samples = 0

        max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        with self._ring_lock:
            self._ring_buffer = np.concatenate([self._ring_buffer, new_samples])
            if len(self._ring_buffer) > max_samples:
                self._ring_buffer = self._ring_buffer[-max_samples:]

    def _dispatch_to_worker(self):

        with self._ring_lock:
            if len(self._ring_buffer) < MIN_SAMPLES:
                return
            snapshot = self._ring_buffer.copy()

        self._enqueue_for_transcription(snapshot, source='ring')

    def _enqueue_for_transcription(self, audio_float32: np.ndarray, source: str,
                                   total_samples: int = 0):
        if source == 'partial':
            temp = []
            while True:
                try:
                    item = self._process_queue.get_nowait()
                    if item[0] == 'partial':
                        self._process_queue.task_done()
                    else:
                        temp.append(item)
                except queue.Empty:
                    break
            for item in temp:
                self._process_queue.put(item)
        elif source == 'ring':
            pending_ring = sum(
                1 for item in list(self._process_queue.queue)
                if item[0] == 'ring'
            )
            if pending_ring >= MAX_RING_QUEUE_DEPTH:
                temp = []
                dropped = False
                while True:
                    try:
                        item = self._process_queue.get_nowait()
                        if item[0] == 'ring' and not dropped:
                            self._process_queue.task_done()
                            dropped = True
                        else:
                            temp.append(item)
                    except queue.Empty:
                        break
                for item in temp:
                    self._process_queue.put(item)

        self._process_queue.put((source, audio_float32, total_samples))

        if self._process_thread is None or not self._process_thread.is_alive():
            self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
            self._process_thread.start()

    def _maybe_promote_partial(self, partial_text, partial_samples,
                               segment_samples):
        """Return the partial text to use as the final result, or None to fall
        back to a full decode.

        Promote only when a recent partial covered nearly the whole segment
        (>= PROMOTE_MIN_COVERAGE) and the uncovered tail is shorter than
        PROMOTE_MAX_TAIL_S, so trailing words spoken just before the pause are
        never lost. Worst case (no usable partial) is identical to the old
        always-decode behaviour.
        """
        if not partial_text or partial_samples <= 0 or segment_samples <= 0:
            return None

        # The partial may have been decoded from slightly more or fewer samples
        # than the final segment (padding, leftover). Coverage is how much of
        # the segment the partial already saw; tail is what it missed.
        coverage = min(partial_samples, segment_samples) / segment_samples
        tail_samples = max(0, segment_samples - partial_samples)
        tail_s = tail_samples / SAMPLE_RATE

        if coverage < PROMOTE_MIN_COVERAGE or tail_s > PROMOTE_MAX_TAIL_S:
            LOG_MSG.debug("Not promoting partial (coverage=%.2f, tail=%.2fs) "
                          "-- full decode", coverage, tail_s)
            return None

        # Apply the same content filters a decoded final would get.
        text = _collapse_repetitions(partial_text.strip())
        if not text or text.lower() in _HALLUCINATIONS:
            return None
        if SPECIAL_PATTERN.match(text):
            return None
        return text

    def _process_worker(self):
        """Background worker to process audio"""
        while not self._stop_processing:
            try:
                source, audio, total_samples = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._whisper is None:
                self._process_queue.task_done()
                continue

            # Option 1: on a finalize, try to promote the last partial instead
            # of re-decoding the whole segment. Partials are consumed (reset)
            # on every finalize so a stale one can't be reused next utterance.
            if source != 'partial':
                with self._partial_lock:
                    partial_text = self._last_partial_text
                    partial_samples = self._last_partial_text_samples
                    self._last_partial_text = None
                    self._last_partial_text_samples = 0

                promoted = self._maybe_promote_partial(
                    partial_text, partial_samples, len(audio))
                if promoted is not None:
                    LOG_MSG.info("Promoted last partial as final (skipped "
                                 "re-decode): '%s'", promoted)
                    text = promoted
                    if text[-1] not in '.?!':
                        text += '.'
                    GLib.idle_add(self._emit_text, text)
                    self._process_queue.task_done()
                    continue

                # Promotion failed (the last partial lagged the segment close by
                # more than PROMOTE_MAX_TAIL_S). Rather than re-decode the WHOLE
                # segment -- which on a long (>~15s) utterance makes whisper
                # duplicate phrases and drop the tail (observed: a 26s blob came
                # back worse than the streamed partials) -- keep the correct
                # streamed text and decode ONLY the uncovered tail, then append.
                # The streamed partial is already-agreed text; whisper only ever
                # sees the short tail, so its long-audio failure mode is avoided.
                if (partial_text and partial_samples > 0
                        and 0 < partial_samples < len(audio)):
                    tail_audio = audio[partial_samples:]
                    tail_s = len(tail_audio) / SAMPLE_RATE
                    LOG_MSG.info("Promotion failed; keeping streamed text + "
                                 "decoding %.1fs tail (avoids long re-decode)",
                                 tail_s)
                    # Give the tail decode the last few streamed words as
                    # context so it continues the sentence instead of starting
                    # a new (capitalised) one.
                    prompt = ' '.join(
                        partial_text.strip().split()[-TAIL_PROMPT_WORDS:])
                    try:
                        tail_words = self._decode_words(
                            tail_audio, initial_prompt=prompt)
                    except Exception as e:
                        LOG_MSG.error("Tail decode failed: %s", e)
                        tail_words = []
                    # The tail is short, low-context audio -- whisper's prime
                    # hallucination case ("Thank you.", "you.", "um."). The
                    # streamed text already holds the real words, so drop a tail
                    # that, on its own, is nothing but a known filler phrase.
                    tail_text = ' '.join(tail_words).strip()
                    if tail_text and tail_text.lower() in _HALLUCINATIONS:
                        LOG_MSG.info("Dropped hallucinated tail: '%s'", tail_text)
                        tail_words = []
                    text = (partial_text.strip() + " "
                            + ' '.join(tail_words)).strip()
                    text = _collapse_repetitions(text)
                    if text and text.lower() not in _HALLUCINATIONS:
                        if text[-1] not in '.?!':
                            text += '.'
                        LOG_MSG.info("Final (streamed + tail): '%s'", text)
                        GLib.idle_add(self._emit_text, text)
                    self._process_queue.task_done()
                    continue

            # Partials use LocalAgreement-2 streaming, handled separately so the
            # full-decode path below stays simple. The partial item's audio is
            # the recent capped window; total_samples is the whole utterance.
            if source == 'partial':
                try:
                    self._handle_partial_agreement(audio, total_samples)
                except Exception as e:
                    LOG_MSG.error("Partial agreement error: %s", e, exc_info=True)
                self._process_queue.task_done()
                continue

            try:
                LOG_MSG.debug("Starting transcription of %d samples", len(audio))
                segments = self._whisper.transcribe(audio)
                text_parts = []
                low_confidence = False
                for segment in segments:
                    if not hasattr(segment, 'text'):
                        continue

                    segment_text = segment.text.strip()
                    if SPECIAL_PATTERN.match(segment_text):
                        continue

                    prob = getattr(segment, 'probability', float('nan'))
                    if prob == prob and prob < MIN_SEGMENT_PROB:
                        LOG_MSG.debug("Low confidence segment (%.3f): '%s'",
                                      prob, segment_text)
                        low_confidence = True
                        continue

                    if segment_text:
                        text_parts.append(segment_text)
                        LOG_MSG.debug("Segment text: '%s' (prob=%.3f)",
                                      segment_text, prob)

                text = ' '.join(text_parts).strip()
                text = _collapse_repetitions(text)

                if text and text.lower() not in _HALLUCINATIONS:
                    deduped = _collapse_repetitions(text)
                    if deduped != text:
                        LOG_MSG.info("Collapsed repetition: '%s' -> '%s'",
                                     text, deduped)
                        text = deduped
                    if text and text[-1] not in '.?!':
                        text += '.'
                    LOG_MSG.info("Whisper transcription result: '%s'", text)
                    GLib.idle_add(self._emit_text, text)
                elif text:
                    LOG_MSG.debug("Filtered hallucination: '%s'", text)
                elif low_confidence:
                    LOG_MSG.info("Discarded low-confidence audio segment")
                else:
                    LOG_MSG.debug("No text transcribed from audio")

            except Exception as e:
                LOG_MSG.error("Whisper transcription error: %s", e, exc_info=True)

            self._process_queue.task_done()

    def _decode_words(self, audio, initial_prompt=None):
        """Decode audio and return a flat list of words (whitespace tokens),
        applying the special-pattern and confidence filters but not repetition
        collapse (that is applied to the assembled preview/final text).

        ``initial_prompt`` gives whisper the words spoken just before this audio
        so an isolated tail slice is decoded as a mid-sentence continuation
        (lowercased, not re-capitalised) instead of a standalone sentence. The
        model is loaded with no_context=True, but initial_prompt is an explicit
        user prompt and is honoured independently of that flag."""
        kwargs = {}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        parts = []
        for segment in self._whisper.transcribe(audio, **kwargs):
            if not hasattr(segment, 'text'):
                continue
            seg_text = segment.text.strip()
            if not seg_text or SPECIAL_PATTERN.match(seg_text):
                continue
            prob = getattr(segment, 'probability', float('nan'))
            if prob == prob and prob < MIN_SEGMENT_PROB:
                continue
            parts.append(seg_text)
        return ' '.join(parts).split()

    def _handle_partial_agreement(self, audio, total_samples):
        """LocalAgreement-2 partial step (validated byte-exact vs full decode).

        Re-decode the recent audio, strip the already-agreed word prefix, then
        commit the words this decode and the previous one agree on (longest
        common prefix). Emit committed + current unconfirmed tail as the live
        preview, and store the full preview so a finalize can promote it.

        The decode is capped to bound cost, so it may cover only a recent suffix
        of the utterance; _new_tail_after_agreed aligns by word-overlap so the
        agreed prefix is stripped correctly whether or not it is still in the
        decode window. The agreement is what guarantees correctness (no dropped
        opening, no overlap duplication) -- validated byte-exact on jfk.wav and
        on a 22s clip that exercises the capped path.
        """
        words = self._decode_words(audio)

        with self._partial_lock:
            agreed = self._agreed_words
            prev_tail = self._prev_tail

            new_tail = _new_tail_after_agreed(words, agreed)
            n_agree = _common_prefix_len(new_tail, prev_tail)
            if n_agree:
                agreed = agreed + new_tail[:n_agree]
                prev_tail = new_tail[n_agree:]
            else:
                prev_tail = new_tail

            self._agreed_words = agreed
            self._prev_tail = prev_tail

            preview = _collapse_repetitions(' '.join(agreed + prev_tail).strip())
            # Store the full preview so finalize can promote it instead of
            # re-decoding (coverage measured against the whole utterance).
            if preview and preview.lower() not in _HALLUCINATIONS:
                self._last_partial_text = preview
                self._last_partial_text_samples = total_samples

        if preview and preview.lower() not in _HALLUCINATIONS:
            LOG_MSG.debug("Partial (agreed=%dw): '%s'", len(agreed), preview)
            GLib.idle_add(self._emit_partial_text, preview)

    def _start_partial_timer(self):
        self._last_partial_samples = 0
        # New speech run: drop any partial captured in a previous run so it can
        # never be promoted onto this utterance's segment, and clear the
        # local-agreement state so committed words never leak across utterances.
        with self._partial_lock:
            self._last_partial_text = None
            self._last_partial_text_samples = 0
            self._agreed_words = []
            self._prev_tail = []
        self._partial_timer_id = GLib.timeout_add(
            int(PARTIAL_INTERVAL_S * 1000), self._on_partial_tick)

    def _stop_partial_timer(self):
        if self._partial_timer_id != 0:
            GLib.source_remove(self._partial_timer_id)
            self._partial_timer_id = 0
        self._last_partial_samples = 0

    def _on_partial_tick(self):
        if self._vad is None or not self._vad._in_speech:
            self._partial_timer_id = 0
            return False

        pending = self._vad.get_pending_audio()
        if pending is None or len(pending) == self._last_partial_samples:
            return True

        self._last_partial_samples = len(pending)
        # Cap the decoded audio to the most recent PARTIAL_DECODE_CAP_S so decode
        # time stays bounded on long utterances (the agreement, not the cap,
        # guarantees correctness). total_samples is the whole utterance length,
        # used so a finalize can measure promotion coverage correctly.
        total_samples = len(pending)
        cap = int(PARTIAL_DECODE_CAP_S * SAMPLE_RATE)
        window = pending[-cap:] if total_samples > cap else pending
        self._enqueue_for_transcription(
            window, source='partial', total_samples=total_samples)
        return True

    def _emit_partial_text(self, text):
        self.emit("partial-text", text)
        return False

    def _emit_text(self, text):
        self.emit("text", text)
        return False

    def get_final_results(self, wait=True):
        """Finalize the current utterance.

        wait=True (default) blocks until the worker has emitted the final text
        -- used by callers that need the result settled (recognition-off, the
        shortcut dialog, internal finalize).

        wait=False enqueues the final job and returns IMMEDIATELY. This MUST be
        used from the IBus main/UI thread (e.g. the key-event handler): the
        worker is single-threaded, so a blocking _process_queue.join() there
        holds the main loop until decoding finishes -- which froze the keyboard
        (no keystrokes delivered) while a partial/final decode was in flight.
        The result is still emitted asynchronously via GLib.idle_add.
        """
        if self._vad is not None:
            remaining = self._vad.flush()
            if remaining is not None:
                self._enqueue_for_transcription(remaining, source='vad')
        else:
            self._flush_chunks_to_ring()
            self._dispatch_to_worker()

        if wait:
            self._process_queue.join()

    def get_results(self):
        pass

    def set_use_partial_results(self, active):
        self._use_partial_results = active

    def set_alternatives_num(self, num):
        pass

    def is_recognizing(self):
        """Whether speech is being transcribed (distinct from capturing)."""
        return self._recognizing

    def set_recognizing(self, active):
        active = bool(active)
        if active == self._recognizing:
            return
        self._recognizing = active
        if active:
            LOG_MSG.info("recognition on")
        else:
            # Turning off: flush any in-progress speech to a final result, then
            # reset transcription state so nothing leaks while monitoring only.
            LOG_MSG.info("recognition off")
            self._stop_partial_timer()
            self.get_final_results()
            if self._vad is not None:
                self._vad._in_speech = False
            self._chunk_buffer.clear()
            self._chunk_samples = 0

    # --- Status surfaced to the IBus widget --------------------------------

    def get_audio_level(self):
        """Current mic level as a 0..1 RMS value (0 when not recording)."""
        if not self.is_running():
            return 0.0
        return self._audio_level

    def get_vad_status(self):
        """(backend_name, in_speech). backend_name is '' if VAD is unavailable."""
        if self._vad is None:
            return ("", False)
        return (self._vad.backend_name, bool(self._recognizing and self._vad._in_speech))

    def get_model_name(self):
        """Friendly name of the loaded model, or its file basename, or None."""
        if self._model is None:
            return None
        name = self._model.get_name()
        if name:
            return name
        path = self._model.get_path()
        return os.path.basename(path) if path else None

    # --- Capture device selection ------------------------------------------

    @staticmethod
    def list_audio_sources():
        """Return [(node_name, description), ...] for selectable input devices."""
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
        """Configured device node.name, or '' when following the system default."""
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

    def has_model(self):
        if self._model is None or self._model.available() is False:
            return False
        return super().has_model()

    def _stop_real(self):
        self.get_final_results()
        return super()._stop_real()
