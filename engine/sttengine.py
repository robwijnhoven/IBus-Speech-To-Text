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

import os
import time
import subprocess
import logging

from gettext import gettext as _

import gi

gi.require_version('IBus', '1.0')
gi.require_version('Pango', '1.0')
gi.require_version('Gtk', '4.0')

from gi.repository import IBus
from gi.repository import Gtk, Adw
from gi.repository import Gio

from sttutils import *
from sttgstfactory import stt_gst_factory_default
from sttsegmentprocess import STTSegmentProcess, STTParseModes


__all__ = (
    "STTEngine"
)

GLib.set_prgname('ibus-engine-stt')

LOG_MSG=logging.getLogger()
# LED indicators shown at the right of the status lines in the IBus widget.
_LED_GREEN  = "🟢"
_LED_ORANGE = "🟠"
_LED_RED    = "🔴"
# Mic energy detection: an RMS level above this counts as "heard something".
_ENERGY_THRESHOLD   = 0.01
# Mic LED stays green for this many 1 Hz ticks after the last detected energy,
# then turns orange. 10 ticks ~= 10 seconds.
_ENERGY_GREEN_TICKS = 10

# On the US-International (us+intl) keyboard layout these chars are dead keys:
# typed alone they wait to compose with the next character (' + m -> ḿ). Since
# we inject text with `ydotool type`, which goes through the active layout, we
# follow each dead-key char with a space. The us-intl compose engine consumes
# that space and emits the literal symbol, so "I'm" types as "I'm" not "Iḿ".
# (Verified: typing "A' B" yields "A'B".)
_US_INTL_DEAD_KEYS = "'\"`~^"

def _escape_dead_keys(text):
    out = []
    for ch in text:
        out.append(ch)
        if ch in _US_INTL_DEAD_KEYS:
            out.append(' ')
    return ''.join(out)


def _detect_display_server():
    """Return 'wayland' or 'x11' for the current session.

    The text-injection backend depends on this: Wayland needs ydotool (uinput),
    while X11 uses xdotool/XTEST. We never hardcode a backend -- one repo is
    shared between a Wayland desktop and an X11 laptop. An explicit
    STT_INJECT_BACKEND env var ('ydotool' | 'xdotool' | 'wayland' | 'x11')
    overrides detection for odd setups (e.g. XWayland-only tooling).

    Detection order: XDG_SESSION_TYPE, then the presence of WAYLAND_DISPLAY vs
    DISPLAY. Defaults to x11 if nothing is conclusive (the safest legacy path).
    """
    override = os.environ.get("STT_INJECT_BACKEND", "").strip().lower()
    if override in ("wayland", "ydotool"):
        return "wayland"
    if override in ("x11", "xdotool"):
        return "x11"

    session = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if session == "wayland":
        return "wayland"
    if session == "x11":
        return "x11"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    return "x11"


_layout_dead_keys_cache = None


def _layout_uses_dead_keys():
    """True if the active keyboard layout composes dead keys (e.g. us-intl).

    When we type *through* the layout (ydotool, or `xdotool type`), us-intl
    turns ' " ` ~ ^ into dead keys that compose with the next char ("I'm" ->
    "Iḿ"); _escape_dead_keys() works around it. On a plain `us` layout that
    escaping would instead emit a literal space, so it must only run when the
    layout actually needs it.

    STT_DEAD_KEYS=1/0 forces it. Otherwise we probe setxkbmap then localectl
    for an 'intl'/'dvorak-intl'/... variant. Defaults to False when nothing is
    conclusive -- the safe direction for `xdotool type` (text stays verbatim).
    Cached: the layout does not change within a session in practice.
    """
    global _layout_dead_keys_cache
    if _layout_dead_keys_cache is not None:
        return _layout_dead_keys_cache

    forced = os.environ.get("STT_DEAD_KEYS", "").strip().lower()
    if forced in ("1", "true", "yes"):
        _layout_dead_keys_cache = True
        return True
    if forced in ("0", "false", "no"):
        _layout_dead_keys_cache = False
        return False

    result = False
    for cmd in (["setxkbmap", "-query"], ["localectl", "status"]):
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=2,
                                 text=True)
        except Exception:
            continue
        blob = (out.stdout or "") + (out.stderr or "")
        if "intl" in blob.lower():   # us(intl), us-intl, X11 Variant: intl, ...
            result = True
            break
        if "variant:" in blob.lower() or "X11 Layout" in blob:
            # We got a definitive answer with no intl variant -> plain layout.
            break
    _layout_dead_keys_cache = result
    LOG_MSG.info("Keyboard layout dead-key escaping: %s", result)
    return result

class STTEngine(IBus.Engine):
    __gtype_name__ = 'STTEngine'

    def __init__(self, bus, object_path):
        if hasattr(IBus.Engine.props, 'has_focus_id'):
            LOG_MSG.info("STTEngine has focus-in-id capabilities")
            # FIXME: hopefully ibus 1.5.28 will have property needs_surrounding_text
            super().__init__(connection=bus.get_connection(),
                             object_path=object_path,
                             has_focus_id=True)
        else:
            LOG_MSG.info("STTEngine has NO focus-in-id capabilities")
            super().__init__(connection=bus.get_connection(),
                             object_path=object_path)

        LOG_MSG.info("STTEngine created %s %i", self, self.get_property("has_focus_id"))

        self._text_processor=STTSegmentProcess()
        self._text_processor.connect("mode-changed", self._mode_changed)
        self._text_processor.connect("need-results", self._need_results)
        self._text_processor.connect("cancel", self._cancel)
        self._text_processor.connect("shortcut", self._shortcut)
        self._text_processor.connect("partial-text", self._partial_formatted_text)
        self._text_processor.connect("final-text", self._final_formatted_text)

        self._left_text=""
        self._left_text_reset=True
        # Last focused IBus client name, for injection diagnostics.
        self._focus_client="?"

        self._preediting=False

        self._settings=Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        self._settings.connect("changed::stop-on-keypress", self._stop_on_key_pressed_changed)
        self._stop_on_key_pressed=False
        self._update_stop_on_key_pressed()

        self._settings.connect("changed::preedit-text", self._on_preedit_text_changed)
        self._settings.connect("changed::format-preedit", self._on_format_preedit_changed)
        self._preedit_text=self._settings.get_boolean("preedit-text")
        self._format_preedit=self._settings.get_boolean("format-preedit")

        self._engine_connected=False
        self._engine=stt_gst_factory_default().new_engine()
        if self._engine.has_model() == False:
            LOG_MSG.error("engine has no valid model")

        # ~1 Hz refresh of the live status labels, only while recording.
        self._status_timer_id=0
        # (prop_key, node_name, label) for each microphone radio entry.
        self._mic_entries=[]
        # Cache of last-pushed status labels. The 1 Hz refresh only calls
        # update_property when a label actually changed -- otherwise the open
        # panel menu gets rebuilt and closes under the cursor.
        self._last_labels={}
        # Mic energy history driving the signal LED (reset each recording start).
        self._energy_seen_ever=False
        self._energy_idle_ticks=0
        # While the panel popup is open we freeze status pushes, since any
        # update_property rebuilds the popup and closes it under the cursor.
        self._menu_visible=False

        self.__prop_list=IBus.PropList()
        self.__prop_list.append(IBus.Property(key="toggle-recording",
                                              label=_("Recognition off"),
                                              icon="audio-input-microphone",
                                              type=IBus.PropType.TOGGLE,
                                              state=IBus.PropState.UNCHECKED,
                                              tooltip=_("Toggle speech recognition")))

        menu_prop_list = IBus.PropList()
        menu_prop_list.append(IBus.Property(key="dictation-mode",
                                            label=_("Dictate"),
                                            type=IBus.PropType.RADIO,
                                            state=IBus.PropState.CHECKED,
                                            tooltip=_("Toggle dictation mode")))
        menu_prop_list.append(IBus.Property(key="literal-mode",
                                            label=_("Dictate (no formatting)"),
                                            type=IBus.PropType.RADIO,
                                            state=IBus.PropState.UNCHECKED,
                                            tooltip=_("Toggle dictation mode with no automatic formatting")))
        menu_prop_list.append(IBus.Property(key="spelling-mode",
                                            label=_("Spell"),
                                            type=IBus.PropType.RADIO,
                                            state=IBus.PropState.UNCHECKED,
                                            tooltip=_("Toggle spelling mode")))
        self.__prop_list.append(IBus.Property(key="mode-menu",
                                              label=_("Recognition mode"),
                                              icon=None,
                                              type=IBus.PropType.MENU,
                                              sensitive=False,
                                              sub_props=menu_prop_list))

        self.__prop_list.append(IBus.Property(key="mic-menu",
                                              label=_("Microphone"),
                                              icon=None,
                                              type=IBus.PropType.MENU,
                                              sensitive=True,
                                              sub_props=self._build_mic_prop_list()))

        self.__prop_list.append(IBus.Property(key="digit-mode",
                                              label=_("Use digits"),
                                              type=IBus.PropType.TOGGLE,
                                              state=IBus.PropState.UNCHECKED,
                                              sensitive=False,
                                              tooltip=_("Toggle the use of digits")))

        self.__prop_list.append(IBus.Property(key="signal-status",
                                              label=_("Microphone: off"),
                                              type=IBus.PropType.NORMAL,
                                              sensitive=False,
                                              tooltip=_("Live microphone input level")))
        self.__prop_list.append(IBus.Property(key="vad-status",
                                              label=_("VAD: —"),
                                              type=IBus.PropType.NORMAL,
                                              sensitive=False,
                                              tooltip=_("Voice activity detection status")))
        self.__prop_list.append(IBus.Property(key="model-status",
                                              label=_("Model: —"),
                                              type=IBus.PropType.NORMAL,
                                              sensitive=False,
                                              tooltip=_("Active speech recognition model")))

        self.__prop_list.append(IBus.Property(key="configuration",
                                              label=_("Settings"),
                                              type=IBus.PropType.NORMAL,
                                              sensitive=True,
                                              tooltip=_("Configure IBus STT")))
        self.__prop_list.append(IBus.Property(key="about",
                                              label=_("About IBus Speech To Text"),
                                              type=IBus.PropType.NORMAL,
                                              sensitive=True,
                                              tooltip=_("Learn more about IBus STT")))

    def __del__(self):
        LOG_MSG.info("STTEngine destroyed %s", self)

    def _disconnect_from_engine(self):
        if self._engine_connected == False:
            LOG_MSG.debug("not connected to engine %s", self)
            return

        LOG_MSG.info("disconnect from engine %s", self)
        self._engine.disconnect_by_func(self._model_changed)
        self._engine.disconnect_by_func(self._state_changed)
        self._engine.disconnect_by_func(self._got_text)
        self._engine.disconnect_by_func(self._got_partial_text)

        self._engine_connected=False

    def _connect_to_engine(self):
        if self._engine_connected == True:
            LOG_MSG.debug("already connected to engine %s", self)
            return

        LOG_MSG.debug("connect to engine %s", self)
        self._engine.connect("model-changed", self._model_changed)
        self._engine.connect("state-changed", self._state_changed)
        self._engine.connect("text", self._got_text)
        self._engine.connect("partial-text", self._got_partial_text)
        self._engine.set_use_partial_results(self._preedit_text)
        self._engine_connected=True

    def do_destroy (self):
        # This method is inherited from IBusObject
        LOG_MSG.info("STTEngine destruction %s", self)

        self._stop_status_timer()

        self._settings.disconnect_by_func(self._stop_on_key_pressed_changed)
        self._settings=None

        self._text_processor.disconnect_by_func(self._mode_changed)
        self._text_processor=None

        # we need to do that since _engine might live on if preloaded
        self._disconnect_from_engine()
        self._engine.release()
        self._engine = None

        # This function needs CHAINING and this way or it leaks
        IBus.Engine.do_destroy(self)

    def _update_stop_on_key_pressed(self):
        self._stop_on_key_pressed=self._settings.get_boolean("stop-on-keypress")

    def _stop_on_key_pressed_changed(self, settings, key):
        self._update_stop_on_key_pressed()

    def _update_preedit_text(self):
        self._preedit_text=self._settings.get_boolean("preedit-text")
        self._engine.set_use_partial_results(self._preedit_text)

    def _on_preedit_text_changed(self, settings, key):
        self._update_preedit_text()

    def _on_format_preedit_changed(self, settings, key):
        self._format_preedit=self._settings.get_boolean("format-preedit")

    def _is_recognizing(self):
        # The mic keeps capturing (for the live meter) whenever the engine is
        # enabled; "recognising" is the separate transcription state shown on
        # the toggle. Fall back to is_running() for backends without the split.
        if hasattr(self._engine, "is_recognizing"):
            return self._engine.is_recognizing()
        return self._engine.is_running()

    def _set_recognizing(self, active):
        if hasattr(self._engine, "set_recognizing"):
            self._engine.set_recognizing(active)
        elif active:
            self._engine.run()
        else:
            self._engine.stop()

    def _update_state(self):
        if self._is_recognizing() == True:
            button_state=IBus.PropState.CHECKED
            button_label=IBus.Text(_("Recognition on"))
        else:
            button_state=IBus.PropState.UNCHECKED
            button_label=IBus.Text(_("Recognition off"))

        is_dictation = bool(self._text_processor.mode == STTParseModes.DICTATION)
        is_spelling = bool(self._text_processor.mode == STTParseModes.SPELLING)
        is_literal = bool(self._text_processor.mode == STTParseModes.LITERAL)

        prop=IBus.Property(key="toggle-recording",
                           label=button_label,
                           icon="audio-input-microphone",
                           type=IBus.PropType.TOGGLE,
                           state=button_state,
                           sensitive=self._engine.has_model(),
                           tooltip=_("Toggle speech recognition"))
        self.update_property(prop)

        prop = IBus.Property(key="mode-menu",
                             label=_("Recognition modes"),
                             icon=None,
                             type=IBus.PropType.MENU,
                             sensitive=(button_state == IBus.PropState.CHECKED))
        self.update_property(prop)

        prop=IBus.Property(key="dictation-mode",
                           label=_("Dictate"),
                           type=IBus.PropType.RADIO,
                           state=IBus.PropState.CHECKED if is_dictation else IBus.PropState.UNCHECKED,
                           sensitive=self._text_processor.can_dictate,
                           tooltip=_("Toggle dictation mode"))
        self.update_property(prop)

        prop=IBus.Property(key="literal-mode",
                           label=_("Dictate (no formatting)"),
                           type=IBus.PropType.RADIO,
                           state=IBus.PropState.CHECKED if is_literal else IBus.PropState.UNCHECKED,
                           tooltip=_("Toggle dictation mode with no automatic formatting"))
        self.update_property(prop)

        prop=IBus.Property(key="spelling-mode",
                           label=_("Spell"),
                           type=IBus.PropType.RADIO,
                           state=IBus.PropState.CHECKED if is_spelling else IBus.PropState.UNCHECKED,
                           sensitive=self._text_processor.can_spell,
                           tooltip=_("Toggle spelling mode"))
        self.update_property(prop)

        use_digits = self._text_processor.use_digits
        prop=IBus.Property(key="digit-mode",
                           label=_("Use digits"),
                           type=IBus.PropType.TOGGLE,
                           state=IBus.PropState.CHECKED if use_digits else IBus.PropState.UNCHECKED,
                           sensitive=self._text_processor.can_use_digits,
                           tooltip=_("Toggle the use of digits"))
        self.update_property(prop)

        if self._engine.is_running():
            self._start_status_timer()
        else:
            self._stop_status_timer()
        self._update_status_labels()

    def _state_changed(self, engine):
        # Be careful that we don't call this too often
        self._update_state()

    def _model_changed(self, engine):
        LOG_MSG.debug("engine model has changed")
        if self._engine.has_model() == False:
            LOG_MSG.error("engine has no model")

        self._update_state()

    def _mode_changed(self, text_processor):
        self._update_state()

    def _build_mic_prop_list(self):
        # Radio submenu: "Follow system default" plus each detected input device.
        # Only the Whisper backend exposes device selection.
        if hasattr(self._engine, "list_audio_sources"):
            sources = self._engine.list_audio_sources()
            current = self._engine.get_audio_device()
        else:
            sources = []
            current = ""

        self._mic_entries = []
        prop_list = IBus.PropList()

        default_label = _("Follow system default")
        prop_list.append(IBus.Property(key="mic-default",
                                       label=default_label,
                                       type=IBus.PropType.RADIO,
                                       state=IBus.PropState.UNCHECKED if current else IBus.PropState.CHECKED,
                                       tooltip=_("Use the system default input device")))
        self._mic_entries.append(("mic-default", "", default_label))

        for node_name, desc in sources:
            key = "mic:" + node_name
            prop_list.append(IBus.Property(key=key,
                                           label=desc,
                                           type=IBus.PropType.RADIO,
                                           state=IBus.PropState.CHECKED if current == node_name else IBus.PropState.UNCHECKED,
                                           tooltip=_("Capture speech from this device")))
            self._mic_entries.append((key, node_name, desc))

        return prop_list

    def _update_mic_state(self, current):
        for key, node, label in self._mic_entries:
            checked = (node == current)
            prop = IBus.Property(key=key,
                                 label=IBus.Text(label),
                                 type=IBus.PropType.RADIO,
                                 state=IBus.PropState.CHECKED if checked else IBus.PropState.UNCHECKED)
            self.update_property(prop)

    def _format_signal(self, level, running):
        # Fixed-width output (10-cell bar + 7-char field) so the trailing LED
        # never shifts horizontally, and so a resting mic produces a byte-stable
        # string that the cache can suppress (keeping the menu open).
        width = 10
        if not running:
            bar = "─" * width
            field = _("off")
        elif level < _ENERGY_THRESHOLD:
            bar = "░" * width
            field = _("silent")
        else:
            import math
            db = 20.0 * math.log10(min(1.0, level))
            frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
            filled = int(round(frac * width))
            bar = "█" * filled + "░" * (width - filled)
            field = "%4.0f dB" % db
        return "🎤 %s %-7s" % (bar, field)

    def _set_label(self, key, label):
        # Only push when the text changed, so an idle widget keeps the menu open.
        if self._last_labels.get(key) == label:
            return
        self._last_labels[key] = label
        LOG_MSG.debug("widget push %s = %r", key, label)
        self.update_property(IBus.Property(key=key,
                                           label=IBus.Text(label),
                                           type=IBus.PropType.NORMAL,
                                           sensitive=False))

    def _update_status_labels(self):
        running = self._engine.is_running()

        # Microphone LED is the only live-activity indicator: red when not
        # recording or no signal seen yet, green while audio is flowing,
        # orange once it has gone quiet for a while.
        level = self._engine.get_audio_level() if hasattr(self._engine, "get_audio_level") else 0.0
        if running:
            if level > _ENERGY_THRESHOLD:
                self._energy_idle_ticks = 0
                self._energy_seen_ever = True
            else:
                self._energy_idle_ticks += 1

        if not running or not self._energy_seen_ever:
            mic_led = _LED_RED
        elif self._energy_idle_ticks <= _ENERGY_GREEN_TICKS:
            mic_led = _LED_GREEN
        else:
            mic_led = _LED_ORANGE
        self._set_label("signal-status",
                        "%s  %s" % (mic_led, self._format_signal(level, running)))

        # VAD LED reflects backend availability, not whether we are recording:
        # green = backend loaded and ready, red = unavailable. The text still
        # shows the live speaking/idle state.
        if hasattr(self._engine, "get_vad_status"):
            backend, in_speech = self._engine.get_vad_status()
        else:
            backend, in_speech = ("", False)
        if backend:
            vad_led = _LED_GREEN
            state = _("speaking") if (running and in_speech) else _("idle")
            vad_text = "%s · %-8s" % (backend, state)
        else:
            vad_led = _LED_RED
            vad_text = _("VAD: unavailable")
        self._set_label("vad-status", "%s  %s" % (vad_led, vad_text))

        # Model LED reflects whether a model is loaded/ready, independent of
        # recording: green = model up, red = none / failed to load.
        name = self._engine.get_model_name() if hasattr(self._engine, "get_model_name") else None
        if name:
            model_led = _LED_GREEN
            model_text = _("Model: %s") % name
        else:
            model_led = _LED_RED
            model_text = _("Model: %s") % _("none")
        self._set_label("model-status", "%s  %s" % (model_led, model_text))

    def _on_status_tick(self):
        if not self._engine.is_running():
            self._status_timer_id = 0
            self._update_status_labels()
            return False
        # Freeze pushes while the panel popup is open, otherwise each
        # update_property rebuilds it and closes it under the cursor.
        if not self._menu_visible:
            self._update_status_labels()
        return True

    def do_property_show(self, prop_name):
        LOG_MSG.debug("property show %s", prop_name)
        self._menu_visible = True

    def do_property_hide(self, prop_name):
        LOG_MSG.debug("property hide %s", prop_name)
        self._menu_visible = False

    def _start_status_timer(self):
        if self._status_timer_id == 0:
            # Fresh recording session: restart the mic energy history.
            self._energy_seen_ever = False
            self._energy_idle_ticks = 0
            self._status_timer_id = GLib.timeout_add(1000, self._on_status_tick)

    def _stop_status_timer(self):
        if self._status_timer_id != 0:
            GLib.source_remove(self._status_timer_id)
            self._status_timer_id = 0

    def do_enable(self):
        LOG_MSG.info('enable %s', self)

        # Necessary to indicate we'll use surrounding text
        # FIXME: hopefully ibus 1.5.28 will have property needs_surrounding_text
        (ibus_text, cursor_pos, anchor_pos)=self.get_surrounding_text()
        self._set_left_text(ibus_text, cursor_pos)

        # Allow for update of surrounding text by do_set_surrounding_text()
        # though it is very unlikely it will be called at the moment.
        self._left_text_reset=True

        self._connect_to_engine()
        if self._engine.has_model() == False:
            # There is something wrong with our model, display config dialog
            subprocess.Popen([os.path.join(stt_utils_get_libexec(), "ibus-setup-stt")])
            return

        active_on_start = self._settings.get_boolean("active-on-start")
        LOG_MSG.info("engine enabled %s (active_on_start=%s)", self, active_on_start)
        # Start capturing immediately so the mic meter is live; recognition
        # (transcription) follows the active-on-start preference.
        self._engine.run()
        self._set_recognizing(active_on_start == True)
        self._update_state()

    def do_disable(self):
        LOG_MSG.info('disable %s', self)
        self._engine.stop()
        self._disconnect_from_engine()

    def do_focus_in(self):
        LOG_MSG.debug("focus in")
        self.do_focus_in_id("", "")

    def do_focus_in_id(self, object_path, client):
        LOG_MSG.debug("focus in id %s %s", object_path, client)
        # Remember the focused client so injection logging can show where text
        # was typed (helps diagnose dropped/misrouted keystrokes).
        self._focus_client = client or object_path or "?"
        # Safety: never stay frozen if a property-hide was missed.
        self._menu_visible = False
        # FIXME: hopefully ibus 1.5.28 will have property needs_surrounding_text
        (ibus_text, cursor_pos, anchor_pos)=self.get_surrounding_text()
        self.register_properties(self.__prop_list)
        self._update_state()

        # Shortcut depends on the client, only IBus gtk2/gtk3 clients allow it
        if client.startswith("gtk3-im:") or client.startswith("gtk2-im:"):
            self._text_processor.supports_shortcuts=True
        else:
            self._text_processor.supports_shortcuts=False

        # With recent gtk versions the "focus-in" is not always preceded by a
        # "reset" signal. Mainly when switching to a gtk4 window with no text.
        # Reset the left text just to make sure as it is not a time-consuming
        # function.
        self._reset()

    def do_focus_out(self):
        LOG_MSG.debug("focus out")
        self.do_focus_out_id("")

    def do_focus_out_id(self, object_path):
        LOG_MSG.debug("focus out id")
        self._reset()

    def do_reset(self):
        LOG_MSG.debug("do reset")
        self._reset()

    def do_property_activate(self, prop_name, state):
        # Reminder: no need to call final_results() since do_reset()
        # do_focus_out() is called.
        self._menu_visible = False

        if prop_name == 'toggle-recording':
            # Toggle transcription only; the mic keeps capturing for the meter.
            self._set_recognizing(bool(state) == True)
            self._update_state()
        elif prop_name == 'dictation-mode':
            if state == True:
                self._text_processor.mode = STTParseModes.DICTATION
        elif prop_name == 'spelling-mode':
            if state == True:
                self._text_processor.mode = STTParseModes.SPELLING
        elif prop_name == 'literal-mode':
            if state == True:
                self._text_processor.mode = STTParseModes.LITERAL
        elif prop_name == 'digit-mode':
            self._text_processor.use_digits = bool(state)
        elif prop_name == 'mic-default' or prop_name.startswith('mic:'):
            if bool(state) == True and hasattr(self._engine, 'set_audio_device'):
                device = "" if prop_name == 'mic-default' else prop_name[len('mic:'):]
                self._engine.set_audio_device(device)
                self._update_mic_state(device)
        elif prop_name == 'configuration':
            subprocess.Popen([os.path.join(stt_utils_get_libexec(), "ibus-setup-stt")])
        elif prop_name == 'about':
            dialog = Adw.AboutWindow(application_name=_("IBus Speech To Text"),
                            title=_("About IBus Speech To Text"),
                            application_icon="user-available-symbolic",
                            version=stt_utils_get_version(),
                            copyright="Copyright © 2022 Philippe Rouquier",
                            comments=_("What you say is always write."),
                            website="https://github.com/PhilippeRo/IBus-Speech-To-Text",
                            issue_url="https://github.com/PhilippeRo/IBus-Speech-To-Text/issues",
                            license_type=Gtk.License.GPL_3_0,
                            translator_credits=_("translator-credits"))
            dialog.present()

    def _need_results(self, text_process):
        if self._preediting == True:
            self._engine.get_results()

    def _cancel(self, text_process, cancel_size):
        # Handle potential pending cancellations
        if (self.client_capabilities & IBus.Capabilite.SURROUNDING_TEXT) == 0:
            LOG_MSG.debug("client application has no surrounding text capability")

        self.delete_surrounding_text(-cancel_size, cancel_size)

        # Keep our left text updated
        text_len=len(self._left_text)
        text_len=text_len-cancel_size if cancel_size <= text_len else text_len
        self._left_text=self._left_text[:text_len]

    def _shortcut(self, text_process, keyval, modifiers):
        if self._preediting == True:
            # Don't call this if there was no preediting before
            self.update_preedit_text_with_mode(IBus.Text.new_from_string(""),
                                               0,
                                               True,
                                               IBus.PreeditFocusMode.CLEAR)
            self._preediting=False

        self.forward_key_event(keyval, 0, modifiers)

    def _add_preedit_text(self, utterance):
        # Note: we accept "" (in case we need to remove previous partial text)
        ibus_text=IBus.Text.new_from_string(utterance)
        self.update_preedit_text_with_mode(ibus_text,
                                           0,
                                           True,
                                           IBus.PreeditFocusMode.CLEAR)
        self._preediting=True

    def _partial_formatted_text(self, text_process, utterance):
        self._add_preedit_text(utterance)

    def _inject_text(self, text):
        """Send committed text to the focused app via the right backend.

        Both backends TYPE the text directly (no clipboard): clipboard paste
        clobbers the user's clipboard and Ctrl+V is "quoted insert" in
        terminals (silently drops text). Both type *through* the active layout,
        so us-intl dead keys (' " ` ~ ^) are escaped via _escape_dead_keys()
        when _layout_uses_dead_keys() says the layout needs it.

        Wayland (ydotool): injects through the kernel uinput layer.
        --key-delay 0 --key-hold 0 removes ydotool's per-key animation (a long
        sentence would otherwise take 1-2s to appear); with --file - the 1.x
        client disables its own escape processing, so _escape_dead_keys stays
        authoritative. Requires the ydotoold user daemon
        (scripts/install-ydotoold.sh). wtype is out: GNOME/Mutter lacks the
        virtual-keyboard Wayland protocol.

        X11 (xdotool type): injects through XTEST -- instant, no daemon, no
        clipboard. --delay 0 disables xdotool's inter-key delay.

        Backend AND dead-key escaping are decided at runtime, so the same code
        runs unchanged on the Wayland desktop and the X11 laptop.
        """
        backend = _detect_display_server()
        out_text = _escape_dead_keys(text) if _layout_uses_dead_keys() else text
        # Diagnostics: dropped/misrouted keystrokes (a whole segment silently not
        # landing) leave NO error -- the subprocess returns 0 but the compositor
        # dropped the keys. Log enough to correlate a future drop: char/byte
        # count, elapsed time, return code, stderr, and the focused client.
        n_chars = len(out_text)
        n_bytes = len(out_text.encode("utf-8"))
        client = getattr(self, "_focus_client", "?")
        LOG_MSG.info("Inject START backend=%s chars=%d bytes=%d client=%s text=%r",
                     backend, n_chars, n_bytes, client,
                     out_text[:80] + ("…" if n_chars > 80 else ""))
        t0 = time.monotonic()
        try:
            if backend == "wayland":
                proc = subprocess.run(
                    ["ydotool", "type", "--key-delay", "0",
                     "--key-hold", "0", "--file", "-"],
                    input=out_text.encode("utf-8"), timeout=10,
                    capture_output=True)
            else:
                proc = subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "0",
                     "--", out_text], timeout=10, capture_output=True)
            elapsed = time.monotonic() - t0
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            rate = (n_chars / elapsed) if elapsed > 0 else 0.0
            level = LOG_MSG.warning if (proc.returncode != 0 or stderr) else LOG_MSG.info
            level("Inject DONE backend=%s rc=%d chars=%d in %.3fs (%.0f ch/s)%s",
                  backend, proc.returncode, n_chars, elapsed, rate,
                  (" stderr=%r" % stderr) if stderr else "")
        except subprocess.TimeoutExpired:
            LOG_MSG.error("Inject TIMEOUT backend=%s after %.1fs chars=%d -- "
                          "keystrokes likely partially dropped", backend,
                          time.monotonic() - t0, n_chars)
        except FileNotFoundError as e:
            LOG_MSG.error(
                "Text-injection tool missing for %s backend (%s). "
                "On Wayland run scripts/install-ydotoold.sh; on X11 install "
                "xdotool. Override with STT_INJECT_BACKEND.",
                backend, e)
        except Exception as e:
            LOG_MSG.error("Text injection failed (%s backend): %s", backend, e)

    def _final_formatted_text(self, text_process, utterance):
        if self._preediting == True:
            # Don't call this if there was no preediting before
            self.update_preedit_text_with_mode(IBus.Text.new_from_string(""),
                                               0,
                                               True,
                                               IBus.PreeditFocusMode.CLEAR)
            self._preediting=False

        # Note : there could be text to write even after cancellation ("cancel
        # write this").
        if utterance != "":
            paste_text = utterance.lstrip(' ')
            if paste_text != utterance:
                paste_text = paste_text + ' '
            # Keep consecutive sentences from gluing together: a sentence that
            # ends with terminal punctuation gets a trailing space so the next
            # utterance is not glued onto it.
            if paste_text and paste_text[-1] in '.?!':
                paste_text = paste_text + ' '
            # Inject via the backend appropriate to the running display server
            # (ydotool on Wayland, xdotool+clipboard on X11). See _inject_text.
            self._inject_text(paste_text)
            self._left_text+=paste_text
            self._left_text_reset=False
            LOG_MSG.debug("current left text (after commit) (%s)", self._left_text)

    def _got_partial_text(self, engine, utterance):
        if self._format_preedit == True:
            self._text_processor.utterance_process_begin(utterance, self._left_text)
        else:
            self._add_preedit_text(utterance)

    def _got_text(self, engine, utterance):
        self._text_processor.utterance_process_end(utterance, self._left_text)

    def _reset(self):
        # Reminder don't call final_results() or when the window is focused out,
        # the new window will get the final result.
        # Let the partial text be committed instead ? But in this case we need
        # to reset the current analysis to avoid the new window to have the text
        # In the current situation, the new window continues the voice
        # recognition as if nothing has happened.

        # Reset left text since the window might have changed
        self._left_text=""
        self._left_text_reset=True
        self._text_processor.reset()

        # Note: we used to do this in the hope it would force update but there
        # is a potential problem here: select text and click -> the selected
        # text is deleted !!
        # if self._engine.is_running() == True:
        #     self.commit_text(IBus.Text.new_from_string(""))

    def do_process_key_event(self, keyval, keycode, state):
        if (state & IBus.ModifierType.RELEASE_MASK) != 0:
            if self._stop_on_key_pressed == True:
                # Stop transcribing on keypress, but keep capturing (meter live).
                self._set_recognizing(False)
                self._update_state()
        else:
            # Any keystroke should stop a potential ongoing processing.
            # wait=False is REQUIRED here: this runs on the IBus main/UI thread,
            # and a blocking finalize (_process_queue.join) held the main loop
            # until decoding finished, which froze the keyboard -- keystrokes
            # were not delivered while a decode was in flight, "fixed" only once
            # speaking drained the queue. The final text is still emitted
            # asynchronously by the worker.
            if self._text_processor.is_processing() == True:
                self._engine.get_final_results(wait=False)

            # Usually there is a "set-surrounding-text" event after a key press.
            # So get ready for the update (though we keep our current one if
            # none comes). This is in case the key press is an arrow that moved
            # the cursor. Instead of tracking this kind of strokes, let IBus
            # tell us how the surrounding text changed.
            self._left_text_reset = True

        # Let the keystroke be propagated
        return False

    def _set_left_text(self, ibus_text, cursor_pos):
        # Each text commit or preedit may reliably (but not always for example
        # gtk3 and gtk4) sets the surrounding text. Problem is, preedit text
        # is included.
        # Note: at one point only bytes were used but commit in gtk change that
        # text_bytes=ibus_text.get_text().encode()
        # self._left_text=text_bytes[:cursor_pos].decode("utf-8")
        self._left_text=ibus_text.get_text()[:cursor_pos]
        LOG_MSG.debug("left text changed (%s) (original text=%s / cursor pos=%i)",
                      self._left_text, ibus_text.get_text(), cursor_pos)

        # Reminder we do not care about the context on the right, it is up to
        # the user to add a potential missing whitespace.

    def do_set_surrounding_text(self, ibus_text, cursor_pos, anchor_pos):
        if self._left_text_reset == True:
            self._set_left_text(ibus_text, cursor_pos)

        # We need to chain this function if we want get_surrounding_text to work
        IBus.Engine.do_set_surrounding_text(self, ibus_text, cursor_pos, anchor_pos)
