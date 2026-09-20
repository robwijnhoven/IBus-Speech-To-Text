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
import subprocess
import logging
import unicodedata

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

# Dead keys, for layouts like us+intl that have them.
#
# IBus provides compose/dead-key handling through IBusEngineSimple, which the
# built-in layout engines extend. An engine that does not extend it -- this one
# -- gets none, so a dead_acute reaches the client as a raw keysym and is
# dropped: on us+intl that makes ' " ` ~ ^ produce nothing at all, neither the
# accented character nor the bare one, for as long as this engine is selected.
#
# keyval -> (combining mark, the literal the key types on its own). The literal
# matters: dead_acute + space is the standard way to type a plain apostrophe,
# and without it a quote is unreachable.
_DEAD_KEYS = {
    IBus.KEY_dead_acute:      ("\u0301", "'"),
    IBus.KEY_dead_grave:      ("\u0300", "`"),
    IBus.KEY_dead_diaeresis:  ("\u0308", '"'),
    IBus.KEY_dead_tilde:      ("\u0303", "~"),
    IBus.KEY_dead_circumflex: ("\u0302", "^"),
    IBus.KEY_dead_cedilla:    ("\u0327", ","),
    IBus.KEY_dead_caron:      ("\u030C", "\u02C7"),
    IBus.KEY_dead_breve:      ("\u0306", "\u02D8"),
    IBus.KEY_dead_macron:     ("\u0304", "\u00AF"),
    IBus.KEY_dead_ogonek:     ("\u0328", "\u02DB"),
    IBus.KEY_dead_abovering:  ("\u030A", "\u02DA"),
    IBus.KEY_dead_doubleacute:("\u030B", "\u02DD"),
    IBus.KEY_dead_abovedot:   ("\u0307", "\u02D9"),
}


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

        self._preediting=False
        self._pending_dead=None   # combining mark from a dead key, awaiting its base

        self._settings=Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        self._settings.connect("changed::stop-on-keypress", self._stop_on_key_pressed_changed)
        self._stop_on_key_pressed=False
        self._update_stop_on_key_pressed()

        self._settings.connect("changed::preedit-text", self._on_preedit_text_changed)
        self._settings.connect("changed::format-preedit", self._on_format_preedit_changed)
        self._preedit_text=self._settings.get_boolean("preedit-text")
        self._format_preedit=self._settings.get_boolean("format-preedit")

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

        self.__prop_list.append(IBus.Property(key="digit-mode",
                                              label=_("Use digits"),
                                              type=IBus.PropType.TOGGLE,
                                              state=IBus.PropState.UNCHECKED,
                                              sensitive=False,
                                              tooltip=_("Toggle the use of digits")))

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

        self._engine_connected=False
        self._engine=stt_gst_factory_default().new_engine()
        if self._engine.has_model() == False:
            LOG_MSG.error("engine has no valid model")

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

    def _update_state(self):
        if self._engine.is_running() == True:
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
        if active_on_start == True:
            self._engine.run()
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

        if prop_name == 'toggle-recording':
            # State will be updated by the engine
            if bool(state) == True:
                self._engine.run()
            else:
                self._engine.stop()
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
            self.commit_text(IBus.Text.new_from_string(utterance))
            self._left_text+=utterance
            self._left_text_reset=False
            LOG_MSG.debug("current left text (after commit) (%s)", self._left_text)

    def _got_partial_text(self, engine, utterance):
        if (self.client_capabilities & IBus.Capabilite.PREEDIT_TEXT) == 0:
            LOG_MSG.debug("client has no Preedit capability")
            return

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

    def _compose_dead_key(self, keyval, mods):
        """Combine a pending dead key with this keystroke.

        Returns True to swallow the key, False to let it through, or None when
        dead keys are not involved and normal handling should continue.
        """
        # Never interfere with Ctrl/Alt/Super combos.
        if mods & ~IBus.ModifierType.SHIFT_MASK:
            self._pending_dead = None
            return None

        entry = _DEAD_KEYS.get(keyval)
        if entry is not None:
            if self._pending_dead is not None:
                # Same dead key twice types the bare mark, and two different
                # ones cannot combine: emit the pending literal, keep this one.
                prev_literal = self._pending_dead[1]
                self._pending_dead = entry
                self.commit_text(IBus.Text.new_from_string(prev_literal))
                return True
            self._pending_dead = entry
            return True      # swallow: nothing shows until the base arrives

        if self._pending_dead is None:
            return None      # ordinary key, nothing pending

        combining, literal = self._pending_dead
        self._pending_dead = None

        base = IBus.keyval_to_unicode(keyval)
        if not base:
            # Arrow, Escape, Backspace...: the accent is abandoned, but emit
            # its literal so the keystroke is not silently lost.
            self.commit_text(IBus.Text.new_from_string(literal))
            return None

        if base == " ":
            # The canonical "I meant the plain character" escape.
            self.commit_text(IBus.Text.new_from_string(literal))
            return True

        composed = unicodedata.normalize("NFC", base + combining)
        if len(composed) != 1:
            # No precomposed form (dead_acute + q): type both, as a real dead
            # key does, rather than dropping the accent silently.
            composed = literal + base

        self.commit_text(IBus.Text.new_from_string(composed))
        return True

    def do_process_key_event(self, keyval, keycode, state):
        # Dead-key compose must run first: without it the keysym is forwarded
        # raw and the client drops it, so ' + e yields nothing.
        if (state & IBus.ModifierType.RELEASE_MASK) == 0:
            handled = self._compose_dead_key(
                keyval, state & ~IBus.ModifierType.RELEASE_MASK)
            if handled is not None:
                return handled

        if (state & IBus.ModifierType.RELEASE_MASK) != 0:
            if self._stop_on_key_pressed == True:
                self._engine.stop()
                self._update_state()
        else:
            # Any keystroke should stop a potential ongoing processing
            if self._text_processor.is_processing() == True:
                self._engine.get_final_results()

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
