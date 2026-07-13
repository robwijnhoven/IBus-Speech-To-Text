"""Self-check for the stuck-modifier commit-deferral logic (sttengine.py).

Not a framework test -- run directly:  python3 test_modifier_defer.py
Fails loudly (AssertionError) if the defer/flush state machine breaks.

Reproduces the exact bit arithmetic of do_process_key_event + _final_formatted_text
without needing a live IBus, so a bad mask/keyval never ships as "no valid model".
"""
import gi
gi.require_version("IBus", "1.0")
from gi.repository import IBus

M = IBus.ModifierType
_MOD_MASK = M.SHIFT_MASK | M.CONTROL_MASK | M.MOD1_MASK | M.SUPER_MASK
_MOD_KEYS = (IBus.KEY_Shift_L, IBus.KEY_Shift_R, IBus.KEY_Control_L,
             IBus.KEY_Control_R, IBus.KEY_Alt_L, IBus.KEY_Alt_R,
             IBus.KEY_Super_L, IBus.KEY_Super_R, IBus.KEY_Caps_Lock)


class Fake:
    def __init__(self):
        self.mods_held = False
        self.pending = None
        self.committed = []

    def key(self, keyval, state):
        is_release = (state & M.RELEASE_MASK) != 0
        mods = state & _MOD_MASK
        if is_release and keyval in _MOD_KEYS:
            mods = 0
        self.mods_held = (mods != 0)
        if not self.mods_held and self.pending is not None:
            self.committed.append(self.pending)
            self.pending = None

    def finalize(self, text):
        if self.mods_held:
            self.pending = (self.pending or "") + text
        else:
            self.committed.append(text)


def demo():
    # 1. The reported bug: Shift down, Tab, talk -> commit must be deferred,
    #    then flushed on Shift release. Nothing commits mid-hold.
    f = Fake()
    f.key(IBus.KEY_Shift_L, M.SHIFT_MASK)          # Shift pressed (state pre-apply=0? IBus sends SHIFT on the press of a chord's 2nd key)
    f.key(IBus.KEY_Tab, M.SHIFT_MASK)              # Tab while Shift held
    f.finalize("hello world")                      # whisper finalizes mid-hold
    assert f.committed == [], f.committed           # nothing committed yet
    assert f.pending == "hello world"
    f.key(IBus.KEY_Shift_L, M.SHIFT_MASK | M.RELEASE_MASK)  # Shift released
    assert f.committed == ["hello world"], f.committed      # flushed exactly once
    assert f.pending is None

    # 2. No modifier held -> immediate commit, no deferral.
    f = Fake()
    f.key(IBus.KEY_a, 0)
    f.finalize("plain text")
    assert f.committed == ["plain text"], f.committed
    assert f.pending is None

    # 3. Caps Lock ON must NOT wedge commits (LOCK_MASK excluded from _MOD_MASK).
    f = Fake()
    f.key(IBus.KEY_a, M.LOCK_MASK)                 # a keystroke while Caps latched on
    assert f.mods_held is False
    f.finalize("still commits")
    assert f.committed == ["still commits"], f.committed

    # 4. Multiple finalizes during one hold accumulate, flush together.
    f = Fake()
    f.key(IBus.KEY_Control_L, M.CONTROL_MASK)
    f.finalize("one ")
    f.finalize("two")
    assert f.committed == []
    f.key(IBus.KEY_Control_L, M.CONTROL_MASK | M.RELEASE_MASK)
    assert f.committed == ["one two"], f.committed

    print("all modifier-defer checks passed")


if __name__ == "__main__":
    demo()
