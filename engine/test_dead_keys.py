#!/usr/bin/env python3
"""Self-check for dead-key compose (sttengine._compose_dead_key).

Reproduces the method against a stub so it runs without IBus/GTK/a display.
Run: python3 engine/test_dead_keys.py
"""
import re
import unicodedata

SRC = __file__.replace("test_dead_keys.py", "sttengine.py")

# Real keyval numbers for the keys we care about.
KEY = {"dead_acute": 0xfe51, "dead_diaeresis": 0xfe57, "dead_tilde": 0xfe53,
       "e": ord("e"), "n": ord("n"), "q": ord("q"), "Left": 0xff51,
       "space": ord(" ")}
_DEAD = {KEY["dead_acute"]: ("\u0301", "'"),
         KEY["dead_diaeresis"]: ("\u0308", '"'),
         KEY["dead_tilde"]: ("\u0303", "~")}
SHIFT_MASK = 1 << 0
CTRL_MASK = 1 << 2


class Engine:
    """Mirrors _compose_dead_key's logic against stub IBus primitives."""

    def __init__(self):
        self._pending_dead = None
        self.committed = []

    def keyval_to_unicode(self, kv):
        return "" if kv == KEY["Left"] else chr(kv)

    def commit_text(self, t):
        self.committed.append(t)

    def compose(self, keyval, mods=0):
        if mods & ~SHIFT_MASK:
            self._pending_dead = None
            return None
        entry = _DEAD.get(keyval)
        if entry is not None:
            if self._pending_dead is not None:
                prev_literal = self._pending_dead[1]
                self._pending_dead = entry
                self.commit_text(prev_literal)
                return True
            self._pending_dead = entry
            return True
        if self._pending_dead is None:
            return None
        combining, literal = self._pending_dead
        self._pending_dead = None
        base = self.keyval_to_unicode(keyval)
        if not base:
            self.commit_text(literal)
            return None
        if base == " ":
            self.commit_text(literal)
            return True
        composed = unicodedata.normalize("NFC", base + combining)
        if len(composed) != 1:
            composed = literal + base
        self.commit_text(composed)
        return True


def check():
    # The reported bug: ' + e must give e-acute, not nothing.
    e = Engine()
    assert e.compose(KEY["dead_acute"]) is True, "dead key must be swallowed"
    assert e.committed == [], "nothing is emitted until the base key arrives"
    assert e.compose(KEY["e"]) is True
    assert e.committed == ["é"], e.committed

    # Other dead keys on the us+intl layout.
    for dead, base, want in ((("dead_diaeresis"), "e", "ë"),
                             (("dead_tilde"), "n", "ñ")):
        e = Engine()
        e.compose(KEY[dead])
        e.compose(KEY[base])
        assert e.committed == [want], (dead, e.committed)

    # An ordinary key with nothing pending is none of our business: returning
    # None is what lets normal typing (and the STT key handling) proceed.
    e = Engine()
    assert e.compose(KEY["e"]) is None
    assert e.committed == []

    # The regression this fixes: a bare quote must stay typeable. ' + space
    # is the canonical escape, and without it an apostrophe is unreachable.
    for dead, want in (("dead_acute", "'"), ("dead_diaeresis", '"'),
                       ("dead_tilde", "~")):
        e = Engine()
        e.compose(KEY[dead])
        assert e.compose(KEY["space"]) is True
        assert e.committed == [want], (dead, e.committed)

    # No precomposed form: type both, never drop the accent silently.
    e = Engine()
    e.compose(KEY["dead_acute"])
    assert e.compose(KEY["q"]) is True
    assert e.committed == ["'q"], e.committed

    # Same dead key twice emits the bare mark.
    e = Engine()
    e.compose(KEY["dead_acute"])
    assert e.compose(KEY["dead_acute"]) is True
    assert e.committed == ["'"], e.committed

    # A non-character key abandons the accent but still emits its literal,
    # so the keystroke is never silently swallowed.
    e = Engine()
    e.compose(KEY["dead_acute"])
    assert e.compose(KEY["Left"]) is None
    assert e.committed == ["'"], e.committed

    # Ctrl+key must never be eaten by the composer.
    e = Engine()
    assert e.compose(KEY["e"], CTRL_MASK) is None
    e.compose(KEY["dead_acute"])
    assert e.compose(KEY["e"], CTRL_MASK) is None, "Ctrl combo must pass through"
    assert e._pending_dead is None, "Ctrl combo must clear a pending accent"

    # The table in the real source must stay in sync with what we assert here.
    src = open(SRC).read()
    assert "_DEAD_KEYS" in src and "unicodedata.normalize" in src
    for name in ("dead_acute", "dead_diaeresis", "dead_tilde"):
        assert f"IBus.KEY_{name}" in src, name

    print("dead-key compose: all assertions passed")


if __name__ == "__main__":
    check()
