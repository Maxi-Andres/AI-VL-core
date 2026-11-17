#!/usr/bin/env python3
"""
keep_check.py — Does the RUNNING model tell an order to keep something from everything else?

keep_common.wants_kept asks the VLM server whether a message asks the robot to keep a picture,
a note or a fact it sees. Why it exists (2026-11-17, during a demo): typed "que estas viendo?"
was taken for "save the picture" and the robot answered "Listo, guardé la imagen en mi memoria."
instead of saying what it saw; "que ves?" became "remember what you see". Typed messages have
no accents, capitals or question marks, and the prompt's own example "¿Guardá lo que estás
viendo?" pulled them in. 25/32 before the fix, 32/32 after.

Runs the code on disk against the llama-server the app runs (default :8081), so a prompt change
is measured before the app is restarted:

    .venv/bin/python src/keep_check.py [--url http://127.0.0.1:8081]
"""
import argparse
import sys

import keep_common

# (message, what it asks to keep: "picture" | "note" | "seen" | None). Questions about what it
# sees, typed the way people type, next to orders typed the same way.
CASES = [
    ("que estas viendo?", None), ("que estas viendo", None), ("¿Qué estás viendo?", None),
    ("que ves?", None), ("que ves", None), ("Qué ves", None), ("que hay ahi", None),
    ("que tenes adelante?", None), ("decime que ves", None), ("describí lo que ves", None),
    ("que es eso?", None), ("cuantas personas ves?", None), ("que dice el cartel", None),
    ("que guardaste?", None), ("te acordas del codigo?", None), ("mirá la mesa", None),
    ("hola", None), ("como estas?", None), ("que color es la silla", None),
    ("what are you looking at?", None), ("what do you see", None),
    ("sacá una foto", "picture"), ("saca una foto", "picture"),
    ("guardá lo que estás viendo", "picture"), ("guarda esta imagen", "picture"),
    ("sacale una foto a la puerta", "picture"), ("¿podés sacar una foto?", "picture"),
    ("take a picture", "picture"),
    ("acordate que el codigo es 4455", "note"), ("anotá que mañana viene Juan", "note"),
    ("acordate de la temperatura del horno", "seen"), ("guardate este dato", "seen"),
]


def main():
    p = argparse.ArgumentParser(description="Score keep_common.wants_kept on the live model.")
    p.add_argument("--url", default="http://127.0.0.1:8081")
    a = p.parse_args()
    wrong = []
    for text, want in CASES:
        got = keep_common.wants_kept(text, a.url)
        kind = got[0] if got else "not asked (server down?)"
        if kind != want:
            wrong.append((text, want, kind))
    print(f"{len(CASES) - len(wrong)}/{len(CASES)}")
    for text, want, kind in wrong:
        print(f"  wrong: {text!r}: wanted {want}, got {kind}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
