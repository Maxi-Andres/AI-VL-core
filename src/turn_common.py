"""turn_common.py — did the person finish, or did they stop mid-sentence?

The robot's ear (ear_common) closes an utterance after END_SILENCE_MS of silence. A pause to
think inside a sentence closes it too: "Acordate de que yo llevo" came in alone and was kept as
the note "yo llevo" (2026-10-28). Waiting longer after every utterance would slow every answer;
so the ear asks this, and holds only what looks cut off, for the rest to arrive.

The VLM server decides from the words, like keep_common: ~45 ms, any wording, both languages,
no list of words.
"""
import requests

CUT = (
    "A person talks to a robot and paused. Is what they said so far a finished message, or did "
    "they stop in the middle of a sentence and will go on? A short message, a question, an order "
    "or a greeting is DONE even without context. CUT only when the sentence is grammatically "
    "unfinished: it ends on a word that needs more after it, or what it announces is missing. "
    "Answer only DONE or CUT.")
EXAMPLES = [
    ("¿Qué ves?", "DONE"),
    ("Acordate de que el código es", "CUT"),
    ("Sacá una foto de la puerta", "DONE"),
    ("Can you tell me what the", "CUT"),
    ("Hola", "DONE"),
    ("Quiero que me digas si la", "CUT"),
    ("What time is it?", "DONE"),
    ("Acordate del número", "DONE"),
    ("Decime de qué marca es y cuánto", "CUT"),
]
TIMEOUT_S = 1.5


def looks_cut(text, url, timeout=TIMEOUT_S):
    """True when `text` stops mid-sentence; False when finished or when it cannot be asked (an
    utterance is never held on a guess)."""
    # Whisper closes "?" and "!" only where the intonation closed: finished, whatever the words.
    if text.rstrip().endswith(("?", "!")):
        return False
    shots = []
    for said, label in EXAMPLES:
        shots += [{"role": "user", "content": said}, {"role": "assistant", "content": label}]
    try:
        r = requests.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": "default", "temperature": 0, "max_tokens": 3,
            "messages": [{"role": "system", "content": CUT}, *shots,
                         {"role": "user", "content": text}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip().upper().startswith("CUT")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return False
