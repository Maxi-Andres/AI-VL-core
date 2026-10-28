"""keep_common.py — does a message ask the robot to KEEP something? A picture, a note, or neither.

"Guardá esta imagen" and "acordate que…" are kept by the gateway (AI-VL-backend memory.py).
Matching them with patterns missed every new wording Whisper wrote — "Sacad una foto",
"Lo guardamos por todo este control", "A notar que hoy es el cumpleaños de Mateo" — and the
model then claimed it had kept them (2026-10-28). So the VLM server decides, from the text
alone, like search_common.model_says_web: ~45 ms, any wording, both languages, and for a note
it also says WHAT to remember.

Measured 2026-10-28 on 43 messages none of the examples below (pictures and notes in Whisper's
wordings, polite requests, refusals, questions about the memory, the person's own memories):
41 right; the two misses were a question ("What did I ask you to remember?" — the gateway
never asks about a question that is not a polite request) and "registrá lo que ves" taken for
a note.
"""
import requests

KEEP = (
    "A person talks to a robot that has a camera. Does the message ask the robot to KEEP "
    "something in its memory? PICTURE: it asks to take, save, register or keep a picture or what "
    "the robot sees or is looking at — anything about an image, photo or what is in view is "
    "PICTURE, never NOTE. NOTE: it asks the robot to remember, note down or keep a fact in words "
    "— answer NOTE: followed by that fact, in the person's words. NO: anything else — a question "
    "about what it saved or remembers, a description, a refusal like 'don't save it', the person "
    "telling their own memories. Answer only PICTURE, NO, or NOTE: <fact>.")
EXAMPLES = [
    ("Sacá una foto de la puerta", "PICTURE"),
    ("Acordate que el código del portón es 4455", "NOTE: el código del portón es 4455"),
    ("¿Qué hay en la mesa?", "NO"),
    ("Remember the meeting moved to Friday", "NOTE: the meeting moved to Friday"),
    ("¿Te acordás del código?", "NO"),
    ("Keep a photo of this", "PICTURE"),
    ("No la guardes", "NO"),
    ("Guardá lo que estás viendo", "PICTURE"),
]
NOTE_CHARS = 500
TIMEOUT_S = 2.0


def wants_kept(text, url, timeout=TIMEOUT_S):
    """("picture", ""), ("note", what to remember) or (None, ""); None when it could not be
    asked (the caller falls back on its own patterns)."""
    shots = []
    for said, label in EXAMPLES:
        shots += [{"role": "user", "content": said}, {"role": "assistant", "content": label}]
    try:
        r = requests.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": "default", "temperature": 0, "max_tokens": 60,
            "messages": [{"role": "system", "content": KEEP}, *shots,
                         {"role": "user", "content": text}]})
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
    head = answer.upper()
    if head.startswith("PICTURE"):
        return ("picture", "")
    if head.startswith("NOTE"):
        note = answer[4:].lstrip(" :").strip().strip('"“”').strip()
        return ("note", note[:NOTE_CHARS]) if note else (None, "")
    return (None, "")
