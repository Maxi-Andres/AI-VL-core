"""keep_common.py — does a message ask the robot to KEEP something? A picture, a note, or neither.

"Guardá esta imagen" and "acordate que…" are kept by the gateway (AI-VL-backend memory.py).
Matching them with patterns missed every new wording Whisper wrote — "Sacad una foto",
"Lo guardamos por todo este control", "A notar que hoy es el cumpleaños de Mateo" — and the
model then claimed it had kept them (2026-10-28). So the VLM server decides, from the text
alone, like search_common.model_says_web: ~45 ms, any wording, both languages, and for a note
it also says WHAT to remember.

Five answers, because a request to remember is not always a fact in words (2026-10-29):
  PICTURE        keep the picture                     "sacá una foto de esto"
  NOTE: <fact>   keep a fact said in words            "acordate que el código es 4455"
  SEEN: <what>   keep something only the picture has  "acordate de la medida", "acordate de esto"
                 — kept as the picture plus the fact the VLM reads off it (seen_fact), so a later
                 "¿cuál era la medida?" has the number; the note "la medida" alone answered nothing
  ASK            asks to remember but says nothing    "acordate de que…" (cut off) — the robot
                 asks what, instead of keeping a note that only says "yo llevo"
  NO             anything else
Questions go to the model too: Whisper puts question marks on orders ("¿Sacá esta foto?"), and
the gateway's rule "a question is never a request" dropped that one.

Measured 2026-10-29 on 65 messages none of the examples (the 43 from 2026-10-28 plus orders with
Whisper's question marks, "seen", cut-off requests and questions about the memory): 64 right,
~43 ms each; the miss is "Lo guardamos por todo este control" (Whisper's garble of "guardá una
foto de este control"). seen_fact on a real frame: ~0.2 s, and "No se muestra ninguna medida en
la imagen" when what it points at is not there.
"""
import requests

KEEP = (
    "A person talks to a robot that has a camera. Does the message ask the robot to KEEP "
    "something in its memory? PICTURE: it asks to take, save, register or keep a picture or what "
    "the robot sees or is looking at. NOTE: it asks the robot to remember, note down or keep a "
    "fact given in words — answer NOTE: followed by that fact, in the person's words. SEEN: it "
    "asks to remember something whose value is only in what the robot sees (this, that, the "
    "reading, the measurement, what it shows) — answer SEEN: followed by what to remember. ASK: "
    "it asks to remember but the fact is missing or cut off. NO: anything else — a question about "
    "what it saved or remembers, a description, a refusal like 'don't save it', the person "
    "telling their own memories. Only an ORDER to keep something is PICTURE, NOTE, SEEN or ASK. "
    "A question that asks for information — what it sees, what it saved, what was said, what a "
    "value was or is — is NO, even when it mentions a picture, a reading or remembering. Speech "
    "recognition also puts question marks on orders ('¿Sacá una foto?' is still an order): judge "
    "the words, not the punctuation. Answer only PICTURE, SEEN: <what>, NOTE: <fact>, ASK, or NO.")
EXAMPLES = [
    ("Sacá una foto de la puerta", "PICTURE"),
    ("Acordate que el código del portón es 4455", "NOTE: el código del portón es 4455"),
    ("¿Qué hay en la mesa?", "NO"),
    ("Remember the meeting moved to Friday", "NOTE: the meeting moved to Friday"),
    ("¿Te acordás del código?", "NO"),
    ("Acordate de la temperatura del horno", "SEEN: la temperatura del horno"),
    ("Yo me acuerdo de cuando era chico", "NO"),
    ("Keep a photo of this", "PICTURE"),
    ("Acordate de que", "ASK"),
    ("¿Qué te dije que anotaras?", "NO"),
    ("Anotá", "ASK"),
    ("Guardate este dato", "SEEN: este dato"),
    ("No la guardes", "NO"),
    ("¿Guardá lo que estás viendo?", "PICTURE"),
    ("¿Qué dice el cartel?", "NO"),
    ("Remember this number", "SEEN: this number"),
    ("What did you write down?", "NO"),
    ("¿Te acordás lo que marcaba?", "NO"),
]
NOTE_CHARS = 500
TIMEOUT_S = 2.0
SEEN_TIMEOUT_S = 20.0
SEEN_FACT = (
    "The person told you: «{said}». They want you to remember {what}, from what you see. Write "
    "that fact as one short, complete sentence (under 20 words) in {language}, with the concrete "
    "values you read or see: numbers and units, names, colors. If it is not in the picture, say "
    "that in one short sentence. Only the sentence, nothing else.")


def wants_kept(text, url, timeout=TIMEOUT_S):
    """("picture", ""), ("note", fact), ("seen", what to read off the picture), ("ask", "") or
    (None, ""); None when it could not be asked (the caller falls back on its own patterns)."""
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
    if head.startswith("ASK"):
        # Asking back "what should I remember?" never answers a question: the 4B took "What
        # did I ask you to remember?" for a cut-off request. Let the assistant answer it.
        return (None, "") if "?" in text else ("ask", "")
    for label, kind in (("NOTE", "note"), ("SEEN", "seen")):
        if head.startswith(label):
            said = answer[len(label):].lstrip(" :").strip().strip('"“”').strip()
            if not said:
                return (None, "")
            # One word is not a fact, it points at one: "acordate de esto" came back as the
            # note "esto". The picture has what it points at.
            if kind == "note" and len(said.split()) == 1:
                kind = "seen"
            return (kind, said[:NOTE_CHARS])
    return (None, "")


def seen_fact(said, what, image_b64, url, model="default", timeout=SEEN_TIMEOUT_S):
    """The fact `what` points at, read off the picture by the VLM ("la medida" -> "la medida es
    14.3 V"); "" when it could not be read. `said` is the whole request: its language and its
    context ("acordate de la medida del tablero 3")."""
    from vlm_common import asked_in_english
    language = "English" if asked_in_english(said) else "Spanish"
    image = image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}"
    try:
        r = requests.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": model, "temperature": 0, "max_tokens": 80,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": SEEN_FACT.format(said=said, what=what, language=language)}]}]})
        r.raise_for_status()
        fact = r.json()["choices"][0]["message"]["content"].strip().strip('"“”').strip()
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return ""
    return fact[:NOTE_CHARS]
