#!/usr/bin/env python3
"""
vlm_chat_check.py — Does the RUNNING app talk well? Manners, not readings.

vlm_reading_check.py asks whether the assistant reads the photos right. This one asks how it
talks: to a question about the frame, about itself, the time, general knowledge, a passing
comment, a greeting, thanks, a word Whisper made out of noise, in Spanish and in English, and
in conversations where the earlier turns ride along. It runs the code on disk through the very
function the app's route calls (vlm_common.query_vlm_stream), against the llama-server the app
runs, so a prompt change is measured before the app is restarted.

Why it exists (2026-10-28): spoken "Game", "¡Qué bueno es!" and "Buen día" each got the same
"¡Hola! 🤖 Estás mostrándome un suelo de madera… ¿Te gustaría saber más? 😊" — the model copied
its own earlier answer out of the history — and "¿Qué hora es?" got "No puedo determinar la
hora… Es 14:12". A prompt that fixes that can break reading (it has, twice): run BOTH checks.

Every answer is scored by generic checks — none knows a topic:
  emoji        no emoji
  acting       no *actions between asterisks*
  greeting     does not open with a greeting, unless it was greeted
  ends_asking  does not end with a question, unless it did not understand
  language     answers in the language it was asked in
  length       short, unless asked to describe or tell
  on_topic     a comment, greeting or noise is not answered with a description of the frame
  sees         "¿qué ves?", "what are you looking at?" are answered with what is in the frame,
               never "no puedo ver"
  own_eyes     the frame is what it sees, not something it was shown ("Estás mostrándome un
               suelo de madera…", 2026-10-28)
  repeats      not a copy of one of its earlier answers in the conversation
  expect       the case's own regex, when it has one (the time, its name, a fact)

Usage:
    python3 src/vlm_chat_check.py                         # every case, 1 try
    python3 src/vlm_chat_check.py --tries 2 --only chat noise
    python3 src/vlm_chat_check.py --set CHAT_STYLE="No emojis."   # a prompt, without editing code
Results: results/vlm_chat_check_<timestamp>.json (every answer) and a table of what failed.
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime

import vlm_common
from vlm_common import PROJECT_ROOT, results_path
from vlm_reading_check import frame_b64

OFFICE, STREET, METER, POWER, DOG = ("ciudad/101.jpeg", "ciudad/100.jpeg", "clean/15.jpeg",
                                     "clean/14.jpeg", "scenes/go2.jpeg")
NAME = {"g1": r"bender", "go2": r"huesos"}
TIME = r"\b\d{1,2}[:.]\d{2}\b"

OFFICE_SEEN = r"oficina|escritorio|personas|computadora|office|desk|people|computer"
STREET_SEEN = r"calle|auto|coche|edificio|bicicleta|street|car|building|bike"

# (kind, photo, robot, message, expect regex or None). Kinds: sees (what it is looking at),
# image, self, time, knowledge, chat (a comment), greeting, thanks, noise (what Whisper makes
# of a cough or a TV).
SINGLE = [
    ("sees", OFFICE, "g1", "¿Qué estás viendo?", OFFICE_SEEN),
    ("sees", OFFICE, "go2", "¿Qué ves?", OFFICE_SEEN),
    ("sees", OFFICE, "g1", "¿Qué tenés adelante?", OFFICE_SEEN),
    ("sees", OFFICE, "go2", "¿Dónde estás?", r"oficina|office"),
    ("sees", OFFICE, "g1", "What are you looking at?", OFFICE_SEEN),
    ("sees", OFFICE, "g1", "Where are you?", r"office"),
    ("sees", STREET, "go2", "¿Qué estás viendo ahora?", STREET_SEEN),
    ("sees", STREET, "g1", "Mirá, ¿qué hay ahí?", STREET_SEEN),
    ("sees", STREET, "g1", "What can you see?", STREET_SEEN),
    ("sees", METER, "g1", "¿Qué tenés en frente?", r"mult[ií]metro|multimeter|fluke"),
    ("sees", DOG, "go2", "¿Qué ves enfrente tuyo?", r"robot|perro|dog"),
    ("sees", DOG, "g1", "¿Hay alguien con vos?", r"robot|perro|dog"),
    ("image", OFFICE, "g1", "¿Qué ves?", None),
    ("image", OFFICE, "g1", "¿Cuántas personas hay?", r"\b(2|dos)\b"),
    ("image", OFFICE, "go2", "¿Qué dice en la pared?", r"silk"),
    ("image", OFFICE, "g1", "¿Hay alguien trabajando?", r"\b(s[ií]|yes)\b"),
    ("image", OFFICE, "g1", "What brand is the printer?", r"\bhp\b"),
    ("image", STREET, "go2", "¿Es seguro cruzar?", None),
    ("image", STREET, "g1", "¿De qué color es el auto del centro?", None),
    ("image", STREET, "g1", "How many bikes are there?", r"\b(1|one|a)\b"),
    ("image", METER, "g1", "¿Qué marca el multímetro?", r"199[.,]98"),
    ("image", METER, "g1", "¿Es un valor normal para un enchufe?", None),
    ("image", POWER, "g1", "¿Cuántos kW marca?", r"17[.,]84"),
    ("image", POWER, "g1", "What is the power factor?", None),
    ("image", DOG, "g1", "¿Qué es eso?", r"robot|perro|dog"),
    ("image", DOG, "go2", "¿Qué número tiene?", r"\b0?2\b"),
    ("self", OFFICE, "g1", "¿Quién sos?", NAME["g1"]),
    ("self", OFFICE, "go2", "¿Cómo te llamás?", NAME["go2"]),
    ("self", OFFICE, "g1", "What's your name?", NAME["g1"]),
    ("self", OFFICE, "g1", "¿Quién te creó?", r"maxi"),
    ("self", OFFICE, "go2", "¿Tenés hermanos?", NAME["g1"]),
    ("self", OFFICE, "g1", "¿Dónde trabajás?", r"silk"),
    ("self", OFFICE, "g1", "¿Qué sos?", r"g1|humanoide|humanoid"),
    ("time", OFFICE, "go2", "¿Qué hora es?", TIME),
    ("time", OFFICE, "go2", "¡Qué hora es!", TIME),
    ("time", OFFICE, "g1", "What time is it?", TIME),
    ("time", OFFICE, "g1", "¿Qué día es hoy?", r"\b\d{1,2}\b|lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo"),
    ("knowledge", OFFICE, "g1", "¿Cuál es la capital de Japón?", r"tokio|tokyo"),
    ("knowledge", OFFICE, "g1", "¿Cuánto es 12 por 7?", r"\b84\b"),
    ("knowledge", OFFICE, "go2", "¿Qué es un manómetro?", r"presi[oó]n"),
    ("knowledge", OFFICE, "g1", "Who wrote Don Quixote?", r"cervantes"),
    ("knowledge", OFFICE, "g1", "Contame un chiste", None),
    ("knowledge", OFFICE, "g1", "¿Cómo se dice gracias en inglés?", r"thank"),
    ("chat", OFFICE, "go2", "¡Qué bueno es!", None),
    ("chat", OFFICE, "g1", "Sos un genio", None),
    ("chat", OFFICE, "g1", "Estoy cansado", None),
    ("chat", OFFICE, "go2", "Me gusta tu voz", None),
    ("chat", OFFICE, "g1", "That's cool", None),
    ("chat", OFFICE, "g1", "Bueno, después seguimos", None),
    ("chat", OFFICE, "go2", "¿Cómo estás?", None),
    ("chat", OFFICE, "g1", "How are you?", None),
    ("greeting", OFFICE, "go2", "Buen día, bienvenido", None),
    ("greeting", OFFICE, "g1", "Hola", None),
    ("greeting", OFFICE, "g1", "Hello there", None),
    ("thanks", OFFICE, "g1", "Gracias", None),
    ("thanks", OFFICE, "go2", "Muchas gracias, Huesos", None),
    ("thanks", OFFICE, "g1", "Thanks!", None),
    ("noise", OFFICE, "go2", "Game", None),
    ("noise", OFFICE, "go2", "¡Dios lejos!", None),
    ("noise", OFFICE, "g1", "Bender Bender Bender", None),
    ("noise", OFFICE, "g1", "Mmm", None),
    ("noise", OFFICE, "go2", "y entonces la", None),
]
# Conversations: each turn goes with the earlier ones as history, as the page sends them.
CONVERSATIONS = [
    ("the spoken session of 2026-10-28", OFFICE, "go2", [
        ("noise", "Bender Bender Bender", None), ("chat", "¡Qué bueno es!", None),
        ("time", "¿Qué hora es?", TIME), ("greeting", "Buen día, bienvenido", None),
        ("noise", "Game", None), ("image", "¿Cuántas personas hay?", r"\b(2|dos)\b"),
    ]),
    ("looking around", STREET, "go2", [
        ("sees", "¿Qué estás viendo?", STREET_SEEN), ("sees", "¿Y qué más?", None),
        ("image", "¿Hay gente?", None), ("chat", "Qué lindo día", None),
        ("sees", "¿Dónde estamos?", r"calle|ciudad|street|city"),
    ]),
    ("follow-ups about the frame", OFFICE, "g1", [
        ("image", "¿Qué hay sobre el escritorio de la izquierda?", None),
        ("image", "¿Y en el de la derecha?", None),
        ("image", "¿De qué color es la silla?", None),
        ("chat", "Perfecto, gracias", None),
        ("self", "¿Qué te pregunté primero?", r"escritorio|izquierda"),
    ]),
    ("readings in a row", METER, "g1", [
        ("image", "¿Qué valor marca?", r"199[.,]98"),
        ("image", "¿Y en qué unidad?", r"\bv\b|volt"),
        ("chat", "Bien, anotalo", None),
        ("image", "¿De qué marca es?", r"fluke"),
    ]),
    ("english chat", OFFICE, "g1", [
        ("greeting", "Hi Bender", None), ("image", "What do you see?", None),
        ("chat", "Nice", None), ("self", "Who made you?", r"maxi"), ("thanks", "Thank you", None),
    ]),
]

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
ACTING = re.compile(r"\*[a-záéíóúñ][^*\n]{0,40}\*", re.IGNORECASE)
GREETS = re.compile(r"^\W*(hola|buen[oa]s?\b|hi\b|hello|hey)", re.IGNORECASE)
DESCRIBES = re.compile(r"\b(imagen|foto|veo|se ve|suelo|piso|madera|escritorio|oficina|image|"
                       r"picture|photo|floor|i see|i can see|desk|office)\b", re.IGNORECASE)
DENIES_SEEING = re.compile(r"no (puedo|podr[ií]a) ver|no tengo (ojos|vista|c[aá]mara)|no veo nada|"
                           r"can'?t see|cannot see|don'?t have (eyes|a camera)|unable to see",
                           re.IGNORECASE)
SHOWN = re.compile(r"me (est[aá]s )?mostr|mostr[aá]ndome|me (ense[nñ]|pas)[aá]s|en (la|esta) "
                   r"(imagen|foto) (que|se)|you('re| are) showing|you showed|in (the|this) "
                   r"(image|picture|photo) you", re.IGNORECASE)
TELL = re.compile(r"describ|cont[aá]|explic|tell|chiste|joke", re.IGNORECASE)
ES = {"el", "la", "de", "que", "y", "en", "es", "un", "una", "los", "las", "con", "por", "para",
      "no", "se", "del", "al", "lo", "mi", "te", "hay", "está", "estás", "soy", "son"}
EN = {"the", "a", "is", "of", "and", "to", "in", "it", "you", "that", "with", "for", "are",
      "on", "this", "i", "my", "i'm", "there", "what", "your", "am"}
LONG = 320


def language(text):
    words = re.findall(r"[a-záéíóúñ']+", text.lower())
    es, en = sum(w in ES for w in words), sum(w in EN for w in words)
    if es + en < 2:
        return None
    return "es" if es > en else "en" if en > es else None


def score(kind, message, answer, expect, earlier_answers):
    """{check: passed} for one answer; a check that does not apply is left out."""
    last = re.split(r"(?<=[.!?…])\s+", answer.strip())[-1] if answer.strip() else ""
    checks = {
        "emoji": not EMOJI.search(answer),
        "acting": not ACTING.search(answer),
        "ends_asking": kind == "noise" or not last.rstrip().endswith("?"),
        "length": bool(TELL.search(message)) or kind in ("image", "sees") or len(answer) <= LONG,
    }
    if kind not in ("greeting",):
        checks["greeting"] = not GREETS.search(answer)
    asked_in, answered_in = language(message), language(answer)
    if asked_in and answered_in:
        checks["language"] = asked_in == answered_in
    if kind in ("chat", "greeting", "thanks", "noise"):
        checks["on_topic"] = not DESCRIBES.search(answer)
    if kind == "sees":
        checks["sees"] = not DENIES_SEEING.search(answer)
    if kind in ("sees", "image"):
        checks["own_eyes"] = not SHOWN.search(answer)
    if earlier_answers:
        checks["repeats"] = all(difflib.SequenceMatcher(None, answer, a).ratio() < 0.6
                                for a in earlier_answers)
    if expect:
        checks["expect"] = bool(re.search(expect, answer, re.IGNORECASE))
    return checks


def ask(url, image, message, robot, history):
    t0 = time.time()
    text = "".join(vlm_common.query_vlm_stream(image, "default", message, max_tokens=300, url=url,
                                               robot=robot, history=history, search=False))
    return text.strip(), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="the VLM server (default: what the app uses)")
    ap.add_argument("--tries", type=int, default=1)
    ap.add_argument("--size", type=int, default=1152, help="longest side sent (0 = as is)")
    ap.add_argument("--only", nargs="*", default=[], help="kinds to keep (and 'conversations')")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override a vlm_common constant for this run (a prompt to try)")
    ap.add_argument("--label", default="", help="a name for this run in the results file")
    args = ap.parse_args()
    for item in args.set:
        name, _, value = item.partition("=")
        if not hasattr(vlm_common, name):
            ap.error(f"vlm_common has no {name}")
        setattr(vlm_common, name, value)
    cfg = vlm_common.load_config()
    url = args.url or os.environ.get("VLM_URL") or cfg.get("llamacpp_url", vlm_common.LLAMACPP_HOST)
    path = lambda photo: os.path.join(PROJECT_ROOT, "fotos", photo)  # noqa: E731

    results = []

    def record(group, kind, robot, message, answer, seconds, checks):
        results.append({"group": group, "kind": kind, "robot": robot, "message": message,
                        "answer": answer, "seconds": round(seconds, 2), "checks": checks})
        bad = [c for c, ok in checks.items() if not ok]
        print(f"{'ok ' if not bad else 'BAD'} {kind:9} [{robot}] {message[:34]:34} "
              f"{','.join(bad):22} | {answer.replace(chr(10), ' ')[:110]}", flush=True)

    for kind, photo, robot, message, expect in SINGLE:
        if args.only and kind not in args.only:
            continue
        for _ in range(args.tries):
            answer, s = ask(url, frame_b64(path(photo), args.size), message, robot, None)
            record("single", kind, robot, message, answer, s, score(kind, message, answer, expect, []))
    if not args.only or "conversations" in args.only:
        for title, photo, robot, turns in CONVERSATIONS:
            for _ in range(args.tries):
                print(f"--- {title}")
                history = []
                for kind, message, expect in turns:
                    answer, s = ask(url, frame_b64(path(photo), args.size), message, robot, history)
                    checks = score(kind, message, answer, expect, [t["answer"] for t in history])
                    record(title, kind, robot, message, answer, s, checks)
                    history.append({"question": message, "answer": answer})

    names = sorted({c for r in results for c in r["checks"]})
    print("\ncheck         passed")
    for name in names:
        got = [r["checks"][name] for r in results if name in r["checks"]]
        print(f"  {name:11} {sum(got):3}/{len(got):<3} ({100 * sum(got) / len(got):.0f} %)")
    clean = sum(all(r["checks"].values()) for r in results)
    print(f"\n{clean}/{len(results)} answers passed every check "
          f"({100 * clean / max(len(results), 1):.0f} %).")
    out = results_path(f"vlm_chat_check_{args.label + '_' if args.label else ''}"
                       f"{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "set": args.set,
                   "tries": args.tries, "size": args.size, "results": results},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
