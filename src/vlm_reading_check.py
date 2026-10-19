#!/usr/bin/env python3
"""
vlm_reading_check.py — Does the RUNNING app read the photos right? Brand, model, reading.

vlm_server_benchmark.py compares models with a handful of facts. This one asks the question
"is what the app serves today still reading correctly?" over every photo in fotos/, with the
answer known for each, through the very function the voice assistant's route calls
(vlm_common.query_vlm_stream, what iacore's /vlm/stream runs), with the robot's identity on
(persona_for) as the page sends it, against the llama-server the app runs (:8081). It runs the
code on disk, not the service's: a running iacore keeps the identity it started with, and this
needs no restart of the app. Run it after any change to the prompt, the identity, the model or
the image size.

Each question is asked --tries times, each with a FRESH frame (one pixel changed: the
server caches image embeddings, see vlm_server_benchmark.py), and scored by regexes that
must ALL match. A reading the model never gets is a model limit; one it gets sometimes is
the kind a prompt change breaks first, so both are reported.

The expected answers were read off the photos by eye on 2026-10-19. Where a photo shows no
brand or reading, only what it IS is asked.

Usage:
    python3 src/vlm_reading_check.py                     # identity of the G1, 3 tries, 1152 px
    python3 src/vlm_reading_check.py --robot none        # no identity, for comparison
    python3 src/vlm_reading_check.py --only 14 17 --tries 6
Results: results/vlm_reading_check_<robot>_<timestamp>.json (every answer) and a table.
"""
import argparse
import base64
import io
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime

import vlm_common
from vlm_common import PROJECT_ROOT, results_path

JPEG_QUALITY = 85
_FRESH = itertools.count(1)

# (photo, question, regexes that must ALL match, case-insensitive). Questions are spoken the way
# the user asks them, in Spanish.
CASES = [
    ("clean/1.jpeg", "¿De qué marca es el manómetro y en qué unidades mide?", [r"beyca", r"kg"]),
    # the pointer is the long thin end, resting on 0; the black paddle is its counterweight
    ("clean/1.jpeg", "¿Qué presión marca la aguja?", [r"\b0\b|cero"]),
    ("clean/2.jpeg", "¿En qué unidad mide este transmisor?", [r"mpa"]),
    ("clean/3.jpeg", "¿De qué marca es este instrumento y en qué unidad mide?",
     [r"jmp", r"hg|mercurio"]),
    ("clean/4.jpeg", "¿Qué es este objeto?", [r"temperatura|termo|rtd|pt\s*100"]),
    ("clean/5.jpeg", "¿Qué marca tiene el termómetro?", [r"de\s*wit"]),
    ("clean/5.jpeg", "¿En qué escalas mide?", [r"°\s*c\b|celsius|cent[ií]grad", r"°\s*f\b|fahrenheit"]),
    ("clean/6.jpeg", "¿Qué temperatura muestra la pantalla?", [r"650"]),
    ("clean/6.jpeg", "¿Qué emisividad tiene configurada?", [r"0[.,]95"]),
    ("clean/7.jpeg", "¿De qué marca es el caudalímetro?", [r"contatec"]),
    # the second line reads "198 1": a decimal point or not is not visible, so "198" is enough
    ("clean/8.jpeg", "¿Qué números muestra el display?", [r"31\s*244", r"198"]),
    ("clean/8.jpeg", "¿Qué modelo de envolvente es?", [r"adf\s*-?\s*60\s*v"]),
    ("clean/9.jpeg", "¿Qué tipo de medidor es este?", [r"ultras[oó]nic|ultrasonic"]),
    ("clean/10.jpeg", "¿De qué marca es este caudalímetro?", [r"\babb\b"]),
    ("clean/11.jpeg", "¿Qué instrumento es este?", [r"rot[aá]metro|rotameter|caudal|flow"]),
    ("clean/12.jpeg", "¿Qué tipo de sensor es este?", [r"proximidad|proximity|inductiv"]),
    ("clean/13.jpeg", "¿Qué marca y modelo es el analizador?", [r"fluke", r"438"]),
    ("clean/13.jpeg", "¿A cuántas rpm gira el motor según la pantalla?", [r"3\s*\.?471"]),
    ("clean/13.jpeg", "¿Qué eficiencia muestra?", [r"\b82\b"]),
    ("clean/14.jpeg", "¿Qué valor en kW muestra la pantalla?", [r"17[.,]84"]),
    ("clean/14.jpeg", "¿Cuántos kVAR marca?", [r"12[.,]26"]),
    ("clean/14.jpeg", "¿Cuántos kVA marca?", [r"21[.,]65"]),
    ("clean/14.jpeg", "¿Qué modelo y marca es el instrumento?", [r"8220", r"chauvin|arnoux"]),
    ("clean/15.jpeg", "¿Qué marca y modelo es el multímetro?", [r"fluke", r"\b87"]),
    ("clean/15.jpeg", "¿Qué valor muestra el multímetro?", [r"199[.,]98"]),
    ("clean/16.jpeg", "¿Qué equipo es este?",
     [r"transformador|aislador|alta tensi[oó]n|subestaci[oó]n|transformer"]),
    ("clean/17.jpeg", "¿Qué porcentaje de oxígeno marca?", [r"20[.,]7"]),
    ("clean/17.jpeg", "¿Qué temperatura marca?", [r"29[.,]3"]),
    ("clean/17.jpeg", "¿De qué marca y modelo es el detector?", [r"smart\s*sensor", r"ar\s*-?\s*8100"]),
    ("clean/18.jpeg", "¿Qué pH marca y a qué temperatura?", [r"6[.,]02", r"78[.,]5"]),
    ("clean/18.jpeg", "¿Qué marca y modelo es el medidor?", [r"\bhm\b", r"ph\s*-?\s*200"]),
    ("clean/19.jpeg", "¿Qué conductividad y temperatura muestra?", [r"982", r"12[.,]7"]),
    ("clean/19.jpeg", "¿Qué marca y modelo es?", [r"hanna", r"9033"]),
    ("clean/20.jpeg", "¿Qué valor muestra y en qué unidad?", [r"103", r"ntu"]),
    ("clean/20.jpeg", "¿Qué marca y modelo es el turbidímetro?", [r"hach", r"2100\s*q"]),
    ("clean/21.jpeg", "¿Qué velocidad de vibración muestra la pantalla?", [r"1[.,]3625"]),
    ("clean/21.jpeg", "¿Qué marca y modelo es el medidor?", [r"pce", r"vm\s*-?\s*20"]),
    ("clean/22.jpeg", "¿De qué marca son estos sensores?", [r"micro\s*-?\s*epsilon"]),
    ("clean/23.jpeg", "¿Qué es esto?", [r"bandeja|portacable|cable\s*tray|canaleta|canal"]),
    ("clean/24.jpeg", "¿Qué es esto?", [r"tablero|panel|cuadro|gabinete"]),
    ("clean/25.jpeg", "¿Cuántos prensaestopas tiene la caja?", [r"\b(4|cuatro)\b"]),
    ("clean/26.jpeg", "¿Cuántos prensaestopas tiene la caja?", [r"\b(6|seis)\b"]),
    ("clean/27.jpeg", "¿De qué marca son los interruptores?", [r"steck"]),
    ("clean/27.jpeg", "¿Cuántos interruptores termomagnéticos rojos hay arriba?", [r"\b(3|tres)\b"]),
    ("clean/28.jpeg", "¿De qué marca es el tablero y qué tensión indica?", [r"soldexel?", r"208"]),  # a tick covers the L
    ("clean/28.jpeg", "¿Cuántas luces verdes están encendidas?", [r"\b(2|dos)\b"]),
    ("clean/29.jpeg", "¿Qué es esto?", [r"tablero|panel|cuadro|gabinete"]),
    ("clean/31.jpeg", "¿Qué instrumento es y de qué marca?",
     [r"teodolito|estaci[oó]n total|theodolite|total station", r"spectra"]),
    ("ciudad/101.jpeg", "¿Cuántas personas hay en la imagen?", [r"\b(2|dos)\b"]),
    ("ciudad/101.jpeg", "¿De qué marca es la impresora?", [r"\bhp\b|hewlett"]),
    ("ciudad/101.jpeg", "¿Qué empresa aparece escrita en la pared?", [r"silk"]),
    ("ciudad/100.jpeg", "¿Cuántas bicicletas hay en la calle?", [r"\b(1|una?|uno)\b"]),
    ("ciudad/100.jpeg", "¿De qué color es el auto que está en el centro de la calle?",
     [r"negr|black|oscur"]),
    ("scenes/go2.jpeg", "¿Qué es esto?", [r"perro|robot|cuadr[uú]pedo|go2|unitree"]),
    ("scenes/go2.jpeg", "¿Qué número tiene pintado?", [r"\b0?2\b"]),
    ("scenes/moon.jpeg", "¿Qué se ve en esta foto?", [r"\bluna\b|moon"]),
    ("scenes/castle.jpeg", "¿Qué edificio es este?", [r"castillo|castle", r"japon|japan"]),
]


def frame_b64(path, max_side):
    """The photo as the page sends it — longest side at most max_side, JPEG 85 — with one
    pixel changed per call so no cache recognizes it."""
    from PIL import Image
    n = next(_FRESH)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_side:
            im.thumbnail((max_side, max_side))
        w, h = im.size
        im.putpixel(((n * 7919) % w, (n * 104729) % h), (n % 256, (n * 31) % 256, (n * 97) % 256))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def ask(cfg, url, image, question, robot, max_tokens):
    """One streamed answer, as /vlm/stream gets it: (text, seconds to first piece, total)."""
    t0 = time.time()
    first = None
    parts = []
    for piece in vlm_common.query_vlm_stream(image, cfg.get("model", "qwen3-vl:4b"), question,
                                             max_tokens=max_tokens,
                                             num_ctx=cfg.get("num_ctx", 16384),
                                             url=url, robot=robot):
        first = first or time.time() - t0
        parts.append(piece)
    return "".join(parts).strip(), first, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="the VLM server (default: what the app uses)")
    ap.add_argument("--robot", default="g1", help='whose identity answers; "none" for none')
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--size", type=int, default=1152, help="longest side sent (0 = as is)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=[],
                    help="photo numbers or names to keep (e.g. 14 go2)")
    args = ap.parse_args()
    robot = None if args.robot.lower() in ("none", "") else args.robot
    cfg = vlm_common.load_config()
    url = args.url or os.environ.get("VLM_URL") or cfg.get("llamacpp_url",
                                                            vlm_common.LLAMACPP_HOST)

    cases = [c for c in CASES
             if not args.only or os.path.splitext(os.path.basename(c[0]))[0] in args.only]
    rows, results = [], []
    for photo, question, patterns in cases:
        path = os.path.join(PROJECT_ROOT, "fotos", photo)
        answers = []
        for _ in range(args.tries):
            try:
                text, first, total = ask(cfg, url, frame_b64(path, args.size), question, robot,
                                         args.max_tokens)
            except Exception as e:  # one failed request counts as a wrong answer
                text, first, total = f"[error] {e}", None, None
            ok = all(re.search(p, text, re.IGNORECASE) for p in patterns)
            answers.append({"answer": text, "ok": ok, "first_s": first, "total_s": total})
        hits = sum(a["ok"] for a in answers)
        results.append({"photo": photo, "question": question, "expect": patterns,
                        "hits": hits, "tries": args.tries, "answers": answers})
        wrong = next((a["answer"] for a in answers if not a["ok"]), "")
        rows.append((photo, question, hits, wrong))
        print(f"{hits}/{args.tries}  {photo:18} {question[:58]:58} "
              f"{('| ' + wrong.replace(chr(10), ' ')[:90]) if wrong else ''}", flush=True)

    total_hits = sum(r["hits"] for r in results)
    total = sum(r["tries"] for r in results)
    always = sum(r["hits"] == r["tries"] for r in results)
    never = sum(r["hits"] == 0 for r in results)
    print(f"\n{total_hits}/{total} answers right ({100 * total_hits / max(total, 1):.0f} %); "
          f"{always}/{len(results)} questions always right, {never} never.")
    out = results_path(f"vlm_reading_check_{robot or 'none'}_"
                       f"{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"robot": robot, "tries": args.tries, "size": args.size,
                   "date": datetime.now().isoformat(timespec="seconds"), "results": results},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
