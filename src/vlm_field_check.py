#!/usr/bin/env python3
"""
vlm_field_check.py — Does the VLM read and SEE a messy field photo, and does it say "I can't
tell" when it can't?

vlm_reading_check.py asks about clean catalog shots, and every model worth trying gets almost
all of it right, so it cannot tell two good models apart. This one uses real photos like the
robot's camera takes — at an angle, dim, with glare, small print, clutter, people at work —
and asks harder things: exact readings with decimals, signs and units; counting; left/right
and which one is on; states (open/closed, needle position, which function a knob is on); small
or partial text; and comparing two instruments. It also asks the questions a robot assistant
gets wrong in the field, scored on BEHAVING right rather than on a value:

  absent         something that is not in the photo ("¿de qué marca es la impresora?" with
                 no printer): it must say there is none, not invent one;
  unreadable     text too small, blurred or covered to read: it must say it can't read it,
                 and the invented value must not appear;
  ambiguous      "¿qué marca la aguja?" with two gauges in view: it must answer for both,
                 or ask which one;
  false_premise  "¿por qué la luz roja está encendida?" when it is green: it must correct
                 the premise, not explain it.

Like vlm_reading_check.py it goes through the very function the voice assistant's route calls
(vlm_reading_check.ask -> vlm_common.query_vlm_stream) with the robot's identity on, against
the llama-server the app runs (:8081), and each question is asked --tries times with a FRESH
frame (vlm_reading_check.frame_b64 changes one pixel so the server's image cache never
answers). An answer is right when every regex in `must` matches and none in `must_not` does
(case-insensitive, across lines). Most questions are in Rioplatense Spanish, as people ask
them; about one in ten is in English.

The photos are in fotos/field/, all from Wikimedia Commons under free licenses; who took
each one, its page and its license are in fotos/field/SOURCES.md (CC BY and BY-SA require
that attribution). Every expected answer was read off the full-size photo by eye, zooming
where needed, on 2026-10-30, BEFORE any model answered; a question whose answer could not
be settled by looking was dropped. A regex may be changed after seeing answers only to
accept a clearly correct answer it rejected, and then every saved run is rescored with
--rescore so all models are judged by the same regexes.

Usage:
    python3 src/vlm_field_check.py                       # 3 tries, label = the server's alias
    python3 src/vlm_field_check.py --label qwen35-4b --tries 1
    python3 src/vlm_field_check.py --only f012 f040      # some photos
    python3 src/vlm_field_check.py --kind absent false_premise
    python3 src/vlm_field_check.py --rescore results/vlm_field_check_X.json [more.json ...]
Results: results/vlm_field_check_<label>_<timestamp>.json (every answer, plus the summary per
question type) and a table on screen with the time to the first word (p50).
"""
import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime

import vlm_common
from vlm_common import PROJECT_ROOT, results_path
from vlm_reading_check import ask, frame_b64

PHOTO_DIR = os.path.join(PROJECT_ROOT, "fotos", "field")

KINDS = ("reading", "counting", "spatial", "state", "small_text", "absent", "unreadable",
         "ambiguous", "false_premise")

# --- The honest answers, shared by the behavior questions ------------------------------------
# "It is not there": no veo / no hay / no aparece / ninguna / there is no ...
NOT_THERE = (r"no (se )?(ve|ven|veo|vemos|aparece|aparecen|observa|observo|distingo|identifico|"
             r"encuentro|detecto|muestra)\b|no hay\b|no logro (ver|identificar|distinguir)|"
             r"\bningun[ao]?\b|no (est[aá]|figura)n? (visible|a la vista|en (la|esta) "
             r"(imagen|foto|vista))|no tengo .{0,30}(a la vista|en la imagen|en la vista)|"
             r"there (is|are) no\b|\bno \w+ (is|are) (visible|shown)|"
             r"no (se )?(indica|menciona|especifica|figura|incluye|dice)\b|no est[aá] "
             r"(expl[ií]citamente )?(indicad|especificad|escrit|marcad)|"
             r"(do not|don'?t|cannot|can'?t) see|not (visible|shown|present)")
# "I can't read it": ilegible / no alcanzo a leer / borroso / too small ...
CANT_READ = (r"no (se )?(puede|alcanza|llega|logra)n? (a )?(leer|ver|distinguir|apreciar)|"
             r"no (puedo|alcanzo a|llego a|logro) (leer|ver|distinguir|determinar|precisar|"
             r"apreciar|identificar|confirmar)|ilegible|borros|desenfoc|difuminad|pixelad|"
             r"(demasiado|muy) (chic|pequeñ|lej|osc)|no (es|son|est[aá]n?|resulta) "
             r"(legibles?|claros?|claras?|n[ií]tid|visibles?)|sin texto legible|"
             r"no tiene texto legible|no hay (texto|n[uú]meros?)[^.]{0,30}legible|"
             r"no se (lee|leen|distingue|distinguen|ve|ven)"
             r"( bien| con claridad| claramente| con nitidez)?\b|no (es|est[aá]) "
             r"(posible|f[aá]cil) (leer|determinar|distinguir|identificar|precisar)|"
             r"can'?t (read|make out)|"
             r"cannot (read|make out|determine)|(not|un)readable|illegible|too (small|blurry|far)")
# A display that shows nothing: apagada / en blanco / no muestra nada ...
BLANK = (r"apagad|en blanco|vac[ií]a|sin (lectura|valor|n[uú]meros|d[ií]gitos|nada)|"
         r"no (muestra|marca|indica|tiene|aparece|se ve) (una? )?(nada|ning[uú]n|ningun|valor|"
         r"n[uú]mero|lectura|d[ií]gito|(una |ninguna )?hora)|\bblank\b|turned off|\boff\b|nothing")
# Asking back which one: ¿cuál de los dos ...?
ASKS_WHICH = r"¿\s*cu[aá]l|cu[aá]l de (los|las|ellos|ellas)|which (one|of)"

_WORDS = {0: "cero|zero|ning[uú]n[ao]?|nadie|nobody|none", 1: "una?|uno|one|single|sol[oa]",
          2: "dos|two|ambos|ambas|both|par de", 3: "tres|three", 4: "cuatro|four", 5: "cinco|five",
          6: "seis|six", 7: "siete|seven", 8: "ocho|eight"}


def n(k):
    """A count said as a digit or a word (Spanish or English)."""
    return rf"\b({k}|{_WORDS[k]})\b"


def dec(whole, frac):
    """A decimal with comma or point: dec(47, 3) matches 47,3 and 47.3."""
    return rf"(?<![\d.,]){whole}[.,]{frac}(?!\d)"


R, C, S, ST, T = "reading", "counting", "spatial", "state", "small_text"
AB, UN, AM, FP = "absent", "unreadable", "ambiguous", "false_premise"

# (photo, kind, question, must = regexes that must ALL match, must_not = none may match).
# The comment above a block says what is in the photo, read by eye.
CASES = [
    # c002 — Waaree pressure gauge, dual scale kg/cm² 0-7 and psi 0-100; the pointer (the long
    # thin end) rests on 0, the black paddle is its counterweight.
    ("c002.jpg", R, "¿De qué marca es el manómetro?", [r"waaree"], []),
    ("c002.jpg", R, "¿Hasta cuántos kg/cm² mide?", [n(7)], []),
    ("c002.jpg", R, "¿Y hasta cuántos psi?", [r"\b100\b"], []),
    # c003 — two gauges in bar: a chrome one 0-6 bar, a black WIKA one 0-0.6 bar; both on 0.
    ("c003.jpg", C, "¿Cuántos manómetros hay y en qué unidad miden?", [n(2), r"\bbar"], []),
    ("c003.jpg", R, "¿De qué marca es el manómetro de caja negra?", [r"wika"], []),
    ("c003.jpg", AM, "¿Hasta cuánto mide el manómetro?",
     [r"(?=.*(?<![.,\d])6(?![.,]?\d))(?=.*0[.,]6)|" + ASKS_WHICH], []),
    # c005 — a pressure transmitter in an open box; behind it, snow on the ground and one worker
    # in a dark vest with a red-and-white hard hat, from behind.
    ("c005.jpg", R, "¿Qué instrumento hay dentro de la caja?",
     [r"transmisor|transmitter|sensor|transductor"], []),
    ("c005.jpg", S, "¿Qué hay en el piso afuera, a la derecha?", [r"nieve|snow|hielo"], []),
    ("c005.jpg", C, "¿Hay alguna persona en la foto? ¿Cuántas?",
     [n(1) + r"|un (trabajador|operario|hombre)"], [r"\bno hay (ninguna )?persona"]),
    # c012 — UNI-T UT201+ clamp meter reading 220.0 V AC at a socket.
    ("c012.jpg", R, "¿Qué tensión está midiendo la pinza?", [r"220"], []),
    ("c012.jpg", R, "¿Qué marca y modelo es la pinza?", [r"uni\s*-?\s*t", r"ut\s*-?\s*201"], []),
    # c014 — Extech EX510, display blank, knob on OFF, IP67.
    ("c014.jpg", R, "¿Qué marca y modelo es el multímetro?", [r"extech", r"ex\s*-?\s*510"], []),
    ("c014.jpg", T, "¿Qué grado de protección IP tiene?", [r"ip\s*-?\s*67"], []),
    ("c014.jpg", ST, "¿En qué posición está la perilla?", [r"\boff\b|apagad"], []),
    ("c014.jpg", UN, "¿Qué valor muestra la pantalla?", [BLANK], [r"\d+[.,]\d+\s*(v|a|ω|ohm)"]),
    # c016 — Voltcraft VC160, display blank.
    ("c016.jpg", R, "¿Qué marca y modelo es el multímetro?", [r"voltcraft", r"vc\s*-?\s*160"], []),
    ("c016.jpg", UN, "¿Qué valor muestra la pantalla?", [BLANK], [r"\d+[.,]\d+"]),
    # c017 — Gossen Metrawatt Metra Hit 23S, shows -0.0000 A DC, knob on A.
    ("c017.jpg", R, "¿Qué marca y modelo es el multímetro?",
     [r"metra\s*hit|gossen|metrawatt", r"23\s*s"], []),
    ("c017.jpg", ST, "¿Qué está midiendo, voltaje o corriente?", [r"corriente|amper|current"], []),
    ("c017.jpg", R, "¿Qué valor exacto muestra la pantalla, con el signo?",
     [r"[-−–]\s*0[.,]0{3,4}|menos\s*(0|cero)|negativ"], []),
    # c018 — Voltcraft VC-20, label "NCV: 200~1000VAC".
    ("c018.jpg", R, "¿De qué marca es el instrumento?", [r"voltcraft"], []),
    ("c018.jpg", T, "¿Qué rango de tensión dice la etiqueta NCV?", [r"200", r"1000"], []),
    # c022 — Maxwell (Jaipur) static meter, S.No 916076, "102" handwritten twice with marker.
    ("c022.jpg", T, "¿Cuál es el número de serie del medidor?", [r"916076"], []),
    ("c022.jpg", T, "¿Quién fabricó el medidor?", [r"maxwell"], []),
    ("c022.jpg", R, "¿Qué número escribieron a mano con marcador?", [r"\b[1i]?02\b"], []),
    # c024 — three meters on a wall in a row; the two left ones are grey plastic, the right one
    # is bare metal (silver).
    ("c024.jpg", C, "¿Cuántos medidores de luz hay en la pared?", [n(3)], []),
    ("c024.jpg", S, "¿Cuál de los medidores es distinto a los otros, el de la izquierda, "
                    "el del medio o el de la derecha?",
     [r"derech|right"], [r"(es|is) (el|the) (de la )?izquierd|(es|is) (el|the) (del )?medio"]),
    # c026 — Siemens three-phase meter, 168536,3 kWh, Nr 4454794, 1990, 3x230/400 V.
    ("c026.jpg", R, "¿Cuántos kWh marca el medidor?", [r"168\s*\.?\s*536"], []),
    ("c026.jpg", T, "¿Cuál es el número de serie del medidor?", [r"4454794"], []),
    ("c026.jpg", T, "¿De qué año es el medidor?", [r"1990"], []),
    ("c026.jpg", R, "What brand is this meter and what voltage is it rated for?",
     [r"siemens", r"230\s*/\s*400"], []),
    # c031 — open breaker panel, hand-written labels; "MASTER BATH FAN" in bold marker; two
    # Dryer breakers of 30 A, Hot Tub 240 Volts 50 A.
    ("c031.jpg", T, "¿Qué dice la etiqueta escrita en negrita con marcador negro?",
     [r"master", r"bath", r"fan"], []),
    ("c031.jpg", T, "¿De cuántos amperes son los interruptores del secarropas (Dryer)?",
     [r"\b30\b"], []),
    ("c031.jpg", T, "How many amps is the Hot Tub breaker?", [r"\b50\b"], []),
    # c034 — Itron water meter, counter 00365 m³ (+ red 313 litres), serial 2023A3003524.
    ("c034.jpg", R, "¿Cuántos metros cúbicos marca el medidor de agua?", [r"\b0*365\b"], []),
    ("c034.jpg", R, "¿De qué marca es el medidor?", [r"itron"], []),
    ("c034.jpg", T, "¿Cuál es el número de serie?", [r"2023\s*a\s*3003524"], []),
    # c036 — two light-blue meters on a red-painted brick wall with red pipes.
    ("c036.jpg", C, "¿Cuántos medidores hay en la pared?", [n(2)], []),
    ("c036.jpg", R, "¿De qué color están pintadas la pared y las cañerías?",
     [r"roj|rosa|red|pink|magenta|fucsia"], []),
    # c037 — Sagemcom water meter, battery symbol "2033", IP54, display blank.
    ("c037.jpg", R, "¿De qué marca es el medidor?", [r"sagemcom"], []),
    ("c037.jpg", T, "¿Qué año dice en el símbolo de la batería?", [r"2033"], []),
    ("c037.jpg", T, "¿Qué grado de protección IP tiene?", [r"ip\s*-?\s*54"], []),
    ("c037.jpg", UN, "¿Qué lectura muestra la pantalla?", [BLANK], [r"\d+[.,]\d+\s*m"]),
    # c039 — four water meters stacked on a white wall, the top one with a blue cap, each with
    # a red-handled valve; all four dials too small to read.
    ("c039.jpg", C, "¿Cuántos medidores de agua hay en la pared?", [n(4)], []),
    ("c039.jpg", S, "¿Cuál de los medidores tiene la tapa azul, el de más arriba o el de "
                    "más abajo?",
     [r"arriba|superior|top|primer"], [r"(es|is) el de (más )?abajo"]),
    ("c039.jpg", AM, "¿Cuánto marca el medidor de agua?",
     [ASKS_WHICH + "|" + CANT_READ + r"|(cuatro|4) medidores"], [r"marca \d{3,}"]),
    # c048 — Malmbergs three-phase induction motor plate: 0.37 kW, 0.5 HP, IP 55, 2730 RPM,
    # 50 Hz, 230/400 V, 1.67/0.97 A. No power factor on the plate.
    ("c048.jpg", R, "¿A cuántas RPM gira el motor?", [r"2730"], []),
    ("c048.jpg", R, "¿Qué potencia tiene el motor en kW?", [dec(0, 37)], []),
    ("c048.jpg", R, "¿Qué corriente consume a 230 V y cuál a 400 V?", [dec(1, 67), dec(0, 97)], []),
    ("c048.jpg", T, "¿Qué grado IP tiene?", [r"ip\s*-?\s*55|\b55\b"], []),
    ("c048.jpg", AB, "¿Cuál es el factor de potencia (coseno fi) del motor?", [NOT_THERE],
     [r"(cos|factor de potencia|power factor)\D{0,25}0[.,]\d"]),
    ("c048.jpg", FP, "¿Por qué este motor es monofásico?",
     [r"trif[aá]sic|three.?phase|tres fases"], [r"es monof[aá]sico porque"]),
    # c051 — Badger Meter water meter "Model 55", Cubic Feet, counter 135732, "0113 64501-116 1"".
    ("c051.jpg", R, "¿En qué unidad mide este medidor de agua?",
     [r"cubic\s*f|pies\s*c[uú]bic|c[uú]bic\w*\s*pies"], []),
    ("c051.jpg", R, "¿Qué número marca el contador?", [r"135\s*\.?\s*732"], []),
    ("c051.jpg", R, "¿De qué marca es el medidor y qué modelo?", [r"badger", r"\b55\b"], []),
    # c060 — Fogg control panel: green lit "E-STOP RESET" top left; CAP FEEDER START and
    # FILLER START green buttons; amber NO CAP lamp; big red mushroom at the bottom.
    ("c060.jpg", R, "¿Qué marca tiene el tablero?", [r"fogg"], []),
    ("c060.jpg", T, "¿Qué dice el cartel del botón verde de arriba a la izquierda?",
     [r"e\s*-?\s*stop", r"reset"], []),
    ("c060.jpg", C, "¿Cuántos pulsadores verdes dicen START?", [n(2)], []),
    ("c060.jpg", ST, "¿De qué color es la luz de NO CAP?",
     [r"naranja|[aá]mbar|amber|orange|amarill"], []),
    # c063 — Baumer dial thermometer 0-150 °C, "Resolution = 2°C", pointer near 30-32 °C.
    ("c063.jpg", R, "¿De qué marca es el termómetro?", [r"baumer"], []),
    ("c063.jpg", T, "¿Cuál es la resolución del termómetro?", [r"\b2\s*°?\s*c?\b|dos grados"], []),
    ("c063.jpg", R, "¿Qué temperatura marca, aproximadamente?", [r"\b(2[89]|3[0-4])\b"], []),
    ("c063.jpg", FP, "¿Qué presión marca este manómetro?",
     [r"term[oó]metr|temperatura|thermometer|°\s*c|grados"], [r"\b(bar|psi|kpa|kg/cm)\b"]),
    # c064 — GCM-306 analog clamp meter.
    ("c064.jpg", R, "¿Qué modelo es este instrumento?", [r"gcm\s*-?\s*306"], []),
    # c065 — Triplett model 321 A.C. clamp milliammeter, scale 0-200 µA DC, knob on OFF.
    ("c065.jpg", R, "¿De qué marca es el instrumento?", [r"triplett"], []),
    ("c065.jpg", R, "¿Hasta cuántos microamperes llega la escala?", [r"\b200\b"], []),
    ("c065.jpg", T, "¿Qué modelo dice en el cuadrante?", [r"321"], []),
    ("c065.jpg", ST, "¿En qué posición está la perilla de escala?", [r"\boff\b|apagad"], []),
    # c066 — Fluke 337 clamp, display 0.0 A DC.
    ("c066.jpg", R, "¿Qué marca y modelo es la pinza?", [r"fluke", r"337"], []),
    ("c066.jpg", R, "¿Qué valor muestra la pantalla?", [r"\b0[.,]0\b|cero"], []),
    # c067 — Fluke 325 clamp in a hand, 0.47 A AC around a black cable.
    ("c067.jpg", R, "¿Qué corriente está midiendo la pinza?", [dec(0, 47)], []),
    ("c067.jpg", R, "¿Qué marca y modelo es la pinza?", [r"fluke", r"325"], []),
    ("c067.jpg", R, "Is it measuring AC or DC?",
     [r"\bac\b|alterna|alternating"], [r"\bit is (measuring )?dc\b"]),

    # k000 — Aclara I-210+ smart meter, 74590 kWh, "85 941 540", "B.R.E.M.C.", inside a
    # Landis & Gyr enclosure (embossed below).
    ("k000.jpg", R, "¿Cuántos kWh marca el medidor?", [r"74\s*\.?\s*590"], []),
    ("k000.jpg", T, "¿Qué número grande dice al lado del modelo?", [r"85\s*941\s*540"], []),
    ("k000.jpg", AM, "¿Qué marca es?", [r"(?=.*aclara)(?=.*landis)|" + ASKS_WHICH], []),
    # k002 — UK Radio Telemeter Series K meter, display showing the DATE 27-08-22; blue button
    # (DISPLAY) above an orange one (BOOST); 2010; serial number deliberately blurred.
    ("k002.jpg", R, "¿Qué está mostrando la pantalla?", [r"27", r"08|agosto|august", r"22"], []),
    ("k002.jpg", T, "¿De qué año es el medidor?", [r"2010"], []),
    ("k002.jpg", S, "¿El botón BOOST es el azul o el naranja?",
     [r"naranja|orange|roj|red"], [r"(es|is) (el )?azul|(is|it's) (the )?blue"]),
    ("k002.jpg", UN, "¿Cuál es el número de serie del medidor?",
     [CANT_READ + r"|tapad|cubiert|blurred|censur|oculto|pixel"], [r"\b\d{6,}\b"]),
    # k011 — Aidon 5530 meter (Göteborg Energi), 9023 kWh, LEDs PWR and RF lit green, SER off,
    # "Brytare max 63A".
    ("k011.jpg", R, "¿Cuántos kWh marca?", [r"9\s*\.?\s*023"], []),
    ("k011.jpg", ST, "¿Qué luces verdes están prendidas en el costado derecho?",
     [r"pwr", r"\brf\b"], [r"\bser\b[^.]{0,20}(encendid|prendid|\bon\b)"]),
    ("k011.jpg", T, "¿De cuántos amperes máximo es el interruptor (brytare)?",
     [r"(?<!\d)63(?!\d)"], []),
    ("k011.jpg", R, "What brand and model is the meter?", [r"aidon", r"5530"], []),
    # k012 — Westinghouse dial meter, 240 volts, 15 amps, type DS, plate "77057", "30 570 643".
    ("k012.jpg", R, "¿De qué marca es el medidor?", [r"westinghouse"], []),
    ("k012.jpg", T, "¿Para qué tensión y corriente es?", [r"240", r"\b15\b"], []),
    ("k012.jpg", T, "¿Qué número está grabado en la chapa de abajo, el más grande?",
     [r"77\s*\.?\s*057"], []),
    ("k012.jpg", C, "¿Cuántos relojitos (diales) tiene para la lectura?", [n(4)], []),
    # k013 — a wall of old meters and junction boxes in India; one boy standing in front, one
    # foot on a box; on the right a box "ROOM NO.99 6th Floor"; below "DANGER A.C 400V
    # SERVICE No.2"; left, light-blue cabinets.
    ("k013.jpg", C, "¿Cuántas personas hay?", [n(1)], []),
    ("k013.jpg", T, "¿Qué número de habitación dice la caja de la derecha?", [r"\b99\b"], []),
    ("k013.jpg", T, "¿Qué tensión dice el cartel de DANGER?", [r"400"], []),
    ("k013.jpg", S, "¿De qué color son los gabinetes de la izquierda?", [r"azul|celeste|blue"], []),
    ("k013.jpg", AB, "¿De qué marca es la bicicleta?", [NOT_THERE], []),
    # k019 — Sangamo KBA 01 key budget meter with a Schlumberger badge; rollers 59711.94 kWh;
    # 240V 50Hz, 20-80A; 1991; "EDF Energy Networks Ltd".
    ("k019.jpg", R, "¿Cuántos kWh enteros marca el contador de rodillos?", [r"59\s*\.?\s*711"], []),
    ("k019.jpg", AM, "¿De qué marca es?", [r"(?=.*sangamo)(?=.*schlumberger)|" + ASKS_WHICH], []),
    ("k019.jpg", T, "¿Qué empresa figura en la etiqueta blanca de abajo?", [r"edf"], []),
    ("k019.jpg", T, "¿De qué año es?", [r"1991"], []),
    # k020 — Siemens single-phase meter, 65521,9 kWh (red tenth), 220 V, 10(60) A, 375 U/kWh,
    # BEWAG 6.047.178.
    ("k020.jpg", R, "¿Cuántos kWh marca, con el decimal?", [r"65\s*\.?\s*521[.,]9"], []),
    ("k020.jpg", T, "¿Cuántas vueltas por kWh dice?", [r"375"], []),
    ("k020.jpg", T, "¿Qué número dice en la chapa de BEWAG?",
     [r"6\s*[.,]?\s*047\s*[.,]?\s*178"], []),
    ("k020.jpg", R, "What voltage and current rating does it show?",
     [r"220", r"10\s*\(\s*60\s*\)|\b10\b.{0,15}\b60\b"], []),
    # k038 — eight beige gas meters in two rows of four on a wall; the two in the left column
    # carry a round "811" sticker.
    ("k038.jpg", C, "¿Cuántos medidores de gas hay?", [n(8)], []),
    ("k038.jpg", C, "¿Cuántos tienen la etiqueta redonda del 811?", [n(2)], []),
    ("k038.jpg", S, "¿En qué columna están los que tienen la etiqueta 811?",
     [r"izquierd|left|primer"], [r"derech|right"]),
    ("k038.jpg", AM, "¿Cuánto marca el medidor?",
     [ASKS_WHICH + "|" + CANT_READ + r"|(ocho|8) medidores"], []),
    # k043 — gas meter, Sensus index plate (B172, 28538G100) on an American Meter Company
    # body (plate "AC-250", 2008), four dials, "CUBIC FEET".
    ("k043.jpg", AM, "¿De qué marca es el medidor?",
     [r"(?=.*sensus)(?=.*american)|" + ASKS_WHICH], []),
    ("k043.jpg", R, "¿En qué unidad mide?", [r"cubic\s*f|pies\s*c[uú]bic|c[uú]bic\w*\s*pies"], []),
    ("k043.jpg", T, "¿Qué modelo dice en la chapa azul?", [r"ac\s*-?\s*250"], []),
    ("k043.jpg", C, "How many large dials does the index have?", [n(4)], []),
    # k045 — gas meter index close-up, slightly blurred: drums 8 0 3 6 followed by printed 0 0;
    # "CUBIC FEET", "TEMP. COMP. 60°F BASE".
    ("k045.jpg", R, "¿Qué número marca el contador?", [r"8\s*\.?\s*036"], []),
    ("k045.jpg", T, "¿A qué temperatura base está compensado?", [r"60\s*°?\s*f|60 grados"], []),
    # k053 — a chrome ball valve with a red butterfly handle lying along the pipe (open), a brass
    # tee on the left, and an out-of-focus dial at the top left.
    ("k053.jpg", R, "¿De qué color es la manija de la llave?", [r"roj|red"], []),
    ("k053.jpg", ST, "¿La llave está abierta o cerrada?",
     [r"abiert|\bopen"], [r"est[aá] cerrad|is closed"]),
    ("k053.jpg", UN, "¿Qué marca el medidor que está arriba a la izquierda?",
     [CANT_READ + r"|fuera de foco|out of focus"], [r"marca \d"]),
    # k057 — railway air-supply pressure gauge 0-1000, pointer near 600; a green inspection
    # sticker covers the lower part; plate "PRESSURE GAUGE OF AIR SUPPLY PIPE".
    ("k057.jpg", R, "¿Cuánto marca la aguja, más o menos?", [r"\b(5[89]\d|6[0-2]\d)\b"], []),
    ("k057.jpg", T, "¿Qué dice la placa de abajo en inglés?",
     [r"pressure\s+gauge", r"air\s+supply"], []),
    ("k057.jpg", R, "¿Hasta cuánto llega la escala?", [r"1\s*\.?\s*000"], []),
    # k061 — touch HMI, tab SENSORS selected; table TEMPERATURE °C and FLOW RATE lpm; flows 1.2,
    # 3.8, 16.8, 18.2; temperatures 27.4, 43.2 (the other two blurred); page "1 OF 3"; red
    # badge "3"; four green toggles.
    ("k061.jpg", T, "¿Qué pestaña está seleccionada en la pantalla?", [r"sensors|sensores"], []),
    ("k061.jpg", T, "¿Qué valores de caudal muestra la columna FLOW RATE?",
     [dec(1, 2), dec(3, 8), dec(16, 8), dec(18, 2)], []),
    ("k061.jpg", T, "¿Qué temperaturas muestran las dos primeras filas?",
     [dec(27, 4), dec(43, 2)], []),
    ("k061.jpg", T, "¿Qué página es, de cuántas?", [r"\b1\b.{0,12}\b3\b"], []),
    ("k061.jpg", AB, "¿Qué presión muestra la pantalla?",
     [NOT_THERE + r"|solo (muestra )?temperatura"], [r"\b(bar|psi|kpa)\b"]),
    # k067 — stainless process skid: ammeter (A, 0-200), two PID controllers; pilot lights red,
    # red, green, orange, orange (+ a red buzzer); OFF/ON selectors; a person in a hair net.
    ("k067.jpg", C, "¿Cuántas luces piloto naranjas hay?", [n(2)], []),
    ("k067.jpg", S, "¿De qué color es la luz que está justo a la izquierda de las naranjas?",
     [r"verde|green"], []),
    ("k067.jpg", R, "¿Hasta cuántos amperes llega la escala del amperímetro?", [r"\b200\b"], []),
    ("k067.jpg", S, "¿Qué tiene en la cabeza la persona de la derecha?",
     [r"cofia|gorr|red (para el|de) pelo|redecilla|hair\s*net|malla|cap\b"], []),
    # k098 — control room: two seated operators facing the screens; men in orange vests and
    # white hard hats standing on the right (4 clear, maybe a 5th hidden: not counted); orange
    # posts with "DANGER KEEP OUT" tape in front.
    ("k098.jpg", C, "¿Cuántas personas están sentadas frente a las pantallas?", [n(2)], []),
    ("k098.jpg", T, "What does the tape in the foreground say?", [r"danger", r"keep\s*out"], []),
    ("k098.jpg", AB, "¿De qué marca es el autoelevador?", [NOT_THERE], []),
    # k103 — CCTV room: Samsung monitors; one man in a yellow hi-vis shirt and cap at the desk;
    # left screen stamped "[001] EE1 Cam 1  5 Oct 2015 11:08:41".
    ("k103.jpg", R, "¿De qué marca son los monitores?", [r"samsung"], []),
    ("k103.jpg", T, "¿Qué fecha muestra la cámara de la pantalla de la izquierda?",
     [r"\b5\b", r"oct", r"2015"], []),
    ("k103.jpg", C, "¿Cuántas personas hay?", [n(1)], []),
    ("k103.jpg", R, "What color is the man's shirt?",
     [r"yellow|amarill|lime|neon|green|verde|fl[uú]o"], []),
    # k107 — car fuse box: top a green 30, a red 10 and a purple 30 cartridge; middle row 20 25
    # 20 10 20 15 10 10 10 20; bottom row 20 15 10 15 10 10 15 (clear) 10 10.
    ("k107.jpg", C, "¿Cuántos fusibles amarillos hay?", [n(5)], []),
    ("k107.jpg", C, "¿Cuántos fusibles azules de 15 hay?", [n(4)], []),
    ("k107.jpg", R, "¿De cuántos amperes es el fusible gris de la fila del medio?",
     [r"\b25\b"], []),
    # k114 — CHINT consumer unit: NH4-125 32A isolator, SDM120 kWh meter with a blank display,
    # two NB1L C13 0.03A RCBOs and one NB1-63 C6; every lever down on O-OFF.
    ("k114.jpg", R, "¿De qué marca son los interruptores?", [r"chi?nt"], []),
    ("k114.jpg", T, "¿De cuántos amperes es el interruptor de la derecha de todo?",
     [r"\b6\b|c\s*6"], []),
    ("k114.jpg", C, "¿Cuántos interruptores diferenciales NB1L hay?", [n(2)], []),
    ("k114.jpg", ST, "¿Los interruptores están encendidos o apagados?",
     [r"apagad|\boff\b|abajo|desconect"], [r"est[aá]n (todos )?(encendid|prendid)"]),
    ("k114.jpg", UN, "¿Cuántos kWh marca el medidor de energía?",
     [BLANK + "|" + CANT_READ], [r"\d+[.,]\d+\s*kwh"]),
    # k127 — ABB F204 A 40A 0.03A residual-current breaker labelled Q82 + six S201 B16 labelled
    # F21..F26.
    ("k127.jpg", C, "¿Cuántos interruptores termomagnéticos hay al lado del diferencial?",
     [n(6)], []),
    ("k127.jpg", T, "¿Qué sensibilidad tiene el diferencial?", [r"0[.,]03\s*a|30\s*ma"], []),
    ("k127.jpg", S, "¿Qué etiqueta tiene el último interruptor de la derecha?", [r"f\s*26"], []),
    ("k127.jpg", AM, "¿De cuántos amperes es?",
     [r"(?=.*(?<!\d)40(?!\d))(?=.*(?<!\d)16(?!\d))|" + ASKS_WHICH], []),
    ("k127.jpg", AB, "¿Cuánto marca el medidor de energía de este tablero?",
     [NOT_THERE], [r"\d+[.,]?\d*\s*kwh"]),
    # k154 — car fuse box in three columns; blue 15 A: three on the left, one on the right;
    # one yellow 20; the rest red 10.
    ("k154.jpg", C, "¿Cuántos fusibles azules hay?", [n(4)], []),
    ("k154.jpg", R, "¿De cuántos amperes es el fusible amarillo?", [r"\b20\b"], []),
    ("k154.jpg", S, "¿En qué columna está el fusible amarillo?",
     [r"medio|centr|middle|segunda|second"], []),
    # k183 — two workers in yellow hard hats smiling at a binder; the woman's hat has a head
    # lamp on a "STREAMLIGHT" strap; both wear safety glasses and grey gloves; more workers
    # behind.
    ("k183.jpg", T, "¿Qué marca dice en la banda de la linterna del casco?", [r"streamlight"], []),
    ("k183.jpg", R, "¿De qué color son los cascos?", [r"amarill|yellow"], []),
    ("k183.jpg", FP, "¿Por qué la mujer no tiene casco?",
     [r"(s[ií]|yes).{0,40}(tiene|lleva|usa|has|wear)|tiene (puesto )?(un )?casco|"
      r"lleva (puesto )?(un )?casco|is wearing"],
     [r"no tiene casco porque"]),
    ("k183.jpg", R, "What are they wearing on their hands?", [r"glove|guante"], []),
    # k187 — three valve handwheels on a rusty manifold: blue (left), red (middle), green (right).
    # No gauge anywhere.
    ("k187.jpg", S, "¿De qué color es el volante de la válvula del medio?", [r"roj|red"], []),
    ("k187.jpg", S, "¿Y el de la derecha?", [r"verde|green"], []),
    ("k187.jpg", AB, "¿Qué presión marca el manómetro de la cañería?",
     [NOT_THERE], [r"\d+\s*(bar|psi|kpa)"]),
    ("k187.jpg", AM, "¿De qué color es la válvula?",
     [r"(?=.*azul|.*blue)(?=.*roj|.*red)(?=.*verde|.*green)|" + ASKS_WHICH], []),
    ("k187.jpg", FP, "¿Por qué la válvula del medio está pintada de azul?",
     [r"roj|red"], [r"azul porque"]),
    # k248 — panel volt/amp/frequency meter: 226 V, 47.3 A, 50 Hz; a current-transformer
    # toroid below it.
    ("k248.jpg", AM, "¿Cuánto marca?", [r"226", dec(47, 3), r"\b50\b"], []),
    ("k248.jpg", R, "¿Qué corriente marca?", [dec(47, 3)], []),
    ("k248.jpg", S, "¿Qué hay adelante del display, abajo?",
     [r"toroid|transformador de corriente|bobina|anillo|ferrita|n[uú]cleo|coil|"
      r"current transformer|\bct\b"],
     []),
    # k252 — two Soviet M265M microammeters (µA, 1977); left scale 100-0-100 with the pointer at
    # the centre zero, right scale 0-100.
    ("k252.jpg", R, "¿Qué modelo son los instrumentos y en qué unidad miden?",
     [r"m\s*265\s*m", r"µ\s*a|μ\s*a|micro\s*amp|microamper"], []),
    ("k252.jpg", ST, "¿Dónde está la aguja del instrumento de la izquierda?",
     [r"\b0\b|cero|centro|medio|middle|center|zero"], []),
    ("k252.jpg", T, "¿De qué año son?", [r"1977"], []),
    ("k252.jpg", R, "How do the scales of the two meters differ?",
     [r"cent(er|re)|middle|zero.{0,30}(middle|cent)|both sides|-\s*100|100\s*-\s*0\s*-\s*100"], []),
    # k328 — wall extinguisher under a red triangle "ALAT PEMADAM API"; to its right a
    # "P.A.S.S" instruction card (Pull the Pin / Aim at the base of fire / Squeeze / Sweep).
    ("k328.jpg", T, "¿Qué dice el cartel triangular rojo?", [r"alat", r"pemadam", r"api"], []),
    ("k328.jpg", S, "¿Qué hay a la derecha del matafuego?",
     [r"p\.?\s*a\.?\s*s\.?\s*s|instruccion|cartel|instructions"], []),
    ("k328.jpg", T, "Según el cartel en inglés, ¿qué significa la A de PASS?", [r"aim"], []),
    # k330 — two extinguishers in a corner, "5TH FLOOR" sign (the hose hides part of it); a green
    # "CUSTOMER" tag on the left one, a yellow tag with "2018" on the right one.
    ("k330.jpg", C, "¿Cuántos matafuegos hay?", [n(2)], []),
    ("k330.jpg", T, "¿Qué año dice la etiqueta amarilla?", [r"2018"], []),
    ("k330.jpg", T, "¿Qué piso dice el cartel?", [r"\b5|quint|fifth"], []),
    ("k330.jpg", T, "¿Qué palabra dice arriba de la etiqueta verde?", [r"customer"], []),
    # k331 — corridor: "CELLAR" sign; two extinguishers hung one behind the other on the left;
    # a fire-hose cabinet with a glass door.
    ("k331.jpg", T, "¿Qué dice el cartel de la pared?", [r"cellar"], []),
    ("k331.jpg", S, "¿Qué hay detrás de la puerta de vidrio?", [r"manguera|hose"], []),
    ("k331.jpg", C, "¿Cuántos matafuegos hay colgados?", [n(2)], []),
    # k347 — tiled wall: extinguisher with its sign (left), a spinal board/stretcher (middle),
    # an "AED" defibrillator box (right).
    ("k347.jpg", S, "¿Qué hay a la derecha de la camilla?",
     [r"desfibrilador|defibrillator|\bdea\b|\baed\b"], []),
    ("k347.jpg", S, "¿Y a la izquierda?", [r"matafuego|extintor|extinguisher|extinguidor"], []),
    ("k347.jpg", AB, "¿Qué número de teléfono de emergencia dice en la pared?", [NOT_THERE],
     [r"(pared|wall|cartel|dice|figura|se lee)[^.]{0,40}\b\d{3,}\b"]),
    # k349 — two red extinguishers "20型", red sign "FIRE EXTINGUISHER 滅火器", "6F-4" handwritten
    # on the left one, phone 27886665 on both labels.
    ("k349.jpg", C, "How many fire extinguishers are there?", [n(2)], []),
    ("k349.jpg", T, "¿Qué escribieron a mano en el matafuego de la izquierda?",
     [r"6\s*f\s*-?\s*4"], []),
    ("k349.jpg", T, "¿Qué número de teléfono dice la etiqueta verde?", [r"2788\s*-?\s*6665"], []),
    ("k349.jpg", FP, "¿Por qué hay tres matafuegos?", [n(2)], [r"hay tres (matafuegos )?porque"]),
    # k391 — Irish road signs: white "Maigh Nuad / 6 MAYNOOTH / R148", white "INTEL" with a
    # factory symbol, brown "Teach Carton / CARTON HOUSE"; a round blue sign with a white arrow
    # pointing left; no Dublin anywhere.
    ("k391.jpg", T, "¿A cuántos kilómetros queda Maynooth?", [r"\b6\b|seis"], []),
    ("k391.jpg", T, "¿Qué ruta indica?", [r"r\s*148"], []),
    ("k391.jpg", R, "¿Qué dice el cartel marrón?", [r"carton\s*house"], []),
    ("k391.jpg", S, "¿Hacia dónde apunta la flecha del cartel azul redondo?",
     [r"izquierd|left"], [r"hacia la derecha|to the right"]),
    ("k391.jpg", AB, "¿A cuántos kilómetros queda Dublín según el cartel?",
     [NOT_THERE], [r"dubl[ií]n (est[aá]|queda|is) a? ?\d"]),
    # k393 — Hungarian street: one pole with two triangular warning signs (traffic lights above,
    # "!" below); small white signs "RÉGÉSZETI PARK" and "MATRICA MÚZEUM" on the left; a
    # traffic light ahead lit green; two red-and-white striped chimneys; no speed-limit sign.
    ("k393.jpg", C, "¿Cuántas señales triangulares hay en el poste?", [n(2)], []),
    ("k393.jpg", T, "¿Qué dicen los carteles blancos chiquitos de la izquierda?",
     [r"r[eé]g[eé]szeti", r"matrica"], []),
    ("k393.jpg", ST, "¿De qué color está el semáforo de la calle, el que se ve al fondo?",
     [r"verde|green"], []),
    ("k393.jpg", C, "¿Cuántas chimeneas a rayas se ven?", [n(2)], []),
    ("k393.jpg", AB, "¿Qué velocidad máxima indica el cartel?",
     [NOT_THERE], [r"\b(30|40|50|60|70|80|90)\s*(km|kilómetros)"]),
    # k409 — garment factory: many women at sewing machines with face masks and light-blue
    # uniforms, dark-blue fabric; date stamp "07/20/2006".
    ("k409.jpg", T, "¿Qué fecha dice la foto abajo a la derecha?",
     [r"0?7\s*/\s*20\s*/\s*2006|20.{0,12}(julio|july).{0,8}2006|(july|julio)\s*20.{0,4}2006"], []),
    ("k409.jpg", R, "What are the workers wearing on their faces?",
     [r"barbijo|tapabocas|mascarilla|m[aá]scara|mask|cubrebocas"], []),
    ("k409.jpg", R, "¿De qué color es la tela que están cosiendo?", [r"azul|navy|blue"], []),
    # k421 — warehouse seen from above: stacked pallets of cans; five people standing on the
    # floor; a green forklift at the back right.
    ("k421.jpg", C, "¿Cuántas personas hay en el piso?", [n(5)], []),
    ("k421.jpg", S, "¿Qué vehículo hay al fondo a la derecha?",
     [r"autoelevador|montacarga|forklift|clark|zorra|elevador"], []),
    ("k421.jpg", R, "¿De qué color es el autoelevador?", [r"verde|green"], []),
    # k425 — loading dock: Komatsu forklift "LOGISTICS TEAM" in front of a container on a PANUS
    # trailer; a man inside the container; container number partly hidden by the mast.
    ("k425.jpg", R, "¿De qué marca es el autoelevador?", [r"komatsu"], []),
    ("k425.jpg", T, "¿Qué dice en el costado del autoelevador?", [r"logistics\s*team"], []),
    ("k425.jpg", S, "¿Hay alguien dentro del contenedor?",
     [r"\bs[ií]\b|hay (un|una)|yes|there is"], [r"^\s*no\b"]),
    ("k425.jpg", UN, "¿Cuál es el número completo del contenedor?",
     [CANT_READ + r"|parcial|incomplet|no (se ve|est[aá]) complet|tapad|cortad|ocult"],
     [r"[a-z]{4}\s*\d{7}"]),
    # k430 — green Clark forklift, side view, nobody in the seat.
    ("k430.jpg", R, "¿De qué marca es el autoelevador?", [r"clark"], []),
    ("k430.jpg", AB, "¿Quién está manejando el autoelevador?",
     [NOT_THERE + r"|vac[ií]o|empty|nadie"], []),
    # k440 — two yellow Yale forklifts with "BRIGGS" decals, each with a driver; a third person
    # walking in the background; the right one's mast reads "519952".
    ("k440.jpg", R, "¿De qué marca son los autoelevadores?", [r"yale"], []),
    ("k440.jpg", C, "¿Cuántas personas hay en total?", [n(3)], []),
    ("k440.jpg", T, "¿Qué número tiene el mástil del autoelevador de la derecha?",
     [r"519\s*952"], []),
    ("k440.jpg", FP, "¿Por qué los autoelevadores Toyota son amarillos?",
     [r"yale"], [r"toyota (es|son|usa|pinta)"]),
    # k475 — elevator lobby: sign "14TH FLOOR / IN CASE OF FIRE USE STAIRS DO NOT USE ELEVATOR /
    # 10 People Maximum Per Elevator"; a destination keypad "Enter floor number".
    ("k475.jpg", T, "¿En qué piso estamos?", [r"\b14|catorce|fourteenth"], []),
    ("k475.jpg", R, "¿Qué pide la pantalla del teclado?",
     [r"floor\s*number|n[uú]mero de piso|piso"], []),
    ("k475.jpg", FP, "¿Por qué dice que entran 12 personas por ascensor?",
     [r"\b10\b|diez"], [r"(dice|entran) 12 (personas )?porque"]),
    ("k475.jpg", R, "What should you do in case of fire, according to the sign?",
     [r"stairs|escalera"], []),
    # k516 — White-Rodgers thermostat, LCD "76 °F" and "68" with a flame icon; FAN ON/AUTO and
    # SYSTEM COOL/OFF/HEAT sliders, system on HEAT.
    ("k516.jpg", R, "¿De qué marca es el termostato?", [r"white\s*-?\s*rodgers"], []),
    ("k516.jpg", AM, "¿Qué temperatura marca?", [r"(?=.*\b76\b)(?=.*\b68\b)|" + ASKS_WHICH], []),
    ("k516.jpg", ST, "¿En qué modo está el sistema: frío, apagado o calor?",
     [r"calor|calef|heat"], [r"(est[aá]|is) (en )?(modo )?(fr[ií]o|cool)\b"]),
    ("k516.jpg", FP, "¿Por qué está en modo COOL?", [r"calor|calef|heat"], [r"cool porque"]),
    # k527 — OMV pump number 4 at night, LCD 000000 Forint / 0000,00 Liter / 000,0 Forint/Liter;
    # notice: 100 liters per transaction into cars, 20 liters into canisters.
    ("k527.jpg", R, "¿Qué número de surtidor es?", [r"\b4\b|cuatro"], []),
    ("k527.jpg", R, "¿Cuántos litros se cargaron?",
     [r"\b0+([.,]0+)?\b|cero|ning|nada"], [r"\b[1-9]\d*[.,]\d+\s*l"]),
    ("k527.jpg", T, "¿Cuál es el máximo de litros por carga para autos según el aviso?",
     [r"\b100\b"], []),
    ("k527.jpg", R, "What brand of fuel is advertised?", [r"\bomv\b|maxx\s*motion"], []),
    # k530 — US pump: THIS SALE $ 8.34, GALLONS 2.076; "DANGER ... EXTREMELY FLAMMABLE".
    ("k530.jpg", AM, "¿Cuánto marca?", [dec(8, 34), dec(2, "076")], []),
    ("k530.jpg", R, "¿Cuántos galones se cargaron?", [dec(2, "076")], []),
    ("k530.jpg", T, "What does the red oval sign say?", [r"danger"], []),
    # k557 — traffic light in an arch: red LEFT ARROW lit (top left) and amber lit (middle
    # right); nothing green lit.
    ("k557.jpg", ST, "¿Qué luces del semáforo están encendidas?",
     [r"roj|red", r"amarill|[aá]mbar|amber|yellow|naranja"], [r"verde (est[aá] )?encendid"]),
    ("k557.jpg", ST, "¿Qué forma tiene la luz roja?", [r"flecha|arrow"], []),
    ("k557.jpg", FP, "¿Por qué está en verde el semáforo?",
     [r"no (est[aá]|hay).{0,40}verde|not green|isn'?t green|roj|amarill|[aá]mbar|amber|red\b"],
     [r"est[aá] en verde (porque|para|ya que)"]),
    ("k557.jpg", AB, "¿Cuántos segundos faltan según el contador del semáforo?",
     [NOT_THERE], [r"\b\d+\s*segundos"]),
    # k587 — Buenos Aires: street sign "Tucumán 1100-1200 Comuna 1" with an arrow to the right;
    # a red-and-white "Escolares" bus; pedestrian light showing the red figure.
    ("k587.jpg", T, "¿Qué numeración de la calle Tucumán indica el cartel?",
     [r"1\.?100", r"1\.?200"], []),
    ("k587.jpg", T, "¿Qué comuna dice?", [r"comuna\s*1\b|\b1\b"], []),
    ("k587.jpg", ST, "¿De qué color está el semáforo peatonal?",
     [r"roj|red"], [r"(est[aá]|is) (en )?(verde|green)"]),
    ("k587.jpg", R, "¿Qué dice en el frente del micro?", [r"escolares"], []),
    ("k587.jpg", S, "Which way does the arrow on the street sign point?", [r"derech|right"], []),
    # k609 — cluttered office desk: an Epson projector with a printed page and a pen on top; one
    # monitor; two white desk phones plus a cordless handset (so "how many phones" is not
    # asked); speakers; no people.
    ("k609.jpg", R, "¿De qué marca es el proyector?", [r"epson"], []),
    ("k609.jpg", S, "¿Qué hay arriba del proyector?",
     [r"hoja|papel|paper|document|lapicera|bol[ií]grafo|pen"], []),
    ("k609.jpg", UN, "¿Qué dice la hoja que está arriba del proyector?", [CANT_READ], []),
    ("k609.jpg", AB, "¿Cuántas personas hay en la oficina?",
     [NOT_THERE + r"|\b(0|cero)\b"], [r"hay (una|dos|\d) persona"]),
    # k610 — iMac with its screen off (only reflections), a grey funnel in front of it, an
    # orange Brother label maker at the bottom right, a binder "LSJ & GSJ BOAT LOG 2019".
    ("k610.jpg", R, "¿Qué marca es la computadora?", [r"apple|imac|mac\b"], []),
    ("k610.jpg", S, "What is in front of the monitor, in the middle?", [r"embudo|funnel"], []),
    ("k610.jpg", S, "¿Qué es el aparato naranja de abajo a la derecha?",
     [r"rotulador|etiquetador|label|impresora de etiquetas|brother|p-?touch"], []),
    ("k610.jpg", T, "¿Qué dice la carpeta blanca de la izquierda?", [r"boat\s*log", r"2019"], []),
    ("k610.jpg", UN, "¿Qué hora marca la pantalla de la computadora?",
     [BLANK + r"|reflej|reflect|negra|black"],
     [r"(pantalla|screen|marca|muestra)[^.]{0,30}\b\d{1,2}:\d{2}\b"]),
    # k612 — dim office: split air conditioner high on the back wall; a cabinet with a red
    # "KEEP OUT" sign; an iMac; a framed document too small and dark to read.
    ("k612.jpg", T, "¿Qué dice el cartel rojo del gabinete de la izquierda?", [r"keep\s*out"], []),
    ("k612.jpg", S, "¿Qué hay en la pared del fondo, arriba al centro?",
     [r"aire acondicionado|split|air condition|\bac\b"], []),
    ("k612.jpg", UN, "¿Qué dice el documento enmarcado de la pared?", [CANT_READ], []),
    # k624 — Alfa Romeo speedometer at night: needle on 0 km/h, odometer 123456 km, 16 °C.
    ("k624.jpg", R, "¿Cuántos kilómetros tiene el auto?", [r"123\s*\.?\s*456"], []),
    ("k624.jpg", R, "¿Qué temperatura marca?", [r"\b16\b"], []),
    ("k624.jpg", R, "What make is the car?", [r"alfa\s*romeo"], []),
    ("k624.jpg", FP, "¿Por qué va a 120 km/h?",
     [r"\b0\b|cero|detenid|parad|quiet|stopped|not moving"], [r"va a 120 porque"]),
    ("k624.jpg", AB, "¿Cuánta nafta le queda?",
     [NOT_THERE], [r"\b\d+\s*(%|litros|l\b)|(medio|un cuarto|lleno)"]),
    # k647 — Japanese cluster at night: OUT SIDE 18 °C, ODO 164335, gear row P R N D with P framed.
    ("k647.jpg", R, "¿Cuántos kilómetros marca el odómetro?", [r"164\s*\.?\s*335"], []),
    ("k647.jpg", R, "¿Qué temperatura exterior marca?", [r"\b18\b"], []),
    ("k647.jpg", ST, "¿En qué marcha está la caja?",
     [r"\bp\b|park|estacionamiento|parking"], [r"\b(en|in) (d|drive|r|reverse|n|neutral)\b"]),
    # k671 — enamel sign: "Meadowside Lane 172024 / 6 600 Volts Electricity Substation /
    # Emergency 0161 228 2628 / enw Electricity North West Limited". No e-mail.
    ("k671.jpg", R, "¿De cuántos voltios es la subestación?", [r"6\s*[.,]?\s*600"], []),
    ("k671.jpg", R, "¿Qué teléfono de emergencia tiene?", [r"0161\s*228\s*2628"], []),
    ("k671.jpg", R, "What is the substation's name and number?", [r"meadowside", r"172024"], []),
    ("k671.jpg", AB, "¿Cuál es el mail de contacto del cartel?", [NOT_THERE], [r"@"]),
    ("k671.jpg", FP, "¿Por qué la subestación es de 13.200 voltios?",
     [r"6\s*[.,]?\s*600"], [r"(?<!no )es de 13"]),
    # k674 — kitchen scale with two carrots: 112 g, 23.0 °C, 00:00.
    ("k674.jpg", R, "¿Cuánto pesan las zanahorias?", [r"\b112\b"], []),
    ("k674.jpg", C, "¿Cuántas zanahorias hay?", [n(2)], []),
    ("k674.jpg", T, "¿Qué temperatura muestra la balanza?", [dec(23, 0) + r"|\b23\b"], []),
    ("k674.jpg", AB, "¿Cuánto cuestan según la balanza?",
     [NOT_THERE + r"|no (es|se trata de) una balanza (comercial|de precio)"], [r"\$\s*\d"]),
    # k677 — DGT60 weight indicator, 100 kg in red LEDs; keys ZERO, TARE, MODE, PRINT, C.
    ("k677.jpg", R, "¿Cuánto marca el indicador de peso?", [r"\b100\b", r"kg|kilo"], []),
    ("k677.jpg", S, "Which button is between TARE and PRINT?", [r"mode"], []),
    ("k677.jpg", T, "¿Qué modelo es el indicador?", [r"dgt\s*-?\s*60"], []),
    ("k677.jpg", AB, "¿Qué temperatura muestra el indicador?", [NOT_THERE], [r"\d+\s*°"]),
    # k711 — Tektronix 475A oscilloscope showing a clipped sine; serial plate 711177.
    ("k711.jpg", R, "¿Qué marca y modelo es el osciloscopio?", [r"tektronix|tek\b", r"475"], []),
    ("k711.jpg", R, "What waveform is shown on the screen?",
     [r"senoidal|sinusoidal|seno|sine"], []),
    ("k711.jpg", T, "¿Qué número de serie dice abajo a la derecha?", [r"711\s*177"], []),
]


def matches(text, must, must_not):
    """Right when every `must` regex matches and no `must_not` one does."""
    flags = re.IGNORECASE | re.DOTALL
    return (all(re.search(p, text, flags) for p in must)
            and not any(re.search(p, text, flags) for p in must_not))


def is_english(question):
    return not question.lstrip().startswith("¿")


def summarize(results):
    """Per question type and overall: answers right, questions, always / never right."""
    rows = {}
    for kind in (*KINDS, "english", "all"):
        if kind == "all":
            rs = results
        elif kind == "english":
            rs = [r for r in results if is_english(r["question"])]
        else:
            rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        hits = sum(r["hits"] for r in rs)
        tries = sum(r["tries"] for r in rs)
        rows[kind] = {"questions": len(rs), "answers": tries, "right": hits,
                      "pct": round(100 * hits / max(tries, 1), 1),
                      "always": sum(r["hits"] == r["tries"] for r in rs),
                      "never": sum(r["hits"] == 0 for r in rs)}
    firsts = [a["first_s"] for r in results for a in r["answers"] if a.get("first_s")]
    totals = [a["total_s"] for r in results for a in r["answers"] if a.get("total_s")]
    timing = {"first_word_p50_s": round(statistics.median(firsts), 3) if firsts else None,
              "answer_p50_s": round(statistics.median(totals), 3) if totals else None}
    return rows, timing


def print_summary(label, rows, timing):
    print(f"\n{label}")
    print(f"{'type':14} {'questions':>9} {'right':>12} {'%':>6} {'always':>7} {'never':>6}")
    for kind, s in rows.items():
        print(f"{kind:14} {s['questions']:9} {s['right']:5}/{s['answers']:<6} {s['pct']:6.1f} "
              f"{s['always']:7} {s['never']:6}")
    print(f"time to first word p50: {timing['first_word_p50_s']} s; "
          f"whole answer p50: {timing['answer_p50_s']} s")


def save(label, payload):
    out = results_path(f"vlm_field_check_{label}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved: {out}")
    return out


def rescore(paths):
    """Score saved answers again with the regexes in CASES as they are now."""
    by_key = {(c[0], c[2]): c for c in CASES}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        changed = 0
        dropped = [r for r in run["results"] if (r["photo"], r["question"]) not in by_key]
        for r in dropped:
            print(f"  dropped (no longer in CASES): {r['photo']} {r['question']}")
        run["results"] = [r for r in run["results"] if (r["photo"], r["question"]) in by_key]
        for r in run["results"]:
            case = by_key[(r["photo"], r["question"])]
            _, kind, _, must, must_not = case
            r["kind"], r["must"], r["must_not"] = kind, must, must_not
            for a in r["answers"]:
                ok = matches(a["answer"], must, must_not)
                changed += ok != a["ok"]
                a["ok"] = ok
            r["hits"] = sum(a["ok"] for a in r["answers"])
        rows, timing = summarize(run["results"])
        run["summary"], run["timing"] = rows, timing
        run["rescored"] = datetime.now().isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print_summary(f"{os.path.basename(path)} ({changed} answers changed verdict)",
                      rows, timing)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="the VLM server (default: what the app uses)")
    ap.add_argument("--label", default=None, help="name for the results file "
                    "(default: the model alias the server reports)")
    ap.add_argument("--robot", default="g1", help='whose identity answers; "none" for none')
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--size", type=int, default=1152, help="longest side sent (0 = as is)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=[], help="photo names to keep (e.g. f012)")
    ap.add_argument("--kind", nargs="*", default=[], choices=KINDS,
                    help="question types to keep")
    ap.add_argument("--rescore", nargs="+", metavar="JSON",
                    help="score saved runs again with the current regexes (no model asked)")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore)
        return 0

    robot = None if args.robot.lower() in ("none", "") else args.robot
    cfg = vlm_common.load_config()
    url = args.url or os.environ.get("VLM_URL") or cfg.get("llamacpp_url",
                                                            vlm_common.LLAMACPP_HOST)
    served = vlm_common.list_llamacpp_models(url)
    label = args.label or (served[0] if served else "unknown")
    label = re.sub(r"[^\w.-]+", "-", label)

    cases = [c for c in CASES
             if (not args.only or os.path.splitext(c[0])[0] in args.only)
             and (not args.kind or c[1] in args.kind)]
    results = []
    for photo, kind, question, must, must_not in cases:
        path = os.path.join(PHOTO_DIR, photo)
        answers = []
        for _ in range(args.tries):
            try:
                text, first, total = ask(cfg, url, frame_b64(path, args.size), question, robot,
                                         args.max_tokens)
            except Exception as e:  # one failed request counts as a wrong answer
                text, first, total = f"[error] {e}", None, None
            answers.append({"answer": text, "ok": matches(text, must, must_not),
                            "first_s": first, "total_s": total})
        hits = sum(a["ok"] for a in answers)
        results.append({"photo": photo, "kind": kind, "question": question, "must": must,
                        "must_not": must_not, "hits": hits, "tries": args.tries,
                        "answers": answers})
        wrong = next((a["answer"] for a in answers if not a["ok"]), "")
        print(f"{hits}/{args.tries}  {photo:9} {kind:13} {question[:55]:55} "
              f"{('| ' + wrong.replace(chr(10), ' ')[:80]) if wrong else ''}", flush=True)

    rows, timing = summarize(results)
    print_summary(label, rows, timing)
    save(label, {"label": label, "served": served, "url": url, "robot": robot,
                 "tries": args.tries, "size": args.size, "max_tokens": args.max_tokens,
                 "date": datetime.now().isoformat(timespec="seconds"),
                 "summary": rows, "timing": timing, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
