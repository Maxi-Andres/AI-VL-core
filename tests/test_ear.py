"""The robot's ear (src/ear_common.py): utterances out of a live stream, "Bender" commands out
of utterances. Made-up audio and a fake transcriber: no model, no robot.

Every test here names the defect it catches. The levels mimic the G1 measured 2026-10-19: fan
noise at an RMS of ~0.008, speech at 0.05-0.11.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ear_common as ec

PACKET = 2560                      # samples per robot packet (160 ms)


def tone(seconds, level, hz=220.0):
    """A sine at RMS `level` — "speech" when loud, "fan noise" when at ~0.008."""
    n = int(seconds * ec.RATE)
    amp = level * math.sqrt(2) * 32767
    return struct.pack(f"<{n}h", *(int(amp * math.sin(2 * math.pi * hz * i / ec.RATE))
                                   for i in range(n)))


def noise(seconds):
    return tone(seconds, 0.008, hz=97.0)


def feed(ear, pcm):
    events = []
    for i in range(0, len(pcm), PACKET * 2):
        events += ear.feed(pcm[i:i + PACKET * 2])
    return events


def feed_timed(ear, pcm, clock):
    """feed(), with the clock running as the audio would: 160 ms a packet."""
    events = []
    for i in range(0, len(pcm), PACKET * 2):
        clock.t += PACKET / ec.RATE
        events += ear.feed(pcm[i:i + PACKET * 2])
    return events


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def ear_saying(*texts, clock=None):
    said = iter(texts)
    heard = []

    def transcribe(wav):
        heard.append(len(wav))
        return next(said, "")
    return ec.Ear(transcribe, clock=clock or Clock()), heard


# --------------------------------------------------------------------------- #
# Segmenter
# --------------------------------------------------------------------------- #
def test_fan_noise_alone_is_never_an_utterance():
    """Catches: the G1's fans transcribed over and over (Whisper turns noise into text)."""
    ear, heard = ear_saying("x")
    assert feed(ear, noise(10.0)) == [] and heard == []


def test_speech_between_noise_is_one_utterance_with_its_start_kept():
    ear, heard = ear_saying("Bender, qué ves")
    events = feed(ear, noise(2.0) + tone(1.2, 0.08) + noise(1.5))
    assert [e["type"] for e in events] == ["speech", "heard", "command"]
    # 1.2 s of voice + the pre-roll before it + the silence that ended it, as a WAV
    assert len(heard) == 1 and heard[0] > 44 + int(1.2 * ec.RATE) * 2


def test_an_endless_talker_is_cut_at_max_seconds():
    ear, heard = ear_saying("uno", "dos")
    feed(ear, noise(1.0) + tone(ec.MAX_S + 3.0, 0.08))
    assert len(heard) == 1, "one utterance closed at MAX_S, the rest still open"


def test_a_click_is_not_an_utterance():
    ear, heard = ear_saying("x")
    feed(ear, noise(1.0) + tone(0.12, 0.1) + noise(1.5))
    assert heard == []


# --------------------------------------------------------------------------- #
# Wake word
# --------------------------------------------------------------------------- #
def test_bender_then_the_request_is_a_command_at_once():
    ear, _ = ear_saying("Bender, ¿qué ves?")
    events = feed(ear, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
    assert events[-1] == {"type": "command", "text": "qué ves"}


def test_bender_alone_waits_for_the_next_utterance():
    clock = Clock()
    ear, _ = ear_saying("Bender.", "caminá un metro para adelante", clock=clock)
    first = feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    assert first[-1] == {"type": "wake"}
    clock.t += 2.0
    second = feed(ear, tone(1.5, 0.08) + noise(1.2))
    assert second[-1] == {"type": "command", "text": "caminá un metro para adelante"}


def test_the_robots_own_si_is_not_the_question():
    """Catches: the ear, no longer muted for the "¿Sí?" acknowledgement, taking its own "Sí"
    as the request — or losing the question that ran into it."""
    clock = Clock()
    ear, _ = ear_saying("Bender.", "¿Sí?", "Sí, ¿qué ves?", clock=clock)
    feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    clock.t += 1.0
    echo = feed(ear, tone(0.6, 0.08) + noise(1.2))
    assert [e["type"] for e in echo] == ["speech", "heard"]          # still waiting
    clock.t += 1.0
    asked = feed(ear, tone(1.5, 0.08) + noise(1.2))
    assert asked[-1] == {"type": "command", "text": "qué ves?"}


def test_a_request_that_starts_with_si_word_is_kept():
    clock = Clock()
    ear, _ = ear_saying("Bender", "siempre hay alguien en la puerta?", clock=clock)
    feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    clock.t += 1.0
    asked = feed(ear, tone(1.5, 0.08) + noise(1.2))
    assert asked[-1] == {"type": "command", "text": "siempre hay alguien en la puerta?"}


def test_the_follow_up_window_closes():
    """Catches: "Bender" said once and every later conversation in the room taken as orders."""
    clock = Clock()
    ear, _ = ear_saying("Bender", "y después fuimos a comer", clock=clock)
    feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    clock.t += ec.FOLLOW_UP_S + 1
    events = feed_timed(ear, tone(1.5, 0.08) + noise(1.2), clock)
    assert [e["type"] for e in events] == ["speech", "heard"]


def test_a_long_question_started_in_time_is_answered():
    """Catches: "¿Qué elemento de la tabla periódica…?" after "¿Sí?" heard but not answered — it
    ENDED after the window, though it started well inside it (2026-10-28)."""
    clock = Clock()
    ear, _ = ear_saying("Bender", "qué elemento de la tabla periódica tiene 73 protones", clock=clock)
    feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    clock.t += ec.FOLLOW_UP_S - 1.0               # starts talking 1 s before the window closes
    events = feed_timed(ear, tone(5.0, 0.08) + noise(1.2), clock)   # 6 s: ends 5 s late
    assert events[-1] == {"type": "command", "text": "qué elemento de la tabla periódica tiene 73 protones"}


def test_talk_without_the_name_is_heard_but_not_obeyed():
    ear, _ = ear_saying("vamos a vender el auto")
    events = feed(ear, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
    assert "command" not in [e["type"] for e in events], "'vender' is not 'Bender'"


@pytest.mark.parametrize("invented", ["¡Suscríbete!", "Thank you, thank you, thank you.",
                                      "¡Gracias!", "Subtítulos realizados por la comunidad"])
def test_whispers_inventions_are_dropped(invented):
    """Catches: Whisper's inventions on noise reaching the UI — and, right after a bare
    "Bender", being obeyed as the command ("Thank you, thank you", 2026-10-19)."""
    ear, _ = ear_saying(invented)
    events = feed(ear, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
    assert [e["type"] for e in events] == ["speech"]


def test_a_real_sentence_with_thanks_in_it_is_kept():
    ear, _ = ear_saying("Bender, gracias por la ayuda")
    events = feed(ear, noise(1.0) + tone(1.0, 0.08) + noise(1.2))
    assert events[-1] == {"type": "command", "text": "gracias por la ayuda"}



# --------------------------------------------------------------------------- #
# Not hearing itself
# --------------------------------------------------------------------------- #
def test_while_the_robot_speaks_the_ear_is_deaf():
    """Catches: the robot hearing its own answer through its own mic and answering it."""
    clock = Clock()
    ear, heard = ear_saying("Bender, la respuesta del propio robot", clock=clock)
    ear.mute(True)
    feed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.2))
    ear.mute(False)
    assert heard == []
    feed(ear, tone(0.2, 0.08))                    # the tail of its own voice, inside the guard
    clock.t += ec.UNMUTE_GUARD_S + 0.1
    feed(ear, noise(1.5))
    assert heard == []


def test_saying_the_name_cuts_the_robot_off():
    """Catches: no way to stop a long answer — the ear was deaf while the robot talked."""
    clock = Clock()
    ear, heard = ear_saying("la mesa es marrón y la silla es Bender", clock=clock)
    ear.mute(True, barge=True)
    events = feed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.2))
    assert heard and events[-1] == {"type": "barge", "text": ""}
    ear.mute(False)
    clock.t += ec.UNMUTE_GUARD_S + 0.1
    ear.transcribe = lambda wav: "¿y la de la izquierda?"
    events = feed(ear, tone(1.0, 0.08) + noise(1.2))
    assert events[-1] == {"type": "command", "text": "¿y la de la izquierda?"}, "the ¿Sí? window opened"


def test_a_question_said_over_the_robot_is_answered_next():
    ear, _ = ear_saying("la mesa es marrón. Bender, ¿qué hora es?")
    ear.mute(True, barge=True)
    events = feed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.2))
    assert events[-1] == {"type": "barge", "text": "qué hora es"}


def test_while_talking_only_the_name_counts():
    """The robot's own words, heard through its mic, are never a question."""
    ear, heard = ear_saying("la mesa es marrón y la silla es gris")
    ear.mute(True, barge=True)
    events = feed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.2))
    assert heard and events == []


def test_a_repeated_name_is_not_the_question():
    """Catches: "Bender, Bender" asked "Bender" (2026-10-28)."""
    assert ec.wake_command("Bender, Bender") == (True, "")
    assert ec.wake_command("Bender bender, ¿qué ves?") == (True, "qué ves")
    assert ec.last_wake_command("Bender, Bender.") == (True, "")


def test_wake_command_parsing():
    assert ec.wake_command("Bender, ¿qué ves?") == (True, "qué ves")
    assert ec.wake_command("hola Bénder") == (True, "")
    assert ec.wake_command("vender la casa") == (False, "")


def test_recording_keeps_each_utterance_and_what_was_heard(tmp_path):
    # To measure the ear on real voices, the audio and the text must both be there, paired.
    import json
    heard = ec.recording(lambda wav: "Bender, qué ves", tmp_path / "ear", clock=lambda: 1761700000.25)
    wav = ec.pcm_to_wav(tone(1.0, 0.08))
    assert heard(wav) == "Bender, qué ves"
    line = json.loads((tmp_path / "ear" / "heard.jsonl").read_text(encoding="utf-8"))
    assert line["text"] == "Bender, qué ves" and line["seconds"] == 1.0
    assert (tmp_path / "ear" / line["wav"]).read_bytes() == wav


# --------------------------------------------------------------------------- #
# A pause to think inside a sentence
def cut_ear(*texts, clock):
    """An ear whose `cut` says a request ending in "que" or "llevo" stopped mid-sentence."""
    said = iter(texts)
    heard = []

    def transcribe(wav):
        heard.append(len(wav))
        return next(said, "")
    return ec.Ear(transcribe, clock=clock, cut=lambda t: t.endswith(("que", "llevo"))), heard


def test_a_sentence_cut_by_a_pause_is_heard_whole():
    """Catches: "Bender, acordate de que yo llevo" closed by a pause to think, and kept as the
    note "yo llevo" (2026-10-28)."""
    clock = Clock()
    ear, heard = cut_ear("Bender, acordate de que yo llevo",
                         "Bender, acordate de que yo llevo la notebook a la reunión", clock=clock)
    first = feed_timed(ear, noise(1.0) + tone(2.0, 0.08) + noise(1.0), clock)
    assert "command" not in [e["type"] for e in first], "held, not sent"
    rest = feed_timed(ear, tone(1.5, 0.08) + noise(1.2), clock)
    assert rest[-1] == {"type": "command", "text": "acordate de que yo llevo la notebook a la reunión"}
    assert heard[1] > heard[0] + ec.HOLD_GAP_S * ec.RATE * 2, "both halves transcribed together"


def test_a_cut_request_that_never_goes_on_is_sent_as_it_was():
    clock = Clock()
    ear, _ = cut_ear("Bender, acordate de que", clock=clock)
    first = feed_timed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.0), clock)
    assert "command" not in [e["type"] for e in first]
    later = feed_timed(ear, noise(ec.HOLD_S + 0.5), clock)
    assert later == [{"type": "command", "text": "acordate de que"}]


def test_talk_after_the_wait_does_not_lose_the_held_request():
    clock = Clock()
    ear, _ = cut_ear("Bender, acordate de que", "y después fuimos a comer", clock=clock)
    feed_timed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.0), clock)
    clock.t += ec.HOLD_S + 1.0                # silence the ear did not get packets for
    later = feed_timed(ear, tone(1.5, 0.08) + noise(1.2), clock)
    commands = [e for e in later if e["type"] == "command"]
    assert commands == [{"type": "command", "text": "acordate de que"}], "the late talk is not a request"


def test_a_finished_request_does_not_wait():
    clock = Clock()
    asked = []
    ear = ec.Ear(lambda wav: "Bender, ¿qué ves?", clock=clock, cut=lambda t: asked.append(t) or False)
    events = feed_timed(ear, noise(1.0) + tone(1.0, 0.08) + noise(1.2), clock)
    assert events[-1] == {"type": "command", "text": "qué ves"}
    assert asked == ["Bender, ¿qué ves?"], "judged on the words as heard, closing mark and all"


def test_a_second_pause_sends_what_was_said_so_far():
    clock = Clock()
    ear, _ = cut_ear("Bender, acordate de que", "Bender, acordate de que yo llevo", clock=clock)
    feed_timed(ear, noise(1.0) + tone(1.5, 0.08) + noise(1.0), clock)
    rest = feed_timed(ear, tone(1.0, 0.08) + noise(1.2), clock)
    assert rest[-1] == {"type": "command", "text": "acordate de que yo llevo"}
