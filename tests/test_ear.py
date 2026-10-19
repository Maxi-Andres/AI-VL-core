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


def test_the_follow_up_window_closes():
    """Catches: "Bender" said once and every later conversation in the room taken as orders."""
    clock = Clock()
    ear, _ = ear_saying("Bender", "y después fuimos a comer", clock=clock)
    feed(ear, noise(1.0) + tone(0.6, 0.08) + noise(1.2))
    clock.t += ec.FOLLOW_UP_S + 1
    events = feed(ear, tone(1.5, 0.08) + noise(1.2))
    assert [e["type"] for e in events] == ["speech", "heard"]


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


def test_wake_command_parsing():
    assert ec.wake_command("Bender, ¿qué ves?") == (True, "qué ves")
    assert ec.wake_command("hola Bénder") == (True, "")
    assert ec.wake_command("vender la casa") == (False, "")
