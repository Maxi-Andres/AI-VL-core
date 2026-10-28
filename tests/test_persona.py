"""The spoken assistant knows who it is: the robot's identity rides on every free question.

Every test here names the defect it catches. No model: the payload sent upstream is captured.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import vlm_common as vc


def test_the_g1_is_bender_made_by_maxi_and_sees_through_its_own_camera():
    """Catches: "parece que te están viendo" / "Soy Qwen" (2026-10-19) — the model not knowing
    it is the robot and that the image is its own view."""
    persona = vc.persona_for("g1")
    assert "Bender" in persona and "your own camera" in persona
    creator = vc.persona_for("g1", question="¿Quién te creó?")
    assert "Maxi, an engineer" in creator, "Maxi alone came back as 'la empresa de Unitree'"


def test_a_question_about_the_image_carries_only_the_short_identity():
    """Catches: every fact riding on every question — it recited them all and the reading of
    the gauges wobbled (2026-10-19)."""
    persona = vc.persona_for("g1", question="¿Qué valor muestra la pantalla?")
    for fact in ("Silk", "Mateo", "Huesos", "130"):
        assert fact not in persona


def test_only_the_fact_asked_about_is_added():
    """Catches: "¿Quién es tu mejor amigo?" answered with the creator, the job and the brother."""
    friend = vc.persona_for("g1", question="¿Quién es tu mejor amigo?")
    assert "Mateo" in friend and "Huesos" not in friend and "130" not in friend
    assert "130 cm" in vc.persona_for("g1", question="¿Cuánto medís?")
    both = vc.persona_for("g1", question="¿Tenés hermanos y dónde trabajás?")
    assert "Huesos" in both and "Silk Technologies" in both


def test_keywords_match_whole_words_not_pieces():
    assert "130" not in vc.persona_for("g1", question="¿Dio un salto la presión?")
    assert "130" in vc.persona_for("g1", question="¿Qué tan alto sos?")


def test_it_is_not_the_cartoon_bender():
    """Catches: the name alone bringing in Futurama's Bender ("¡Ay, qué mierda!")."""
    assert "not the cartoon character" in vc.persona_for("g1")


def test_the_time_is_this_machines_clock_and_only_when_asked():
    import re
    assert re.search(r"right now it is \d\d:\d\d on \w+ \d+ de \w+ de 20\d\d",
                     vc.persona_for("g1", question="¿Qué hora es?"))
    assert "right now" not in vc.persona_for("g1", question="¿Qué ves?")


def test_huesos_knows_his_brother():
    assert "Huesos" in vc.persona_for("go2")
    assert "Bender" in vc.persona_for("go2", question="¿Tenés hermano?")


def test_the_identity_holds_when_the_page_does_not_say_which_robot():
    assert vc.persona_for(None) == vc.persona_for("g1")


def test_the_identity_is_appended_to_the_plain_instruction_never_merged():
    """Catches: rewording the instruction itself — that has broken reading twice (a nameplate
    read "Fluke", 17.84 kW read 1784). The plain sentence must stay first and unchanged."""
    system = vc.free_prompt_system("g1")
    assert system.startswith(vc.FREE_PROMPT_SYSTEM + " ")
    assert vc.free_prompt_system("a-robot-with-no-persona") == vc.FREE_PROMPT_SYSTEM


def test_the_stream_carries_the_identity_upstream(monkeypatch):
    sent = {}

    def fake_chunks(payload, url, timeout):
        sent.update(payload)
        return iter([{"choices": [{"delta": {"content": "Sí."}}]}])

    monkeypatch.setenv("VLM_BACKEND", "llamacpp")
    monkeypatch.setattr(vc, "_openai_chunks", fake_chunks)
    assert list(vc.query_vlm_stream("QUJD", "m", "¿Me ves?", url="http://x", robot="g1")) == ["Sí."]
    assert "Bender" in sent["messages"][0]["content"]


def test_the_time_is_said_in_the_language_asked():
    """Catches: "What time is it?" answered "14:23 on miércoles 28 de octubre" (2026-10-28)."""
    import datetime
    now = datetime.datetime(2026, 10, 28, 14, 23)
    assert vc._now_text(now, english=True) == "14:23 on Wednesday 28 October 2026"
    assert vc._now_text(now) == "14:23 on miércoles 28 de octubre de 2026"
    assert vc.asked_in_english("What time is it?")
    assert not vc.asked_in_english("¿Qué hora es?") and not vc.asked_in_english("que hora es")


class _Said:
    def __init__(self, label):
        self.label, self.sent = label, []

    def __call__(self, url, timeout=None, json=None):
        self.sent.append(json)
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.label}}]}


@pytest.mark.real_looks
def test_talk_is_told_from_a_question_about_the_frame(monkeypatch):
    said = _Said("CHAT")
    monkeypatch.setattr(vc.requests, "post", said)
    assert not vc.talks_about_the_frame("¡Qué bueno es!", "http://vlm")
    assert said.sent[0]["messages"][-1]["content"] == "¡Qué bueno es!"
    said.label = "LOOK"
    assert vc.talks_about_the_frame("¿Cuántos kVA marca?", "http://vlm")


@pytest.mark.real_looks
def test_pointing_at_the_view_is_the_frame_without_asking(monkeypatch):
    said = _Said("CHAT")
    monkeypatch.setattr(vc.requests, "post", said)
    assert vc.talks_about_the_frame("¿Qué es esto?", "http://vlm")
    assert said.sent == []


@pytest.mark.real_looks
def test_a_server_that_cannot_tell_keeps_the_reading_prompt(monkeypatch):
    def down(*a, **k):
        raise vc.requests.ConnectionError("refused")

    monkeypatch.setattr(vc.requests, "post", down)
    assert vc.talks_about_the_frame("Hola", "http://vlm")


STATE = {"name": "Huesos", "online": True, "latency_ms": 42, "link": "lte",
         "battery": {"percent": 81.6, "charging": False, "temp_c": 33.0, "age_s": 5}}


@pytest.mark.parametrize("question, says", [
    ("¿Cuánta batería tenés?", "your battery is at 82 %, not charging"),
    ("How much battery do you have?", "your battery is at 82 %"),
    ("¿Estás cargando?", "not charging"),
    ("¿Estás caliente?", "your battery pack (not a motor) is at 33 °C"),
    ("¿Tenés buena señal?", "you are connected over lte, 42 ms round trip"),
    ("¿Cómo estás?", "your battery is at 82 %"),
])
def test_the_robot_tells_its_own_state_when_asked(question, says):
    """Catches: "¿cuánta batería tenés?" answered without knowing (2026-10-28)."""
    assert says in vc.persona_for("go2", question=question, state=STATE)


@pytest.mark.parametrize("question", ["¿Qué temperatura marca el termómetro?", "Hola", "¿Qué ves?",
                                      "What is the power factor?"])
def test_its_state_stays_out_of_other_questions(question):
    """A reading about a thermometer is not about the robot's own heat."""
    assert "battery" not in vc.persona_for("go2", question=question, state=STATE)


def test_an_old_or_missing_reading_is_said_as_such():
    old = {"battery": {"percent": 40, "age_s": 1800}}
    assert "(read 30 min ago)" in vc.persona_for("g1", question="¿Cuánta batería tenés?", state=old)
    assert "not known" in vc.persona_for("g1", question="¿Cuánta batería tenés?", state={"online": True})


def test_a_question_about_itself_is_talk():
    assert vc.asks_about_itself("go2", "¿Cuánta batería tenés?", STATE)
    assert vc.asks_about_itself("g1", "¿Quién te creó?")
    assert not vc.asks_about_itself("g1", "¿Cuántos kVA marca?", STATE)


def test_what_it_was_asked_to_remember_goes_with_talk_only():
    """"Acordate que la llave está en el cajón", then "¿dónde está la llave?"."""
    notes = ["la llave está en el cajón de arriba"]
    talk = vc.free_prompt_system(question="¿Dónde está la llave?", search=False, chat=True, robot_notes=notes)
    assert "la llave está en el cajón de arriba" in talk
    reading = vc.free_prompt_system(question="¿Cuántos kVA marca?", search=False, robot_notes=notes)
    assert "cajón" not in reading and reading == vc.free_prompt_system(question="¿Cuántos kVA marca?", search=False)
    # About the frame, but a note answers it: that note goes (it was answered from the picture).
    where = vc.free_prompt_system(question="¿Dónde está la llave?", search=False, robot_notes=notes)
    assert "la llave está en el cajón de arriba" in where
