"""The spoken assistant knows who it is: the robot's identity rides on every free question.

Every test here names the defect it catches. No model: the payload sent upstream is captured.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
