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
    assert "Bender" in persona and "Maxi" in persona and "your own camera" in persona


def test_bender_knows_his_facts_and_his_brother_knows_him():
    g1, go2 = vc.persona_for("g1"), vc.persona_for("go2")
    for fact in ("Silk Technologies", "Mateo", "Huesos", "130 cm"):
        assert fact in g1
    assert "Huesos" in go2 and "Bender" in go2


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
