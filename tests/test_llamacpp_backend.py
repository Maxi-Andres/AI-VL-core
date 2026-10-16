"""The llama-server backend: Ollama-style payloads in, the same results out.

Every test here names the defect it catches. No server: requests.post is replaced by a fake
that records the request and replays a canned SSE stream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import command_common as cc
import vlm_common as vc


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self._lines


def _sse(*chunks):
    return [b"data: " + json.dumps(c).encode() for c in chunks] + [b"", b"data: [DONE]"]


def _delta(text, finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def _fake_post(monkeypatch, lines):
    sent = {}

    def post(url, json=None, stream=False, timeout=None):  # requests.post's own names
        sent.update(url=url, body=json)
        return _FakeResponse(lines)

    monkeypatch.setattr(vc.requests, "post", post)
    return sent


def test_images_become_data_uris_before_the_text():
    """Catches: Ollama's `images` list forwarded as is — llama-server ignores the field and
    the model answers about no picture at all."""
    out = vc._openai_payload({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is this", "images": ["QUJD"]}]})
    user = out["messages"][1]["content"]
    assert user[0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert user[1] == {"type": "text", "text": "what is this"}
    assert out["messages"][0]["content"] == "sys"


def test_ollama_options_and_defaults_are_carried_over():
    """Catches: num_predict lost (answers capped at the server's default) or the repeat
    penalty missing — measured 2026-10-16, a long object list then looped to max_tokens."""
    out = vc._openai_payload({"messages": [], "options": {"temperature": 0.0,
                                                          "num_predict": 77}})
    assert out["temperature"] == 0.0
    assert out["max_tokens"] == 77
    assert out["repeat_penalty"] == 1.1 and out["top_k"] == 40 and out["top_p"] == 0.9
    assert out["stream"] is True and out["cache_prompt"] is True


def test_format_json_becomes_a_compact_grammar():
    """Catches: format "json" dropped (free text where the VLA contract wants JSON), or a
    grammar that allows pretty-printing, which spent the token budget on whitespace."""
    out = vc._openai_payload({"messages": [], "format": "json"})
    assert out["grammar"] == vc._COMPACT_JSON_GBNF
    assert 'ws     ::= " "?' in out["grammar"]
    assert "grammar" not in vc._openai_payload({"messages": []})


def test_the_stream_is_parsed_into_stream_chats_contract(monkeypatch):
    """Catches: SSE lines parsed wrong — content, finish reason or token counts lost."""
    monkeypatch.setenv("VLM_BACKEND", "llamacpp")
    sent = _fake_post(monkeypatch, _sse(
        _delta('{"a":'), _delta("1}", finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}))
    content, reasoning, done, in_tok, out_tok = vc.stream_chat(
        {"messages": [], "format": "json"}, url="http://127.0.0.1:8081")
    assert sent["url"] == "http://127.0.0.1:8081/v1/chat/completions"
    assert (content, reasoning, done, in_tok, out_tok) == ('{"a":1}', "", "stop", 12, 3)


def test_the_free_prompt_stream_yields_only_answer_text(monkeypatch):
    """Catches: the voice path reading out empty or reasoning chunks."""
    monkeypatch.setenv("VLM_BACKEND", "llamacpp")
    _fake_post(monkeypatch, _sse(
        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
        _delta("Una "), _delta(""), _delta("oficina.", finish="stop")))
    pieces = list(vc.query_vlm_stream("QUJD", "m", "¿Qué ves?", url="http://127.0.0.1:8081"))
    assert pieces == ["Una ", "oficina."]


def test_the_command_interpreter_works_on_llamacpp(monkeypatch):
    """Catches: interpret() still asking Ollama for model capabilities (an HTTP call to a
    server that may not run) or failing to parse the JSON llama-server streams."""
    monkeypatch.setenv("VLM_BACKEND", "llamacpp")
    monkeypatch.setattr(vc.requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no capability lookup on llama-server")))
    _fake_post(monkeypatch, _sse(_delta('{"skill":"stop","params":{},"say":"Paro."}',
                                        finish="stop")))
    res = cc.interpret("pará", "m", robot="g1", url="http://127.0.0.1:8081")
    assert res["ok"] and res["skill"] == "stop"


def test_the_environment_overrides_config(monkeypatch):
    """Catches: run.sh's fallback to Ollama (it exports VLM_BACKEND=ollama when llama-server
    did not come up) ignored because config.json says llamacpp."""
    monkeypatch.setattr(vc, "load_config", lambda: {"vlm_backend": "llamacpp"})
    monkeypatch.setenv("VLM_BACKEND", "ollama")
    assert vc.vlm_backend() == "ollama"
    monkeypatch.delenv("VLM_BACKEND")
    assert vc.vlm_backend() == "llamacpp"


def test_objects_that_are_not_objects_do_not_crash_bbox_normalization():
    """Catches: "objects": ["person", ...] (valid JSON, wrong shape — qwen3-vl 4B Q8 answered
    that on 2026-10-16) raising AttributeError, a 500 on /vlm whichever server answered."""
    parsed = {"objects": ["person", {"bbox": [10, 20, 30, 40]}, None]}
    vc.normalize_bboxes(parsed, (100, 200))
    assert parsed["objects"][0] == "person"
    assert parsed["objects"][1]["bbox"] == [0.1, 0.1, 0.3, 0.2]
    assert vc.normalize_bboxes({"objects": "none"}, (100, 200)) == {"objects": "none"}
