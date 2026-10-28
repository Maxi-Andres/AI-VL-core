"""The assistant searches the web only when the question asks for it, and says where the answer
came from — or that it could not search.

Every test names the defect it catches. No network: requests.get is replaced by a fake.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import requests

import search_common as sc
import vlm_common as vc


class _Answer:
    def __init__(self, results):
        self._results = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._results}


DOLLAR = [
    {"title": "Cotización del dólar blue", "url": "https://www.cronista.com/dolar.html",
     "content": "Dólar Blue $ 1.575", "publishedDate": "2026-10-28T10:00:00"},
    {"title": "Dólar hoy", "url": "https://dolarhoy.com/", "content": "Seguí la cotización"},
    {"title": "", "url": "https://empty.example/", "content": ""},
]


@pytest.fixture
def asked(monkeypatch):
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(params["q"])
        return _Answer(DOLLAR)

    monkeypatch.setattr(sc.requests, "get", fake_get)
    monkeypatch.setattr(sc, "_page_text", lambda url: "")   # no page reading in these
    return seen


@pytest.mark.parametrize("question", [
    "Bender, buscá en internet cuánto está el dólar blue",
    "averiguá si va a llover mañana en Buenos Aires",
    "Search the weather in Madrid",
    "look up the news about Formula 1",
])
def test_asking_to_search_searches_without_a_model(question):
    assert sc.wants_search(question)


@pytest.mark.parametrize("question", [
    "¿Qué valor muestra la pantalla?",
    "¿De qué color es el piso?",
    "Buscá el matafuego en la imagen",
    "¿Qué hora es?",
])
def test_questions_about_the_picture_are_not(question):
    """Catches: a search (1-2 s) on questions the camera answers, or "buscá X en la imagen"
    sent to the web."""
    assert not sc.wants_search(question)


def test_the_query_is_the_topic_without_the_asking():
    assert sc.search_query("Bender, buscá en internet el clima de hoy en Rosario") == \
        "el clima de hoy en rosario"


def test_results_reach_the_model_with_their_site(asked):
    """Catches: an answer from the web with no way to tell where it came from."""
    system = vc.free_prompt_system("g1", question="Bender, buscá cuánto está el dólar blue")
    assert asked == ["cuanto esta el dolar blue"]
    assert "Dólar Blue $ 1.575" in system and "(cronista.com, 2026-10-28)" in system
    assert "(dolarhoy.com)" in system and "empty.example" not in system
    assert "Bender" in system, "the identity still goes"


def test_no_search_and_no_request_for_a_question_about_the_image(asked):
    system = vc.free_prompt_system("g1", question="¿De qué color es el piso?")
    assert asked == [] and "web search" not in system


def test_a_failed_search_is_said_not_guessed(monkeypatch):
    """Catches: SearXNG down and the model inventing today's dollar rate."""
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(sc.requests, "get", down)
    fact = sc.search_fact("¿Cuánto está el dólar hoy?", decided=True)
    assert "could not search the internet" in fact and "do not guess" in fact


def test_a_question_after_a_figure_gets_the_snippets_with_numbers_first(monkeypatch):
    """Catches: "cuánto está el dólar blue" answered "no lo encontré" because the first four
    results were the sites' front pages, with the figure in the sixth (2026-10-28)."""
    pages = [{"title": f"Portada {i}", "url": f"https://site{i}.example/", "content": "Seguí la cotización"}
             for i in range(5)]
    pages.append({"title": "Blue", "url": "https://figures.example/", "content": "Dólar Blue $ 1.575"})
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: _Answer(pages))
    assert sc.search("dolar blue", figure=True)[0]["content"] == "Dólar Blue $ 1.575"
    assert sc.search("dolar blue")[0]["title"] == "Portada 0", "no figure asked: engines' order"


def test_the_stream_says_it_is_searching_first(monkeypatch):
    """Catches: a 1-4 s silent wait on a searching question, with nothing on screen to say why."""
    from fastapi.testclient import TestClient

    import service
    monkeypatch.setenv("IACORE_WARM_UP", "0")
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: _Answer(DOLLAR))
    monkeypatch.setattr(sc, "_page_text", lambda url: "")
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(vc, "model_supports_thinking", lambda *a, **k: False)
    monkeypatch.setattr(vc, "_openai_chunks", lambda *a: iter([{"choices": [{"delta": {"content": "$1.575"}}]}]))
    monkeypatch.setattr(sc, "model_says_web", lambda q, url: "dólar" in q)   # the model's call
    with TestClient(service.app) as c:
        searched = c.post("/vlm/stream", json={"image": "aGk=", "prompt": "¿Cuánto está el dólar?"}).text
        plain = c.post("/vlm/stream", json={"image": "aGk=", "prompt": "¿Qué ves?"}).text
    assert searched == sc.SEARCHING_MARK + "$1.575"
    assert plain == "$1.575"


def test_only_passages_with_the_topic_and_a_date_are_kept():
    """Catches: a whole calendar page (menus, ads) filling the window instead of the match line."""
    page = ("Menú Fútbol Rugby Tenis F1 Básquetbol " * 20 +
            "Dom., 1 de Nov. Boca Juniors v River Plate 3:00 PM Liga Profesional " +
            "Suscribite al newsletter " * 20)
    found = sc.passages(page, "cuando juega river")
    assert "1 de Nov. Boca Juniors v River Plate 3:00 PM" in found and len(found) <= sc.PAGE_CHARS


def test_pages_on_the_office_network_are_never_fetched(monkeypatch):
    """Catches: a result pointing at 192.168.x or localhost making this machine fetch it."""
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: pytest.fail("fetched"))
    for url in ("http://192.168.20.99:8443/", "http://127.0.0.1:8001/status", "file:///etc/passwd"):
        assert sc._page_text(url) == ""


class _Model:
    """Stand-in for the llama-server's answer to the classifier."""
    def __init__(self, word):
        self.word, self.asked = word, []

    def __call__(self, url, timeout=None, json=None):
        self.asked.append(json["messages"][-1]["content"])
        word = self.word
        return type("R", (), {"raise_for_status": lambda s: None,
                              "json": lambda s: {"choices": [{"message": {"content": word}}]}})()


def test_the_model_decides_what_no_keyword_covers(monkeypatch):
    """Catches: "¿Quién es el presidente de Francia?" unsearched because no word was listed."""
    model = _Model("WEB")
    monkeypatch.setattr(sc.requests, "post", model)
    assert sc.wants_search("¿Quién es el presidente de Francia?", classifier_url="http://vlm")
    assert model.asked == ["¿Quién es el presidente de Francia?"]


def test_asking_and_pointing_are_decided_without_the_model(monkeypatch):
    """Catches: the model overruling "buscá…", or asked about "¿qué dice el cartel?" (the one
    it got wrong in the measurement)."""
    model = _Model("NO")
    monkeypatch.setattr(sc.requests, "post", model)
    assert sc.wants_search("buscá quién ganó en Brasil", classifier_url="http://vlm")
    assert sc.wants_search("search who won in Brazil", classifier_url="http://vlm")
    assert not sc.wants_search("¿Qué dice el cartel?", classifier_url="http://vlm")
    assert not sc.wants_search("what do you see?", classifier_url="http://vlm")
    assert model.asked == []


def test_without_the_model_only_asking_searches(monkeypatch):
    """No topic word list (they needed one word per topic, in two languages): with the model
    down, a question is searched only when the person asks for it."""
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(sc.requests, "post", down)
    assert not sc.wants_search("¿Quién ganó el partido?", classifier_url="http://vlm")
    assert not sc.wants_search("Who won the match?", classifier_url="http://vlm")
    assert sc.wants_search("buscá quién ganó el partido", classifier_url="http://vlm")
    assert sc.wants_search("search who won the match", classifier_url="http://vlm")


def test_there_is_no_topic_word_list():
    assert not hasattr(sc, "TOPICS") and not hasattr(sc, "WHEN")


def test_a_web_question_goes_without_the_frame(monkeypatch):
    """Catches: "no puedo buscar quién ganó porque la imagen es de una habitación" — the frame
    sent with a question that is not about it (2026-10-28)."""
    sent = []
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(vc, "model_supports_thinking", lambda *a, **k: False)
    monkeypatch.setattr(vc, "_openai_chunks", lambda payload, url, t: sent.append(payload) or iter(()))
    monkeypatch.setattr(sc, "search_fact", lambda q, url=None, decided=None, **k: "results" if decided else "")
    list(vc.query_vlm_stream("aGk=", "m", "quién ganó en Brasil", search=True))
    list(vc.query_vlm_stream("aGk=", "m", "buscá en internet qué es esto", search=True))
    list(vc.query_vlm_stream("aGk=", "m", "¿qué ves?", search=False))
    assert "images" not in sent[0]["messages"][-1]
    assert "images" in sent[1]["messages"][-1], "it points at the picture: the frame goes"
    assert "images" in sent[2]["messages"][-1]


@pytest.mark.parametrize("question, query", [
    ("Bender, can you search the web for who won the F1 race last weekend", "who won the f1 race last weekend"),
    ("Please look up the weather in London tomorrow", "the weather in london tomorrow"),
    ("Bender, buscá en internet el clima de hoy en Rosario", "el clima de hoy en rosario"),
])
def test_the_asking_is_dropped_in_both_languages(question, query):
    """Catches: "can you search the web for" searched as words, in an English question."""
    assert sc.search_query(question) == query


def test_dates_are_found_in_english_pages_too():
    page = "Menu Football Tennis " * 20 + "Sun, Nov 1 Real Madrid v Barcelona 3:00 PM LaLiga " + "Sign up " * 20
    assert "Nov 1 Real Madrid v Barcelona 3:00 PM" in sc.passages(page, "when does real madrid play")


def test_the_engines_pick_the_language_from_the_query(asked, monkeypatch):
    """Catches: English questions sent to Spanish pages ("language": "es" on every search)."""
    seen = {}
    monkeypatch.setattr(sc.requests, "get", lambda url, params=None, timeout=None: seen.update(params) or _Answer(DOLLAR))
    sc.search("who won the race")
    assert seen["language"] == "auto"


@pytest.mark.parametrize("question", [
    "que partidos de fútbol hay hoy, buscá eso", "buscá eso", "buscalo en internet",
    "what matches are on today, search that", "look it up", "search it on the web",
])
def test_the_that_after_search_is_the_question_not_the_picture(question):
    """Catches: "¿qué partidos hay hoy?, buscá eso" not searched — "eso" was taken for something
    in view (2026-10-28)."""
    assert sc.wants_search(question)
    assert not sc.about_the_image(question)


@pytest.mark.parametrize("question", ["buscá qué marca es esto", "search what brand this is"])
def test_a_that_inside_the_question_is_still_the_picture(question):
    """It points at something in view: searched with the frame, its query written from it."""
    assert sc.wants_search(question) == "both"


@pytest.mark.parametrize("follow_up, earlier, query", [
    ("pero buscalo en internet vos", ["¿de qué color es?", "¿qué partidos de fútbol hay hoy?"],
     "que partidos de futbol hay hoy"),
    ("look it up then", ["what matches are on today?"], "what matches are on today"),
    ("buscá eso", [], "busca eso"),
    ("y buscá el clima en Rosario", ["¿qué partidos hay hoy?"], "y el clima en rosario"),
])
def test_a_follow_up_without_a_topic_searches_the_previous_question(follow_up, earlier, query):
    """Catches: "pero buscalo en internet vos" searched as "pero vos", and the model, with junk
    results, said it could not search (2026-10-28)."""
    assert sc.search_query(follow_up, earlier) == query


def test_the_stream_searches_the_previous_question(monkeypatch):
    """The conversation's questions reach the search, not only the model."""
    seen = {}
    monkeypatch.setattr(vc, "vlm_backend", lambda: "llamacpp")
    monkeypatch.setattr(vc, "model_supports_thinking", lambda *a, **k: False)
    monkeypatch.setattr(vc, "_openai_chunks", lambda payload, url, t: iter(()))
    monkeypatch.setattr(sc, "search", lambda q, url=None, figure=False: seen.setdefault("q", q) and [])
    list(vc.query_vlm_stream("aGk=", "m", "pero buscalo en internet vos", search=True,
                             history=[{"question": "¿Qué partidos hay hoy?", "answer": "No sé."}]))
    assert seen["q"] == "que partidos hay hoy"


def test_after_a_search_the_model_is_told_it_did_search(asked):
    """Catches: "no puedo realizar búsquedas en Internet" over the results, copied from its own
    earlier answer in the history."""
    assert "never say you cannot search" in sc.search_fact("¿Cuánto está el dólar hoy?", decided=True)


def test_without_a_search_the_model_offers_one():
    assert "ask you to search the internet" in vc.free_prompt_system(question="¿Quién ganó?", search=False)
    assert "search" not in vc.free_prompt_system(question="¿Qué dice el cartel?", search=False), \
        "reading prompts stay as measured"


class _Chat:
    """llama-server's chat answer, and what it was sent."""
    def __init__(self, text, sent):
        self.text, self.sent = text, sent

    def __call__(self, url, timeout=None, json=None):
        self.sent.append(json)
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


@pytest.mark.parametrize("question, kind", [
    ("¿Cuánto cuesta este multímetro?", "both"), ("How much does this multimeter cost?", "both"),
    ("¿Qué horario tiene ese castillo?", "both"), ("Where can I buy one of these?", "both"),
    ("¿Cuánto cuesta un Fluke 87?", "web"), ("¿A cuánto esta el dolar?", "web"),
    ("Did SpaceX launch anything this week?", "web"),
    ("¿Qué dice el cartel?", False), ("What do you see?", False),
])
def test_what_the_question_points_at_decides_the_kind(monkeypatch, question, kind):
    """Catches: "¿cuánto cuesta este multímetro?" searched as "multímetro" without the frame,
    or not searched at all (2026-10-28). The model only says whether the web is needed."""
    monkeypatch.setattr(sc, "model_says_web", lambda q, url: True)
    assert sc.wants_search(question, classifier_url="http://vlm") == kind


def test_a_question_about_something_in_view_searches_what_the_frame_shows(asked, monkeypatch):
    sent = []
    monkeypatch.setattr(sc.requests, "post", _Chat("precio multímetro Fluke 87", sent))
    fact = sc.search_fact("¿Cuánto cuesta este multímetro?", decided="both", image="aGk=",
                          vlm_url="http://vlm")
    assert asked == ["precio multimetro fluke 87"]
    assert sent[0]["messages"][0]["content"][0]["image_url"]["url"].endswith("aGk="), "it looked"
    assert "about what is in the image" in fact and "do not mention the image" not in fact


def test_the_model_s_filler_is_dropped_from_the_query(asked, monkeypatch):
    monkeypatch.setattr(sc.requests, "post", _Chat("Horario de visita del castillo mostrado en la imagen", []))
    sc.search_fact("¿Qué horario tiene este castillo?", decided="both", image="aGk=", vlm_url="http://vlm")
    assert asked == ["horario de visita del castillo"]


@pytest.mark.parametrize("question", ["Did SpaceX launch anything this week?",
                                      "¿Quién ganó este fin de semana?", "¿Qué pasó este año?"])
def test_this_week_is_a_time_not_something_in_view(question):
    assert not sc.points_in_view(question)


def test_without_the_model_the_question_itself_is_searched(asked, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(sc.requests, "post", down)
    sc.search_fact("¿Cuánto cuesta este multímetro?", decided="both", image="aGk=", vlm_url="http://vlm")
    assert asked == ["cuanto cuesta este multimetro"]
