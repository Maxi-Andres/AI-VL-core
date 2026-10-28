"""Web search for the vision assistant: "buscá…", news, weather, prices.

The assistant only knows what is in the image and what it was trained on. When a question asks
for something outside both — today's dollar rate, the weather, a news item — the answer was
made up. Here such a question is searched first, on the local SearXNG
(AI-VL-ecosystem/deploy/searxng: no API key, no quota, it asks the public engines itself), and
the first results go to the model as a fact with their source and date, the same way the time
does (vlm_common.persona_for): only when the question asks for it.

Whether to search (wants_search), in this order:
  1. the person asks for it — "buscá", "en internet", "search", "look up": yes;
  2. the question is about what is in view — "el cartel", "la pantalla", "esto": no;
  3. otherwise the model decides, from the question alone, without the image (model_says_web).
     Measured 2026-10-28 on 42 questions (image, the robot itself, general knowledge, and
     news / sports / prices / weather / schedules / office holders): 41 right, ~27 ms each on
     the 4B. A keyword list had gaps with every new topic ("¿cuándo juega River?");
  4. no model to ask (Ollama backend, server down): the topic keywords, in both languages.
Plan: .claude/roadmap/PLATAFORMA.md, "Bender útil" §1.
"""

import datetime
import html
import ipaddress
import os
import re
import socket
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import requests

# First chunk of a streamed answer that searches: "searching the web now". Control characters
# (record separators) no answer contains; the page reads and drops it (api/backend.ts).
SEARCHING_MARK = "\x1esearching\x1e"

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8095").rstrip("/")
MAX_RESULTS = 4
# Read this many to pick MAX_RESULTS from: for a figure, the first few are often the sites'
# front pages ("Seguí la cotización minuto a minuto") and the number sits further down.
CANDIDATES = 10
SNIPPET_CHARS = 280
# The answer waits on this. SearXNG drops its own slow engines at 3-4 s (settings.yml).
TIMEOUT_S = 6.0

# All lists are matched without accents, at the START of a word ("busc" catches busca, buscá,
# buscame), in Spanish and English.
# 1. The person asks for the web. Naming it wins even over the picture ("buscá en internet qué
#    es esto"); a bare "buscá" does not ("buscá el matafuego en la imagen").
NAMES_THE_WEB = ["internet", "en la web", "en google", "googlea", "en linea", "online",
                 "on the web", "web search", "google it"]
ASKS_TO_SEARCH = ["busc", "averigu", "search", "look up", "find out"]
# 2. About what is in view — whole words only ("esa" is not "esas", "eso" is not "esos").
ABOUT_THE_IMAGE = ["imagen", "foto", "camara", "pantalla", "cartel", "letrero", "frente", "ves",
                   "veo", "esto", "eso", "esa", "ese", "aca", "ahi", "image", "picture", "photo",
                   "camera", "screen", "sign", "this", "see"]
# 4. Without a model: topics only the web has.
TOPICS = [
    "noticia", "novedad", "ultimo momento", "news", "headline",
    "clima", "pronostico", "llover", "llueve", "lluvia", "weather", "forecast", "rain",
    "precio", "cuanto cuesta", "cuanto sale", "cuanto vale", "cotizacion", "dolar", "euro",
    "price", "how much", "exchange rate",
    "juega", "partido", "fixture", "quien gano", "resultado", "campeonato", "torneo", "mundial",
    "match", "who won", "score", "league",
    "cuando es", "cuando sale", "cuando viene", "cuando abre", "a que hora", "estreno",
    "elecciones", "presidente", "when is", "what time", "opening hours", "election", "president",
]
# What the model is told, for step 3 (wording measured, see the module docstring).
CLASSIFIER = (
    "The person is looking at a live camera image and asking a robot assistant. Decide if their "
    "question needs a web search. Answer NO when it is about anything that can be seen in the "
    "image: what is there, colors, what a gauge, thermometer, screen or sign reads, the brand "
    "or model of a thing in view (esto, eso, esa, ese, ahí); also NO for questions about the "
    "assistant itself, the time, or stable general knowledge, math or language. Answer WEB "
    "only when it needs current information from the world outside the image: news, sports "
    "results or fixtures, weather, prices, exchange rates, schedules, opening hours, events, "
    "who holds a post now. One word: WEB or NO.")
CLASSIFIER_TIMEOUT_S = 2.0

# Questions after a figure: price, rate, temperature. Their results are ranked snippets-with-
# numbers first (2026-10-28: "cuánto está el dólar blue" got four front pages and no figure).
FIGURE = ["precio", "cuanto", "cotizacion", "dolar", "euro", "temperatura", "grados", "price",
          "how much", "rate", "temperature", "cost"]
_NUMBER = re.compile(r"\d[\d.,]*")

# Questions after a date, an hour or a result: like FIGURE, the snippets are front pages
# ("consultá las fechas…") and the answer is inside the page (ESPN's calendar: "Dom., 1 de Nov.
# Boca Juniors v River Plate 3:00 PM"). For these, and for FIGURE, the first pages are READ.
WHEN = ["cuando", "a que hora", "fecha", "horario", "dia", "quien gano", "resultado", "salio",
        "when", "what time", "date", "schedule", "day", "who won", "result", "score"]
PAGES = 3                 # read at most this many, at the same time
PAGE_TIMEOUT_S = 3.0      # each; a slow site is skipped, the snippets still go
PAGE_BYTES = 1_500_000
PAGE_CHARS = 600          # of each page, in its best passages
# Dates in either language: "1 de nov", "Nov 1", "dom.", "Sun", "3:00", "hoy", "today".
_DATEISH = re.compile(r"\b(\d{1,2}[:/.]\d{2}|\d{1,2} de [a-z]+|[a-z]{3,9} \d{1,2}(st|nd|rd|th)?|"
                      r"lun|mar|mie|jue|vie|sab|dom|mon|tue|wed|thu|fri|sat|sun|"
                      r"ene|feb|abr|may|jun|jul|ago|sep|oct|nov|dic|jan|apr|aug|dec|"
                      r"hoy|manana|today|tomorrow|tonight)\b")
_STOP = {"cuando", "donde", "como", "cual", "cuanto", "cuanta", "esta", "estan", "para", "que",
         "quien", "hoy", "manana", "el", "la", "los", "las", "de", "del", "en", "un", "una",
         "when", "where", "what", "which", "who", "how", "much", "many", "does", "did", "will",
         "the", "and", "for", "today", "tomorrow", "next", "last", "this"}

# Words that tell the assistant TO search; the rest of the question is what to search for.
_ASKING = re.compile(
    r"\b(bender|huesos|por favor|porfa|che|podes|puedes|podrias|me|"
    r"busca(me|r|lo|la)?|averigua(me|r)?|googlea(me)?|fijate|decime|dime|"
    r"en internet|por internet|en la web|en google|en linea|online|"
    r"please|can you|could you|would you|for me|tell me|search( for| the web for)?|"
    r"look( it)? up|find out|google( it)?|on the (internet|web)|the internet|the web)\b[,.]?",
    re.IGNORECASE)


def _plain(text):
    return "".join(c for c in unicodedata.normalize("NFD", (text or "").lower())
                   if unicodedata.category(c) != "Mn")


def _has(text, words):
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


def _has_word(text, words):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


def about_the_image(question):
    """The question points at what is in view ("el cartel", "esto", "what do you see")."""
    return _has_word(_plain(question), ABOUT_THE_IMAGE)


def model_says_web(question, url, timeout=CLASSIFIER_TIMEOUT_S):
    """Ask the VLM server (llama-server's OpenAI API, text only) whether the question needs the
    web. True / False, or None when it could not be asked."""
    try:
        r = requests.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=timeout, json={
            "model": "default", "temperature": 0, "max_tokens": 3,
            "messages": [{"role": "system", "content": CLASSIFIER},
                         {"role": "user", "content": question}]})
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
    return answer.strip().upper().startswith("WEB")


def wants_search(question, classifier_url=None):
    """True when the question needs the web (the order is in the module docstring).
    `classifier_url`: the llama-server to ask in step 3; without it, the topic keywords."""
    q = _plain(question)
    if not q:
        return False
    if _has(q, NAMES_THE_WEB):
        return True
    image = _has_word(q, ABOUT_THE_IMAGE)
    if _has(q, ASKS_TO_SEARCH):
        return not image
    if image:
        return False
    if classifier_url:
        decided = model_says_web(question, classifier_url)
        if decided is not None:
            return decided
    return _has(q, TOPICS)


def search_query(question):
    """The question without the asking ("Bender, buscá en internet el clima de hoy" -> "el clima
    de hoy"): the engines rank the topic, and the polite words only dilute it."""
    q = _ASKING.sub(" ", _plain(question))
    q = re.sub(r"[¿?¡!]", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ,.")
    return q or _plain(question)


def _public(url):
    """Only http(s) to a public address: the result list comes from the internet, and nothing
    in it may make this machine fetch the office network or itself."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


def _page_text(url):
    """The visible text of a page, or "" (not public, not HTML, too slow, too big)."""
    if not _public(url):
        return ""
    try:
        with requests.get(url, timeout=PAGE_TIMEOUT_S, stream=True, allow_redirects=False,
                          headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "es,en;q=0.9"}) as r:
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                return ""
            body = r.raw.read(PAGE_BYTES, decode_content=True)
    except (requests.RequestException, OSError):
        return ""
    text = body.decode(r.encoding or "utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def passages(text, query, limit=PAGE_CHARS):
    """The parts of a page that answer: windows holding the query's words AND a number or a
    date, best first, kept in page order, up to `limit` characters."""
    words = [w for w in re.findall(r"[a-z0-9]{3,}", _plain(query)) if w not in _STOP]
    if not words or not text:
        return ""
    plain = _plain(text)
    step = 200
    scored = []
    for start in range(0, len(plain), step):
        chunk = plain[start:start + step + 60]
        hits = sum(chunk.count(w) for w in words)
        if not hits:
            continue
        score = 2 * hits + 2 * bool(_DATEISH.search(chunk)) + bool(_NUMBER.search(chunk))
        if score >= 4:
            scored.append((score, start))
    chosen, used = [], 0
    for _, start in sorted(scored, reverse=True):
        if used >= limit:
            break
        chosen.append(start)
        used += step + 60
    return " … ".join(text[s:s + step + 60].strip() for s in sorted(chosen))[:limit]


def read_pages(results, query):
    """Each result gets `found`: its best passages for the query (the first PAGES, at once)."""
    head = results[:PAGES]
    with ThreadPoolExecutor(max_workers=len(head) or 1) as pool:
        texts = list(pool.map(lambda r: _page_text(r["url"]), head))
    for r, text in zip(head, texts, strict=True):
        r["found"] = passages(text, query)
    return results


def search(query, url=SEARXNG_URL, timeout=TIMEOUT_S, figure=False):
    """The first results as [{title, url, content, date}]; with `figure`, those whose snippet
    holds a number come first (otherwise in the engines' order). Raises
    requests.RequestException."""
    # "auto": SearXNG picks the language from the query, so a question in English is searched in
    # English and one in Spanish in Spanish (a fixed "es" sent English questions to Spanish pages).
    r = requests.get(f"{url}/search", params={"q": query, "format": "json", "language": "auto"},
                     timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json().get("results") or []:
        content = " ".join((item.get("content") or "").split())
        if not item.get("url") or not (content or item.get("title")):
            continue
        out.append({
            "title": (item.get("title") or "").strip(),
            "url": item["url"],
            "content": content[:SNIPPET_CHARS],
            "date": (item.get("publishedDate") or "")[:10],
        })
        if len(out) == CANDIDATES:
            break
    if figure:
        out.sort(key=lambda r: not _NUMBER.search(r["content"]))   # stable: order kept otherwise
    return out[:MAX_RESULTS]


def search_fact(question, url=SEARXNG_URL, decided=None):
    """What goes to the model for this question: "" when it asks for no search; the results
    with their sources when it does; a plain "the search failed" when it could not be done, so
    the model says so instead of inventing an answer. `decided`: wants_search's answer when the
    caller already has it (the service asks once, to say "searching" first)."""
    if not (wants_search(question) if decided is None else decided):
        return ""
    query = search_query(question)
    t0 = time.monotonic()
    try:
        asked = _plain(question)
        results = search(query, url=url, figure=_has(asked, FIGURE))
        if results and (_has(asked, FIGURE) or _has(asked, WHEN)):
            read_pages(results, query)
    except (requests.RequestException, ValueError) as e:
        print(f"[search] {query!r} failed after {time.monotonic() - t0:.1f}s: {type(e).__name__}",
              flush=True)
        return ("A web search was needed for this question but it failed: say plainly that you "
                "could not search the internet right now, and do not guess.")
    print(f"[search] {query!r}: {len(results)} results in {time.monotonic() - t0:.1f}s", flush=True)
    if not results:
        return (f"A web search for \"{query}\" found nothing: say so, and do not guess.")
    lines = []
    for i, r in enumerate(results, 1):
        site = urlsplit(r["url"]).hostname or r["url"]
        site = site.removeprefix("www.")
        when = f", {r['date']}" if r["date"] else ""
        found = f" From the page: {r['found']}" if r.get("found") else ""
        lines.append(f"[{i}] {r['title']} — {r['content']}{found} ({site}{when})")
    # Today's date: without it the model placed "el 8 de noviembre" in 2025 and called it past
    # (2026-10-28, "cuándo viene el Papa"). And the question is about the web, not the picture:
    # it kept saying "no hay información en esta imagen".
    today = datetime.date.today().isoformat()
    return (f"Today is {today}. This question is about the web, not about the image: do not "
            "mention the image. Results of a web search made just now: " + " ".join(lines) +
            " Answer in two or three sentences with the fact or figure asked and the name of "
            "the site it comes from (the site, never a number like [4]); if the results do not "
            "answer it, say so.")
