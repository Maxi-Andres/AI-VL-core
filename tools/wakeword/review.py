"""A page to listen to the G1's recordings and compare how each method heard them.

    .venv/bin/python tools/wakeword/review.py            # then open http://127.0.0.1:8770

Every clip of ~/.pilot/wakeword/*/labels.jsonl with its audio, its label, what Whisper heard with
and without the "Bender" hotword, each openWakeWord model's score, and whether each method woke
(green when it was right, red when wrong) — from bench.py's bench.json. A clip's label is set
here too (buttons, or the keys B / N / U), and saved to its labels.jsonl at once: the label is
what every number is measured against, so it is a person's call, made listening.

The recordings are the office's own talk: the server listens on 127.0.0.1 only unless --host says
otherwise, and serves nothing but WAVs from that folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RECORDED = Path.home() / ".pilot" / "wakeword"
NAME = re.compile(r"^[\w.-]+$")
LABELS = ("bender", "not", "unlabeled")


def load(recorded: Path) -> dict:
    bench_path = recorded / "bench.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {"methods": {}, "clips": []}
    cache_path = recorded / "whisper_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    scored = {(c["session"], c["wav"]): c for c in bench["clips"]}
    clips = []
    for labels in sorted(recorded.glob("*/labels.jsonl")):
        for line in labels.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            b = scored.get((labels.parent.name, row["wav"]), {})
            clips.append({"session": labels.parent.name, "wav": row["wav"],
                          "seconds": row.get("seconds"), "label": row["label"],
                          "heard": b.get("heard") or cache.get(row["wav"]) or
                          {"hotword": row.get("whisper_hotword", "")},
                          "scores": b.get("scores", {}), "woke": b.get("woke", {})})
    return {"methods": bench["methods"], "clips": clips}


def set_label(recorded: Path, session: str, wav: str, label: str) -> bool:
    path = recorded / session / "labels.jsonl"
    if not (NAME.match(session) and NAME.match(wav) and label in LABELS and path.exists()):
        return False
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    found = False
    for row in rows:
        if row["wav"] == wav:
            row["label"], row["why"], found = label, "labelled by listening (review.py)", True
    if not found:
        return False
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".labels-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(tmp, path)
    return True


class Handler(BaseHTTPRequestHandler):
    recorded: Path = RECORDED

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/data":
            return self._send(200, json.dumps(load(self.recorded)).encode(), "application/json")
        m = re.match(r"^/audio/([\w.-]+)/([\w.-]+\.wav)$", self.path)
        if m:
            path = self.recorded / m.group(1) / m.group(2)
            if path.is_file() and path.resolve().is_relative_to(self.recorded.resolve()):
                return self._send(200, path.read_bytes(), "audio/wav")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/label":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            ok = set_label(self.recorded, str(body["session"]), str(body["wav"]),
                           str(body["label"]))
        except (ValueError, KeyError):
            ok = False
        return self._send(200 if ok else 400, json.dumps({"ok": ok}).encode(), "application/json")

    def log_message(self, *args) -> None:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wake word review</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dd;--ok:#1f7a3f;--okbg:#e4f3e8;--bad:#b3261e;--badbg:#fbe5e3;--acc:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#33332f;--ok:#6fcf8d;--okbg:#17321f;--bad:#f08a80;--badbg:#3a1b18;--acc:#7f9cf5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:16px;position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:1}
h1{font-size:18px;margin:0 0 8px}table{border-collapse:collapse;font-size:13px}td,th{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
select,button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
main{padding:12px 16px;display:grid;gap:8px;max-width:1100px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:grid;gap:6px}
.clip.sel{outline:2px solid var(--acc)}.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
audio{height:32px;max-width:100%}.name{color:var(--muted);font-size:12px}
.lab button[data-on=true]{background:var(--acc);color:#fff;border-color:var(--acc)}
.heard{display:grid;grid-template-columns:150px 1fr;gap:2px 10px;font-size:13px}.heard b{color:var(--muted);font-weight:500}
.woke{display:flex;gap:6px;flex-wrap:wrap}.tag{font-size:12px;padding:1px 8px;border-radius:10px;border:1px solid var(--line)}
.tag.ok{background:var(--okbg);color:var(--ok)}.tag.bad{background:var(--badbg);color:var(--bad)}
.hint{color:var(--muted);font-size:12px}
</style></head><body>
<header><h1>Wake word review</h1><div id="sum"></div>
<div class="bar"><label>Show <select id="filter"><option value="all">all</option><option value="bender">labelled bender</option>
<option value="not">labelled not</option><option value="unlabeled">unlabeled</option><option value="wrong">some method wrong</option></select></label>
<span class="hint">Keys: J/K next/previous, Space play, B = bender, N = not, U = unlabeled</span></div></header>
<main id="list"></main>
<script>
let data={methods:{},clips:[]},sel=0,shown=[];
const $=s=>document.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function load(){data=await (await fetch("/data")).json();
 const ms=Object.keys(data.methods);
 $("#sum").innerHTML=ms.length?`<table><tr><th>method</th><th>names caught</th><th>false wakes (clips)</th><th>false wakes / hour (stream)</th></tr>`+
  ms.map(m=>{const s=data.methods[m];return `<tr><td>${esc(m)}</td><td>${esc(s.recall)}</td><td>${esc(s.false_wakes)}</td><td>${s.stream?esc(s.stream.per_hour+" ("+s.stream.hours+" h)"):"-"}</td></tr>`}).join("")+`</table>`
  :`<span class="hint">No bench.json yet: run bench.py.</span>`;
 render();}
function right(c,m){const w=c.woke[m];if(w===undefined||c.label==="unlabeled")return null;return w===(c.label==="bender")}
function render(){const f=$("#filter").value;
 shown=data.clips.filter(c=>f==="all"||c.label===f||(f==="wrong"&&Object.keys(c.woke).some(m=>right(c,m)===false)));
 sel=Math.min(sel,Math.max(0,shown.length-1));
 $("#list").innerHTML=shown.map((c,i)=>`<div class="clip ${i===sel?"sel":""}" data-i="${i}">
 <div class="top"><audio controls preload="none" src="/audio/${esc(c.session)}/${esc(c.wav)}"></audio>
 <span class="lab">${["bender","not","unlabeled"].map(l=>`<button data-l="${l}" data-on="${c.label===l}">${l}</button>`).join(" ")}</span>
 <span class="name">${esc(c.session)}/${esc(c.wav)} · ${c.seconds??"?"} s</span></div>
 <div class="heard"><b>Whisper, steered to Bender</b><span>${esc(c.heard.hotword)}</span><b>Whisper, plain</b><span>${esc(c.heard.plain??"(not run yet)")}</span>
 ${Object.entries(c.scores).map(([k,v])=>`<b>${esc(k)}</b><span>${v}</span>`).join("")}</div>
 <div class="woke">${Object.keys(c.woke).map(m=>{const r=right(c,m);return `<span class="tag ${r===null?"":r?"ok":"bad"}">${esc(m)}: ${c.woke[m]?"woke":"quiet"}</span>`}).join("")}</div></div>`).join("");
 document.querySelectorAll(".clip").forEach(el=>{el.onclick=e=>{sel=+el.dataset.i;mark();
  const b=e.target.closest("button[data-l]");if(b)label(shown[sel],b.dataset.l);}});}
function mark(){document.querySelectorAll(".clip").forEach(el=>el.classList.toggle("sel",+el.dataset.i===sel));
 const el=document.querySelector(`.clip[data-i="${sel}"]`);if(el)el.scrollIntoView({block:"nearest"});}
async function label(c,l){const r=await fetch("/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session:c.session,wav:c.wav,label:l})});
 if((await r.json()).ok){c.label=l;render();mark();}}
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;const c=shown[sel];
 if(e.key==="j"){sel=Math.min(shown.length-1,sel+1);mark();}else if(e.key==="k"){sel=Math.max(0,sel-1);mark();}
 else if(e.key===" "&&c){e.preventDefault();const a=document.querySelector(`.clip[data-i="${sel}"] audio`);a.paused?a.play():a.pause();}
 else if(c&&"bnu".includes(e.key)){label(c,{b:"bender",n:"not",u:"unlabeled"}[e.key]);}});
$("#filter").onchange=render;load();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recorded", type=Path, default=RECORDED)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()
    Handler.recorded = a.recorded
    print(f"http://{a.host}:{a.port}  ({a.recorded})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
