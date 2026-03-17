from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import requests
import time
import json
import statistics
import math
import random
import os

# ── Optional RAGAS ────────────────────────────────────────────────
try:
    from ragas.metrics import answer_relevancy
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

app = FastAPI(title="LLM A/B Testing Framework")

MODEL   = "phi3:mini"
TIMEOUT = 300          # raised from 120 → 300s to fix ReadTimeout

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "queries.txt")

# ── Prompt Strategies ─────────────────────────────────────────────
PROMPTS = {
    "Simple":     "Explain {query} in simple terms.",
    "Technical":  "Provide a technical explanation of {query}.",
    "Example":    "Explain {query} with real-world examples.",
    "StepByStep": "Explain {query} step by step.",
    "Comparison": "Explain {query} and compare it with similar technologies.",
}

# ─────────────────────────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────────────────────────

class MetricsResult(BaseModel):
    relevance:     float = Field(..., ge=0, le=1)
    completeness:  float = Field(..., ge=0, le=1)
    coherence:     float = Field(..., ge=0, le=1)
    depth:         float = Field(..., ge=0, le=1)
    fluency:       float = Field(..., ge=0, le=1)
    latency_score: float = Field(..., ge=0, le=1)

    def average(self) -> float:
        v = [self.relevance, self.completeness, self.coherence,
             self.depth, self.fluency, self.latency_score]
        return round(sum(v) / len(v), 3)

    def as_dict(self) -> dict:
        return {
            "Relevance":    self.relevance,
            "Completeness": self.completeness,
            "Coherence":    self.coherence,
            "Depth":        self.depth,
            "Fluency":      self.fluency,
            "Latency":      self.latency_score,
        }


class SingleResult(BaseModel):
    query:     str
    answer_a:  str
    answer_b:  str
    metrics_a: MetricsResult
    metrics_b: MetricsResult
    avg_a:     float
    avg_b:     float
    latency_a: float
    latency_b: float
    winner:    str


class StatReport(BaseModel):
    n:             int
    mean_a:        float
    mean_b:        float
    std_a:         float
    std_b:         float
    wins_a:        int
    wins_b:        int
    ties:          int
    win_rate_a:    float
    win_rate_b:    float
    overall:       str
    p_value:       Optional[float] = None
    significant:   Optional[bool]  = None

# ─────────────────────────────────────────────────────────────────
#  Core Helpers
# ─────────────────────────────────────────────────────────────────

def clamp(v: float) -> float:
    return round(max(0.0, min(float(v), 1.0)), 3)


def load_queries() -> list[str]:
    try:
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return ["What is machine learning?", "Explain blockchain",
                "What is Docker?", "How does RAG work?",
                "What is a transformer model?"]


def random_queries(n: int = 5) -> list[str]:
    pool = load_queries()
    return random.sample(pool, min(n, len(pool)))


def generate(prompt: str) -> tuple[str, float]:
    start = time.time()
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["response"], time.time() - start


def evaluate_answer(query: str, answer: str, latency: float) -> MetricsResult:
    q_words = query.lower().split()
    a_words  = answer.lower().split()
    q_set, a_set = set(q_words), set(a_words)
    sentences    = answer.count(".") + answer.count("?") + answer.count("!")
    overlap      = len(q_set & a_set)
    unique_ratio = len(a_set) / max(len(a_words), 1)
    avg_sent     = len(a_words) / max(sentences, 1)
    return MetricsResult(
        relevance     = clamp(overlap / max(len(q_set), 1)),
        completeness  = clamp(len(a_words) / 150),
        coherence     = clamp(sentences / 10),
        depth         = clamp(unique_ratio * 2),
        fluency       = clamp(avg_sent / 18),
        latency_score = clamp(1 - latency / 60),
    )


def ragas_relevancy(query: str, answer: str) -> Optional[float]:
    if not RAGAS_AVAILABLE:
        return None
    try:
        ds = Dataset.from_dict({
            "question": [query], "answer": [answer], "contexts": [[""]]
        })
        result = evaluate(ds, metrics=[answer_relevancy])
        return round(float(result["answer_relevancy"]), 3)
    except Exception:
        return None


def welch_pvalue(a: list[float], b: list[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    se = math.sqrt((sa**2 / n) + (sb**2 / n))
    if se == 0:
        return None
    t  = (statistics.mean(a) - statistics.mean(b)) / se
    df = n - 1
    x  = df / (df + t**2)
    return min(1.0, round(2 * _ibeta(df / 2, 0.5, x), 4))


def _ibeta(a, b, x, iterations=200):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta) / a
    f, C, D = 1.0, 1.0, 0.0
    for m in range(iterations):
        for sign in (1, -1):
            if m == 0 and sign == 1: d = 1.0
            elif sign == 1: d = m*(b-m)*x / ((a+2*m-1)*(a+2*m))
            else: d = -(a+m)*(a+b+m)*x / ((a+2*m)*(a+2*m+1))
            D = 1 + d*D; D = D or 1e-30
            C = 1 + d/C; C = C or 1e-30
            D = 1/D; delta = C*D; f *= delta
            if abs(delta-1) < 1e-10: break
    return front*(f-1)


def llm_judge(query: str, a: str, b: str) -> str:
    prompt = (
        f"Compare these two answers to the same question.\n\n"
        f"Question: {query}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
        f"Give exactly 6 bullet points. End with a final verdict."
    )
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT,
    )
    return r.json()["response"]


# ─────────────────────────────────────────────────────────────────
#  Design System  (Research-terminal aesthetic)
#  Font: Space Mono (Google Fonts) — monospace, precise, instrument-like
#  Palette: Warm near-black bg, amber gold accent, red/blue prompt colours
# ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@700;800&display=swap');

:root {
  --bg:       #0b0a08;
  --bg2:      #111009;
  --bg3:      #18160f;
  --surface:  #1c1a12;
  --border:   #2e2b1e;
  --border2:  #3d3921;
  --gold:     #d4a847;
  --gold2:    #f0c95a;
  --amber:    #b8860b;
  --cream:    #f5eed8;
  --muted:    #7a7360;
  --dim:      #4a4535;
  --a:        #5b9bd5;   /* Prompt A  — steel blue  */
  --b:        #c96b6b;   /* Prompt B  — dusty rose  */
  --green:    #6aaa64;
  --win:      #8bc34a;
  --r: 4px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--cream);
  font-family: 'Space Mono', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.6;
  min-height: 100vh;
  padding: 0;
  background-image:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 39px,
      rgba(212,168,71,.03) 39px,
      rgba(212,168,71,.03) 40px
    );
}

/* ── Header ── */
.site-header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border2);
  padding: 22px 40px;
  display: flex;
  align-items: flex-end;
  gap: 20px;
}
.site-header h1 {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 800;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: var(--gold2);
  line-height: 1;
}
.site-header .version {
  font-size: 10px;
  color: var(--muted);
  letter-spacing: .15em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.header-rule {
  height: 2px;
  background: linear-gradient(90deg, var(--gold) 0%, transparent 60%);
  margin: 0;
}

/* ── Layout ── */
.main { max-width: 1080px; margin: 0 auto; padding: 36px 32px 80px; }

/* ── Section labels ── */
.section-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border2);
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 22px 24px;
  margin-bottom: 16px;
  position: relative;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--border2);
  border-radius: var(--r) 0 0 var(--r);
}
.card.accent-a::before { background: var(--a); }
.card.accent-b::before { background: var(--b); }
.card.accent-gold::before { background: var(--gold); }

/* ── Form controls ── */
label.lbl {
  display: block;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
input, textarea, select {
  width: 100%;
  padding: 9px 12px;
  background: var(--bg);
  color: var(--cream);
  border: 1px solid var(--border2);
  border-radius: var(--r);
  font-family: 'Space Mono', monospace;
  font-size: 12px;
  margin-bottom: 16px;
  transition: border-color .15s;
  appearance: none;
  -webkit-appearance: none;
}
select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23d4a847'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 32px;
  cursor: pointer;
}
input:focus, textarea:focus, select:focus {
  outline: none;
  border-color: var(--gold);
  box-shadow: 0 0 0 2px rgba(212,168,71,.12);
}
textarea { resize: vertical; min-height: 72px; }

.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* ── Buttons ── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 22px;
  border: none;
  border-radius: var(--r);
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all .15s;
}
.btn-primary {
  background: var(--gold);
  color: var(--bg);
}
.btn-primary:hover { background: var(--gold2); }
.btn-ghost {
  background: transparent;
  color: var(--gold);
  border: 1px solid var(--border2);
}
.btn-ghost:hover { border-color: var(--gold); background: rgba(212,168,71,.06); }

/* ── Mode tabs ── */
.mode-tabs { display: flex; gap: 0; margin-bottom: 20px; }
.mode-tab {
  flex: 1;
  padding: 10px 0;
  text-align: center;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  cursor: pointer;
  border: 1px solid var(--border2);
  background: var(--bg);
  color: var(--muted);
  transition: all .15s;
}
.mode-tab:first-child { border-radius: var(--r) 0 0 var(--r); }
.mode-tab:last-child  { border-radius: 0 var(--r) var(--r) 0; border-left: none; }
.mode-tab.active {
  background: var(--gold);
  color: var(--bg);
  border-color: var(--gold);
}

/* ── Query pills ── */
.query-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 99px;
  font-size: 11px;
  color: var(--cream);
  margin: 4px;
  cursor: pointer;
  transition: all .12s;
}
.query-pill:hover, .query-pill.selected {
  border-color: var(--gold);
  color: var(--gold);
  background: rgba(212,168,71,.08);
}
.query-pill .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--dim);
  transition: background .12s;
}
.query-pill.selected .dot { background: var(--gold); }

/* ── Stats grid ── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px 14px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.stat-box::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
}
.stat-box.col-a::after { background: var(--a); }
.stat-box.col-b::after { background: var(--b); }
.stat-box.col-gold::after { background: var(--gold); }
.stat-val {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  line-height: 1.1;
}
.stat-lbl { font-size: 9px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }
.col-a .stat-val { color: var(--a); }
.col-b .stat-val { color: var(--b); }
.col-gold .stat-val { color: var(--gold2); }

/* ── Metric bars ── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
.metric-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 12px 14px;
}
.metric-name { font-size: 9px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.metric-vals { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 6px; }
.bar-track { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 3px; }
.bar-fill { height: 100%; border-radius: 2px; transition: width .4s; }
.bar-a { background: var(--a); }
.bar-b { background: var(--b); }

/* ── Answer boxes ── */
.answer-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }
.answer-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
}
.answer-header {
  padding: 8px 14px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.answer-header.hdr-a { color: var(--a); }
.answer-header.hdr-b { color: var(--b); }
.answer-body {
  padding: 14px;
  font-size: 12px;
  line-height: 1.7;
  max-height: 200px;
  overflow-y: auto;
  color: var(--cream);
}

/* ── Table ── */
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { padding: 8px 12px; font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
     color: var(--muted); border-bottom: 1px solid var(--border2); text-align: left; font-weight: 400; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(212,168,71,.03); }

/* ── Badges ── */
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 99px;
  font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
}
.badge-a   { background: rgba(91,155,213,.15);  color: var(--a); }
.badge-b   { background: rgba(201,107,107,.15); color: var(--b); }
.badge-win { background: rgba(106,170,100,.15); color: var(--win); }
.badge-tie { background: rgba(122,115,96,.15);  color: var(--muted); }

/* ── Charts ── */
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px;
  height: 280px;
  position: relative;
}

/* ── Judge block ── */
pre {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 16px;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.7;
  color: var(--cream);
}

/* ── Significance ── */
.sig-ok  { color: var(--win); font-size: 11px; }
.sig-no  { color: var(--muted); font-size: 11px; }
.sig-row { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }

/* ── Notice ── */
.notice {
  display: flex; align-items: center; gap: 10px;
  background: rgba(184,134,11,.08);
  border: 1px solid rgba(184,134,11,.3);
  border-radius: var(--r);
  padding: 10px 14px;
  font-size: 11px;
  color: var(--gold);
  margin-top: 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── Back link ── */
.back { color: var(--gold); text-decoration: none; font-size: 11px; letter-spacing: .06em; }
.back:hover { color: var(--gold2); }

/* ── Separator ── */
.sep { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
"""


def page(title: str, body: str, extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
{extra_head}
</head>
<body>
<div class="header-rule"></div>
<header class="site-header">
  <div>
    <div class="version">LLM Evaluation Suite v2.0</div>
    <h1>Prompt A/B Testing Framework</h1>
  </div>
</header>
<main class="main">
{body}
</main>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/random-queries")
def api_random_queries():
    return JSONResponse({"queries": random_queries(5)})


@app.get("/", response_class=HTMLResponse)
def home():
    options_a = "".join(f"<option value='{k}'>{k}</option>" for k in PROMPTS)
    options_b = "".join(
        f"<option value='{k}' {'selected' if k == 'Technical' else ''}>{k}</option>"
        for k in PROMPTS
    )

    # Pre-load 5 random queries for multi mode
    initial_queries = random_queries(5)
    pills_html = "".join(
        f'<span class="query-pill" onclick="togglePill(this)">'
        f'<span class="dot"></span>{q}</span>'
        for q in initial_queries
    )

    body = f"""
<form method="post" action="/run" id="mainForm">

  <!-- ── Query Mode ─────────────────────────── -->
  <p class="section-label">01 — Query Configuration</p>
  <div class="card accent-gold">
    <label class="lbl">Testing Mode</label>

    <div class="mode-tabs">
      <div class="mode-tab active" id="tab-single" onclick="setMode('single')">
        Single Query
      </div>
      <div class="mode-tab" id="tab-multi" onclick="setMode('multi')">
        Multi-Query (from dataset)
      </div>
    </div>

    <input type="hidden" name="mode_type" id="mode_type" value="single">

    <!-- Single mode -->
    <div id="panel-single">
      <label class="lbl">Your Query</label>
      <input name="single_query" id="single_query"
             placeholder="e.g. What is a transformer model?" />
    </div>

    <!-- Multi mode -->
    <div id="panel-multi" style="display:none">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <label class="lbl" style="margin:0">
          Randomly selected from <code style="color:var(--gold)">queries.txt</code>
          ({len(load_queries())} queries)
        </label>
        <button type="button" class="btn btn-ghost" onclick="refreshQueries()" style="padding:6px 14px">
          ↻ Shuffle
        </button>
      </div>
      <div id="pills-container">{pills_html}</div>
      <input type="hidden" name="selected_queries" id="selected_queries" value="">
      <p style="color:var(--muted);font-size:10px;margin-top:10px">
        Click to select/deselect queries. Selected queries run against both prompts.
      </p>
    </div>
  </div>

  <!-- ── Prompt Strategy ───────────────────── -->
  <p class="section-label">02 — Prompt Configuration</p>
  <div class="card">
    <label class="lbl">Input Mode</label>
    <div class="mode-tabs" style="margin-bottom:20px">
      <div class="mode-tab active" id="tab-manual" onclick="setPromptMode('manual')">Manual Prompts</div>
      <div class="mode-tab" id="tab-preset" onclick="setPromptMode('preset')">Preset Strategies</div>
    </div>
    <input type="hidden" name="prompt_mode" id="prompt_mode" value="manual">

    <!-- Manual -->
    <div id="panel-manual">
      <p style="color:var(--muted);font-size:11px;margin-bottom:14px">
        Use <code style="color:var(--gold)">{{query}}</code> as placeholder.
      </p>
      <div class="row2">
        <div>
          <div class="card accent-a" style="padding:14px">
            <label class="lbl" style="color:var(--a)">Prompt A</label>
            <textarea name="promptA" placeholder="Explain {{query}} in simple terms."></textarea>
          </div>
        </div>
        <div>
          <div class="card accent-b" style="padding:14px">
            <label class="lbl" style="color:var(--b)">Prompt B</label>
            <textarea name="promptB" placeholder="Give a technical explanation of {{query}}."></textarea>
          </div>
        </div>
      </div>
    </div>

    <!-- Preset -->
    <div id="panel-preset" style="display:none">
      <div class="row2">
        <div>
          <div class="card accent-a" style="padding:14px">
            <label class="lbl" style="color:var(--a)">Strategy A</label>
            <select name="autoA">{options_a}</select>
            <p id="preview-a" style="font-size:11px;color:var(--muted);margin-top:-8px"></p>
          </div>
        </div>
        <div>
          <div class="card accent-b" style="padding:14px">
            <label class="lbl" style="color:var(--b)">Strategy B</label>
            <select name="autoB">{options_b}</select>
            <p id="preview-b" style="font-size:11px;color:var(--muted);margin-top:-8px"></p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Submit ────────────────────────────── -->
  <button type="submit" class="btn btn-primary" onclick="prepareSubmit()">
    ▶ Run Experiment
  </button>

</form>

{"" if RAGAS_AVAILABLE else '<div class="notice" style="margin-top:16px">ℹ No RAGAS detected — using heuristic metrics. Run <code>pip install ragas datasets</code> to enable answer_relevancy.</div>'}

<script>
const PROMPTS = {json.dumps(PROMPTS)};

// ── Mode switching ─────────────────
function setMode(m) {{
  document.getElementById('mode_type').value = m;
  document.getElementById('panel-single').style.display = m === 'single' ? 'block' : 'none';
  document.getElementById('panel-multi').style.display  = m === 'multi'  ? 'block' : 'none';
  document.getElementById('tab-single').className = 'mode-tab' + (m === 'single' ? ' active' : '');
  document.getElementById('tab-multi').className  = 'mode-tab' + (m === 'multi'  ? ' active' : '');
}}

function setPromptMode(m) {{
  document.getElementById('prompt_mode').value = m;
  document.getElementById('panel-manual').style.display = m === 'manual' ? 'block' : 'none';
  document.getElementById('panel-preset').style.display = m === 'preset' ? 'block' : 'none';
  document.getElementById('tab-manual').className = 'mode-tab' + (m === 'manual' ? ' active' : '');
  document.getElementById('tab-preset').className = 'mode-tab' + (m === 'preset' ? ' active' : '');
}}

// ── Preset previews ─────────────────
function updatePreviews() {{
  const a = document.querySelector('[name=autoA]').value;
  const b = document.querySelector('[name=autoB]').value;
  document.getElementById('preview-a').textContent = PROMPTS[a] || '';
  document.getElementById('preview-b').textContent = PROMPTS[b] || '';
}}
document.querySelector('[name=autoA]').addEventListener('change', updatePreviews);
document.querySelector('[name=autoB]').addEventListener('change', updatePreviews);
updatePreviews();

// ── Query pills ─────────────────────
function togglePill(el) {{
  el.classList.toggle('selected');
}}

async function refreshQueries() {{
  const res = await fetch('/api/random-queries');
  const data = await res.json();
  const c = document.getElementById('pills-container');
  c.innerHTML = data.queries.map(q =>
    `<span class="query-pill" onclick="togglePill(this)"><span class="dot"></span>${{q}}</span>`
  ).join('');
}}

// ── Prepare submit ──────────────────
function prepareSubmit() {{
  if (document.getElementById('mode_type').value === 'multi') {{
    const pills = document.querySelectorAll('.query-pill.selected');
    const qs = Array.from(pills).map(p => p.textContent.trim());
    if (qs.length === 0) {{
      // select all if none chosen
      document.querySelectorAll('.query-pill').forEach(p => p.classList.add('selected'));
    }}
    const finalPills = document.querySelectorAll('.query-pill.selected');
    document.getElementById('selected_queries').value =
      Array.from(finalPills).map(p => p.textContent.trim()).join('\\n');
  }}
}}
</script>
"""
    return page("LLM A/B Testing Framework", body)


@app.post("/run", response_class=HTMLResponse)
def run(
    mode_type:        str = Form("single"),
    single_query:     str = Form(""),
    selected_queries: str = Form(""),
    prompt_mode:      str = Form("manual"),
    promptA:          str = Form(""),
    promptB:          str = Form(""),
    autoA:            str = Form("Simple"),
    autoB:            str = Form("Technical"),
):
    # ── Build query list ──────────────────────────────────────────
    if mode_type == "single":
        query_list = [single_query.strip()] if single_query.strip() else []
    else:
        query_list = [q.strip() for q in selected_queries.splitlines() if q.strip()]

    if not query_list:
        return page("Error", "<p style='color:var(--b);padding:40px'>No queries provided. Go back and enter at least one query.</p>")

    # ── Resolve templates ─────────────────────────────────────────
    if prompt_mode == "preset":
        template_a, template_b = PROMPTS[autoA], PROMPTS[autoB]
        label_a, label_b = autoA, autoB
    else:
        template_a, template_b = promptA.strip(), promptB.strip()
        label_a, label_b = "Custom A", "Custom B"

    if not template_a or not template_b:
        return page("Error", "<p style='color:var(--b);padding:40px'>Both prompts are required.</p>")

    # ── Run ───────────────────────────────────────────────────────
    results: list[SingleResult] = []
    scores_a, scores_b = [], []
    ragas_a,  ragas_b  = [], []

    for query in query_list:
        try:
            ans_a, lat_a = generate(template_a.replace("{query}", query))
            ans_b, lat_b = generate(template_b.replace("{query}", query))
        except requests.exceptions.ReadTimeout:
            return page("Timeout Error", f"""
            <p style="color:var(--b);font-size:14px;margin-bottom:12px">
              ⚠ Ollama timed out after {TIMEOUT}s.
            </p>
            <p style="color:var(--muted)">
              Possible fixes:<br>
              &nbsp;• Make sure Ollama is running: <code>ollama serve</code><br>
              &nbsp;• Check the model is pulled: <code>ollama pull {MODEL}</code><br>
              &nbsp;• Try a faster model like <code>tinyllama</code>
            </p>
            <br><a href="/" class="back">← Back to form</a>
            """)

        m_a = evaluate_answer(query, ans_a, lat_a)
        m_b = evaluate_answer(query, ans_b, lat_b)
        avg_a, avg_b = m_a.average(), m_b.average()
        scores_a.append(avg_a); scores_b.append(avg_b)

        r_a = ragas_relevancy(query, ans_a)
        r_b = ragas_relevancy(query, ans_b)
        if r_a is not None:
            ragas_a.append(r_a); ragas_b.append(r_b)

        results.append(SingleResult(
            query=query, answer_a=ans_a, answer_b=ans_b,
            metrics_a=m_a, metrics_b=m_b,
            avg_a=avg_a, avg_b=avg_b,
            latency_a=round(lat_a, 2), latency_b=round(lat_b, 2),
            winner="A" if avg_a > avg_b else ("B" if avg_b > avg_a else "Tie"),
        ))

    # ── Stats ─────────────────────────────────────────────────────
    n      = len(scores_a)
    mean_a = round(statistics.mean(scores_a), 3)
    mean_b = round(statistics.mean(scores_b), 3)
    std_a  = round(statistics.stdev(scores_a), 3) if n > 1 else 0.0
    std_b  = round(statistics.stdev(scores_b), 3) if n > 1 else 0.0
    wins_a = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
    wins_b = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
    ties   = n - wins_a - wins_b
    pval   = welch_pvalue(scores_a, scores_b) if n > 1 else None
    sig    = pval < 0.05 if pval is not None else None
    winner = "A" if mean_a > mean_b else ("B" if mean_b > mean_a else "Tie")

    # ── Chart data (avg per metric) ───────────────────────────────
    metric_keys = ["relevance","completeness","coherence","depth","fluency","latency_score"]
    metric_labels = ["Relevance","Completeness","Coherence","Depth","Fluency","Latency"]

    def avg_metrics(side: str) -> list[float]:
        return [
            round(sum(getattr(r.metrics_a if side == "a" else r.metrics_b, k)
                      for r in results) / n, 3)
            for k in metric_keys
        ]

    chart_labels = json.dumps(metric_labels)
    chart_a = json.dumps(avg_metrics("a"))
    chart_b = json.dumps(avg_metrics("b"))

    # ── Per-query table rows ──────────────────────────────────────
    table_rows = ""
    for i, r in enumerate(results):
        badge = (
            f"<span class='badge badge-a'>A</span>" if r.winner == "A" else
            f"<span class='badge badge-b'>B</span>" if r.winner == "B" else
            f"<span class='badge badge-tie'>Tie</span>"
        )
        table_rows += f"""
        <tr>
          <td style="color:var(--muted)">{i+1:02d}</td>
          <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
              title="{r.query}">{r.query}</td>
          <td style="color:var(--a)">{r.avg_a}</td>
          <td style="color:var(--b)">{r.avg_b}</td>
          <td>{badge}</td>
          <td style="color:var(--muted)">{r.latency_a}s / {r.latency_b}s</td>
        </tr>"""

    # ── Per-query detail blocks ───────────────────────────────────
    detail_html = ""
    for i, r in enumerate(results):
        metrics_html = ""
        for k, lbl in zip(metric_keys, metric_labels):
            va = getattr(r.metrics_a, k)
            vb = getattr(r.metrics_b, k)
            metrics_html += f"""
            <div class="metric-box">
              <div class="metric-name">{lbl}</div>
              <div class="metric-vals">
                <span style="color:var(--a)">{va}</span>
                <span style="color:var(--b)">{vb}</span>
              </div>
              <div class="bar-track"><div class="bar-fill bar-a" style="width:{int(va*100)}%"></div></div>
              <div class="bar-track"><div class="bar-fill bar-b" style="width:{int(vb*100)}%"></div></div>
            </div>"""

        winner_badge = (
            f"<span class='badge badge-a'>A wins</span>" if r.winner == "A" else
            f"<span class='badge badge-b'>B wins</span>" if r.winner == "B" else
            f"<span class='badge badge-tie'>Tie</span>"
        )
        detail_html += f"""
        <div class="card" style="margin-bottom:14px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
            <span style="font-size:10px;letter-spacing:.12em;color:var(--muted)">
              QUERY {i+1:02d}
            </span>
            <span style="font-size:12px;color:var(--cream);flex:1;margin:0 14px;
                         overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
              {r.query}
            </span>
            {winner_badge}
          </div>
          <div class="answer-wrap">
            <div class="answer-card">
              <div class="answer-header hdr-a">
                <span>Prompt A — {label_a}</span>
                <span>avg {r.avg_a} · {r.latency_a}s</span>
              </div>
              <div class="answer-body">{r.answer_a}</div>
            </div>
            <div class="answer-card">
              <div class="answer-header hdr-b">
                <span>Prompt B — {label_b}</span>
                <span>avg {r.avg_b} · {r.latency_b}s</span>
              </div>
              <div class="answer-body">{r.answer_b}</div>
            </div>
          </div>
          <div class="metric-grid">{metrics_html}</div>
        </div>"""

    # ── RAGAS section ─────────────────────────────────────────────
    ragas_html = ""
    if ragas_a:
        ra = round(sum(ragas_a) / len(ragas_a), 3)
        rb = round(sum(ragas_b) / len(ragas_b), 3)
        ragas_html = f"""
        <p class="section-label">RAGAS Answer Relevancy</p>
        <div class="card accent-gold">
          <div class="stat-grid" style="grid-template-columns:1fr 1fr">
            <div class="stat-box col-a">
              <div class="stat-val">{ra}</div>
              <div class="stat-lbl">Prompt A — RAGAS Relevancy</div>
            </div>
            <div class="stat-box col-b">
              <div class="stat-val">{rb}</div>
              <div class="stat-lbl">Prompt B — RAGAS Relevancy</div>
            </div>
          </div>
        </div>"""

    # ── Significance line ─────────────────────────────────────────
    sig_html = ""
    if pval is not None:
        if sig:
            sig_html = f'<div class="sig-row"><span class="sig-ok">✓ Welch\'s t-test p = {pval} — statistically significant (p &lt; 0.05)</span></div>'
        else:
            sig_html = f'<div class="sig-row"><span class="sig-no">Welch\'s t-test p = {pval} — not statistically significant (p ≥ 0.05)</span></div>'

    # ── Winner badge ──────────────────────────────────────────────
    win_badge_cls = "badge-a" if winner == "A" else ("badge-b" if winner == "B" else "badge-tie")
    win_label     = f"Prompt {winner}" if winner in ("A", "B") else "Tie"

    # ── LLM judge ────────────────────────────────────────────────
    try:
        last  = results[-1]
        judge = llm_judge(last.query, last.answer_a, last.answer_b)
    except Exception as e:
        judge = f"Judge unavailable: {e}"

    body = f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px">
  <div>
    <p class="section-label" style="margin:0">Experiment Results</p>
    <div style="margin-top:4px;display:flex;gap:10px;align-items:center">
      <span class="badge badge-a">A: {label_a}</span>
      <span class="badge badge-b">B: {label_b}</span>
      <span style="font-size:10px;color:var(--muted)">{n} {"query" if n == 1 else "queries"}</span>
    </div>
  </div>
  <a href="/" class="back">← New experiment</a>
</div>

<!-- ── Statistical Report ── -->
<p class="section-label">01 — Statistical Comparison Report</p>
<div class="card accent-gold">
  <div class="stat-grid">
    <div class="stat-box col-a">
      <div class="stat-val">{mean_a}</div>
      <div class="stat-lbl">Prompt A — Mean</div>
    </div>
    <div class="stat-box col-b">
      <div class="stat-val">{mean_b}</div>
      <div class="stat-lbl">Prompt B — Mean</div>
    </div>
    <div class="stat-box col-a">
      <div class="stat-val">{std_a}</div>
      <div class="stat-lbl">Prompt A — Std Dev</div>
    </div>
    <div class="stat-box col-b">
      <div class="stat-val">{std_b}</div>
      <div class="stat-lbl">Prompt B — Std Dev</div>
    </div>
    <div class="stat-box col-a">
      <div class="stat-val">{int(wins_a / max(n,1) * 100)}%</div>
      <div class="stat-lbl">Prompt A — Win Rate</div>
    </div>
    <div class="stat-box col-b">
      <div class="stat-val">{int(wins_b / max(n,1) * 100)}%</div>
      <div class="stat-lbl">Prompt B — Win Rate</div>
    </div>
    <div class="stat-box col-gold">
      <div class="stat-val">{wins_a}/{wins_b}/{ties}</div>
      <div class="stat-lbl">W·A / W·B / Ties</div>
    </div>
    <div class="stat-box col-gold">
      <div class="stat-val"><span class="badge {win_badge_cls}">{win_label}</span></div>
      <div class="stat-lbl">Overall Winner</div>
    </div>
  </div>
  {sig_html}
</div>

{ragas_html}

<!-- ── Per-query table (multi only) ── -->
{"" if n <= 1 else f'''
<p class="section-label">02 — Per-Query Breakdown</p>
<div class="card">
  <table>
    <thead><tr>
      <th>#</th><th>Query</th>
      <th style="color:var(--a)">Score A</th>
      <th style="color:var(--b)">Score B</th>
      <th>Winner</th><th>Latency A/B</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
'''}

<!-- ── Charts ── -->
<p class="section-label">{"03" if n > 1 else "02"} — Metric Visualisation</p>
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <select id="chartMode" onchange="updateCharts()"
            style="width:auto;margin:0;padding:6px 12px">
      <option value="both">Both Prompts</option>
      <option value="A">Prompt A only</option>
      <option value="B">Prompt B only</option>
    </select>
    <div style="display:flex;gap:16px;font-size:11px">
      <span style="color:var(--a)">■ Prompt A</span>
      <span style="color:var(--b)">■ Prompt B</span>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><canvas id="radarChart"></canvas></div>
    <div class="chart-box"><canvas id="barChart"></canvas></div>
  </div>
</div>

<!-- ── Query Detail ── -->
<p class="section-label">{"04" if n > 1 else "03"} — Query Detail</p>
{detail_html}

<!-- ── LLM Judge ── -->
<p class="section-label">{"05" if n > 1 else "04"} — LLM Judge (last query)</p>
<div class="card accent-gold">
  <p style="font-size:10px;color:var(--muted);margin-bottom:12px;letter-spacing:.1em">
    EVALUATED QUERY: <em style="color:var(--cream)">{results[-1].query}</em>
  </p>
  <pre>{judge}</pre>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const labels = {chart_labels};
const dA = {chart_a};
const dB = {chart_b};
const gridColor = 'rgba(46,43,30,.8)';
const tickColor = '#4a4535';

Chart.defaults.color = '#7a7360';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;

const radar = new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels,
    datasets: [
      {{ label:'Prompt A', data:dA, borderColor:'#5b9bd5',
         backgroundColor:'rgba(91,155,213,.08)',
         pointBackgroundColor:'#5b9bd5', pointRadius:3 }},
      {{ label:'Prompt B', data:dB, borderColor:'#c96b6b',
         backgroundColor:'rgba(201,107,107,.08)',
         pointBackgroundColor:'#c96b6b', pointRadius:3 }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    scales: {{ r: {{
      min:0, max:1, ticks:{{ stepSize:.2, color:tickColor, backdropColor:'transparent' }},
      grid:{{ color:gridColor }},
      pointLabels:{{ color:'#f5eed8', font:{{ size:10 }} }},
      angleLines:{{ color:gridColor }}
    }} }},
    plugins:{{ legend:{{ labels:{{ color:'#f5eed8', boxWidth:10 }} }} }}
  }}
}});

const bar = new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [
      {{ label:'Prompt A', data:dA, backgroundColor:'rgba(91,155,213,.7)',
         borderColor:'#5b9bd5', borderWidth:1, borderRadius:2 }},
      {{ label:'Prompt B', data:dB, backgroundColor:'rgba(201,107,107,.7)',
         borderColor:'#c96b6b', borderWidth:1, borderRadius:2 }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    scales: {{
      y: {{ min:0, max:1, ticks:{{ stepSize:.2, color:tickColor }},
           grid:{{ color:gridColor }}, border:{{ color:'transparent' }} }},
      x: {{ ticks:{{ color:tickColor }},
           grid:{{ color:'transparent' }}, border:{{ color:gridColor }} }},
    }},
    plugins:{{ legend:{{ labels:{{ color:'#f5eed8', boxWidth:10 }} }} }}
  }}
}});

function updateCharts() {{
  const m = document.getElementById('chartMode').value;
  radar.data.datasets[0].hidden = (m === 'B');
  radar.data.datasets[1].hidden = (m === 'A');
  bar.data.datasets[0].hidden   = (m === 'B');
  bar.data.datasets[1].hidden   = (m === 'A');
  radar.update(); bar.update();
}}
</script>
"""
    return page("Results — LLM A/B Testing", body,
                extra_head='<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>')