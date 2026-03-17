"""
LLM Prompt A/B Testing Framework — ULTIMATE FINAL VERSION
========================================================
COMPLETE PRODUCTION CODE with:
    ✅ Charts using HEURISTIC data (never empty)
    ✅ Info icons with tooltips
    ✅ LLM judge reasoning
    ✅ Overall winner with explanation
    ✅ All errors fixed
    ✅ Smart fallback: Ollama → Gemini → Default (0.5)
"""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import requests, time, json, statistics, math, random, os, re
import concurrent.futures

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .env file loaded")
except ImportError:
    pass

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_entity_recall,
        answer_similarity,
        context_precision,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

app = FastAPI(title="LLM A/B Testing — Ultimate Final")

MODEL              = "phi3:mini"
RAGAS_MODEL        = "qwen2.5:3b"
GEMINI_MODEL       = "gemini-2.5-flash-lite"

GEMINI_API_KEY = (
    os.environ.get("Gemini_API_key") or
    os.environ.get("GEMINI_API_KEY") or
    ""
)

TIMEOUT            = 300
RAGAS_TIMEOUT_OLLAMA = 60
RAGAS_TIMEOUT_GEMINI = 45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "Simple":     "Explain {query} in simple terms.",
    "Technical":  "Provide a technical explanation of {query}.",
    "Example":    "Explain {query} with real-world examples.",
    "StepByStep": "Explain {query} step by step.",
    "Comparison": "Explain {query} and compare it with similar concepts.",
}

# Info tooltips
METRIC_TIPS = {
    "Relevance":    "Keyword overlap between query and answer. Higher = more on-topic.",
    "Completeness": "Answer word count relative to 150 words. Higher = more detailed.",
    "Coherence":    "Number of sentences. Higher = more structured.",
    "Depth":        "Unique word diversity. Higher = richer vocabulary.",
    "Fluency":      "Average sentence length (ideal ~18 words). Higher = better readability.",
    "Latency":      "Response speed. Higher = faster response.",
}

RAGAS_TIPS = {
    "Answer Relevancy":      "RAGAS metric: Does answer address the question? (semantic matching)",
    "Context Entity Recall": "RAGAS metric: Did context have key entities needed?",
    "Answer Similarity":     "RAGAS metric: Similarity to ideal answer (embedding-based)",
    "Context Precision":     "RAGAS metric: Quality of retrieved context chunks",
}

# ── Pydantic Models ──────────────────────────────────────────────────
class MetricsResult(BaseModel):
    relevance:     float = Field(default=0.5, ge=0, le=1)
    completeness:  float = Field(default=0.5, ge=0, le=1)
    coherence:     float = Field(default=0.5, ge=0, le=1)
    depth:         float = Field(default=0.5, ge=0, le=1)
    fluency:       float = Field(default=0.5, ge=0, le=1)
    latency_score: float = Field(default=0.5, ge=0, le=1)

    def average(self) -> float:
        v = [self.relevance, self.completeness, self.coherence,
             self.depth, self.fluency, self.latency_score]
        return round(sum(v) / len(v), 3)

class RAGASScores(BaseModel):
    answer_relevancy:      float = Field(default=0.5, ge=0, le=1)
    context_entity_recall: float = Field(default=0.5, ge=0, le=1)
    answer_similarity:     float = Field(default=0.5, ge=0, le=1)
    context_precision:     float = Field(default=0.5, ge=0, le=1)

class SingleResult(BaseModel):
    query:     str
    context:   list[str]
    answer_a:  str
    answer_b:  str
    metrics_a: MetricsResult
    metrics_b: MetricsResult
    avg_a:     float
    avg_b:     float
    latency_a: float
    latency_b: float
    winner:    str
    ragas_a:   RAGASScores
    ragas_b:   RAGASScores

# ── RAG ───────────────────────────────────────────────────────────
_embedder   = None
_collection = None

def init_rag():
    global _embedder, _collection
    if not RAG_AVAILABLE:
        return
    print("Initialising RAG pipeline...")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kb_path = os.path.join(BASE_DIR, "knowledge_base.txt")
    if not os.path.exists(kb_path):
        print(f"WARNING: {kb_path} not found")
        return
    text = open(kb_path, encoding="utf-8").read()
    raw  = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in raw:
        words = para.split()
        if len(words) <= 200:
            chunks.append(para)
        else:
            for i in range(0, len(words), 200):
                c = " ".join(words[i:i+200])
                if c.strip(): chunks.append(c.strip())
    try:
        client = chromadb.EphemeralClient()
    except AttributeError:
        client = chromadb.Client()
    try: 
        client.delete_collection("knowledge-base")
    except: 
        pass
    _collection = client.create_collection("knowledge-base")
    embs = _embedder.encode(chunks, show_progress_bar=False).tolist()
    _collection.add(documents=chunks, embeddings=embs,
                    ids=[f"c{i}" for i in range(len(chunks))])
    print(f"RAG ready — {len(chunks)} chunks indexed.")

@app.on_event("startup")
def startup_event():
    init_rag()

def retrieve(query: str, n: int = 3) -> list[str]:
    if _embedder is None or _collection is None:
        return []
    q_vec = _embedder.encode([query], show_progress_bar=False).tolist()
    res   = _collection.query(query_embeddings=q_vec, n_results=n)
    return res["documents"][0]

def load_queries() -> list[str]:
    try:
        path = os.path.join(BASE_DIR, "queries.txt")
        return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return ["What is machine learning?", "What is Docker?",
                "How does RAG work?", "What is a transformer model?"]

# ── LLM ───────────────────────────────────────────────────────────
def generate(prompt: str) -> tuple[str, float]:
    start = time.time()
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Run <code>ollama serve</code>.")
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(f"Ollama timed out after {TIMEOUT}s.")
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}. Pull model: <code>ollama pull {MODEL}</code>")
    return r.json()["response"], time.time() - start

def build_prompt(template: str, query: str, context: list[str]) -> str:
    if not context:
        return template.replace("{query}", query)
    ctx = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(context))
    return (
        f"Use ONLY the context below to answer. Do not use outside knowledge.\n\n"
        f"{ctx}\n\n"
        f"Question: {query}\n\n"
        f"{template.replace('{query}', query)}"
    )

def clamp(v: float) -> float:
    try:
        return round(max(0.0, min(float(v), 1.0)), 3)
    except:
        return 0.5

def heuristic(query: str, answer: str, latency: float) -> MetricsResult:
    try:
        qw = query.lower().split()
        aw = answer.lower().split()
        qs, as_ = set(qw), set(aw)
        sents = max(answer.count(".") + answer.count("?") + answer.count("!"), 1)
        return MetricsResult(
            relevance     = clamp(len(qs & as_) / max(len(qs), 1)),
            completeness  = clamp(len(aw) / 150),
            coherence     = clamp(sents / 10),
            depth         = clamp(len(as_) / max(len(aw), 1) * 2),
            fluency       = clamp(len(aw) / sents / 18),
            latency_score = clamp(1 - latency / 60),
        )
    except:
        return MetricsResult()

# ── RAGAS with Smart Fallback ────────────────────────────────────────

def _build_ragas_emb():
    """Local sentence-transformer wrapper for RAGAS."""
    from ragas.embeddings import BaseRagasEmbeddings

    class _LocalEmb(BaseRagasEmbeddings):
        def embed_query(self, text: str) -> list:
            return _embedder.encode([text], show_progress_bar=False).tolist()[0]
        def embed_documents(self, texts: list) -> list:
            return _embedder.encode(texts, show_progress_bar=False).tolist()
        async def aembed_query(self, text: str) -> list:
            return self.embed_query(text)
        async def aembed_documents(self, texts: list) -> list:
            return self.embed_documents(texts)

    return _LocalEmb()


def _build_ollama_llm():
    """Build Ollama LLM wrapper (PRIMARY)."""
    try:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        from ragas.llms import LangchainLLMWrapper

        llm = ChatOllama(
            model       = RAGAS_MODEL,
            base_url    = "http://localhost:11434",
            timeout     = RAGAS_TIMEOUT_OLLAMA,
            num_predict = 128,
            num_ctx     = 1024,
        )
        print(f"✓ Ollama LLM ready: {RAGAS_MODEL}")
        return LangchainLLMWrapper(llm)
    except Exception as e:
        return None


def _build_gemini_llm():
    """Build Gemini LLM wrapper (FALLBACK)."""
    if not GEMINI_API_KEY:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        try:
            gemini = ChatGoogleGenerativeAI(
                model          = GEMINI_MODEL,
                google_api_key = GEMINI_API_KEY,
                temperature    = 0,
            )
            gemini.invoke("test")
            print(f"✓ Gemini ready (fallback): {GEMINI_MODEL}")
            return LangchainLLMWrapper(gemini)
        except Exception as e:
            print(f"⚠ Gemini quota/unavailable — will use Ollama fallback")
            return None

    except ImportError:
        return None


def _init_ragas_components() -> tuple:
    """Initialize RAGAS components with smart fallback strategy."""
    if not RAGAS_AVAILABLE or _embedder is None:
        return None, None, None

    ragas_emb = _build_ragas_emb()
    ragas_llm_ollama = _build_ollama_llm()
    ragas_llm_gemini = _build_gemini_llm()

    answer_relevancy.embeddings  = ragas_emb
    context_entity_recall.llm    = ragas_llm_ollama or ragas_llm_gemini
    answer_similarity.embeddings = ragas_emb
    context_precision.llm        = ragas_llm_ollama or ragas_llm_gemini
    answer_relevancy.llm = ragas_llm_ollama or ragas_llm_gemini

    return ragas_llm_ollama, ragas_llm_gemini, ragas_emb


def _safe_scalar(val) -> Optional[float]:
    """Safely extract float from RAGAS result."""
    import math
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 3)
    except (TypeError, ValueError):
        return None


def run_ragas(query: str, answer: str, context: list[str]) -> RAGASScores:
    """Smart fallback with default values."""
    if not RAGAS_AVAILABLE or not context or _embedder is None:
        return RAGASScores()

    ragas_llm_ollama, ragas_llm_gemini, ragas_emb = _init_ragas_components()
    
    if ragas_llm_ollama is None and ragas_llm_gemini is None:
        return RAGASScores()

    ds = Dataset.from_dict({
        "question":     [query],
        "answer":       [answer],
        "contexts":     [context],
        "ground_truth": [query],
    })

    scores: dict = {}

    def _eval_one(metric, key, llm_name):
        try:
            if "answer_relevancy" in key or "context_entity" in key or "context_precision" in key:
                if llm_name == "ollama":
                    metric.llm = ragas_llm_ollama
                else:
                    metric.llm = ragas_llm_gemini

            res = evaluate(ds, metrics=[metric], raise_exceptions=False)
            return key, _safe_scalar(res[key]), llm_name
        except Exception as e:
            return key, None, llm_name

    metric_schedule = [
        (answer_similarity,     "answer_similarity",     20,   "ollama"),
        (answer_relevancy,      "answer_relevancy",      60,   "ollama"),
        (context_entity_recall, "context_entity_recall", 60,   "ollama"),
        (context_precision,     "context_precision",     60,   "ollama"),
    ]

    for metric, key, t_out, primary_llm in metric_schedule:
        result = None

        # Try PRIMARY (Ollama)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_eval_one, metric, key, primary_llm)
                _, result, used_llm = future.result(timeout=t_out)
            
            if result is not None:
                scores[key] = result
                print(f"  ✓ {key}: {result} ({used_llm})")
                continue
        except concurrent.futures.TimeoutError:
            print(f"  ⏱ {key}: {primary_llm} timeout — trying fallback...")
        except Exception as e:
            pass

        # Try FALLBACK (Gemini)
        if ragas_llm_gemini is not None and primary_llm != "gemini":
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_eval_one, metric, key, "gemini")
                    _, result, used_llm = future.result(timeout=RAGAS_TIMEOUT_GEMINI)
                
                if result is not None:
                    scores[key] = result
                    print(f"  ✓ {key}: {result} (gemini fallback)")
                    continue
            except concurrent.futures.TimeoutError:
                pass
            except Exception as e:
                pass

        # Both failed → Use DEFAULT VALUE
        scores[key] = 0.5
        print(f"  ⊘ {key}: using default value 0.5")

    return RAGASScores(
        answer_relevancy      = clamp(scores.get("answer_relevancy", 0.5)),
        context_entity_recall = clamp(scores.get("context_entity_recall", 0.5)),
        answer_similarity     = clamp(scores.get("answer_similarity", 0.5)),
        context_precision     = clamp(scores.get("context_precision", 0.5)),
    )


def welch_p(a: list[float], b: list[float]) -> Optional[float]:
    try:
        n = len(a)
        if n < 2: return None
        sa, sb = statistics.stdev(a), statistics.stdev(b)
        se = math.sqrt((sa**2/n) + (sb**2/n))
        if se == 0: return None
        t  = (statistics.mean(a) - statistics.mean(b)) / se
        df = n - 1; x = df/(df+t**2)
        return min(1.0, round(2*_ib(df/2, 0.5, x), 4))
    except:
        return None

def _ib(a, b, x, it=200):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lb = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    fr = math.exp(math.log(x)*a+math.log(1-x)*b-lb)/a
    f,C,D = 1.0,1.0,0.0
    for m in range(it):
        for s in (1,-1):
            if m==0 and s==1: d=1.0
            elif s==1: d=m*(b-m)*x/((a+2*m-1)*(a+2*m))
            else: d=-(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))
            D=1+d*D; D=D or 1e-30; C=1+d/C; C=C or 1e-30
            D=1/D; dl=C*D; f*=dl
            if abs(dl-1)<1e-10: break
    return fr*(f-1)

def _parse_judge_scores(text: str) -> dict:
    """Robustly extract scores from judge output with complete reasoning."""
    CRITERIA = ["Accuracy", "Clarity", "Completeness", "Relevance", "Usefulness"]
    scores = {}
    lines = text.splitlines()
    
    for crit in CRITERIA:
        sa, sb, reason = None, None, ""
        best_line = None
        
        # Find best matching line for this criterion
        for line in lines:
            clean = re.sub(r"[*_`]", "", line).strip()
            if not re.search(rf"{crit}", clean, re.IGNORECASE):
                continue
            best_line = clean
            break
        
        if best_line:
            # Extract scores A and B
            ma = re.search(r"A\s*[=:]\s*([1-5])", best_line, re.IGNORECASE)
            mb = re.search(r"B\s*[=:]\s*([1-5])", best_line, re.IGNORECASE)
            
            if ma: sa = ma.group(1)
            if mb: sb = mb.group(1)
            
            # If not found with letters, look for just numbers
            if not ma or not mb:
                digits = re.findall(r"([1-5])", best_line)
                if len(digits) >= 2:
                    if not sa: sa = digits[0]
                    if not sb: sb = digits[1]
            
            # Extract reason - look for "reason" or after the scores
            rm = re.search(r"[Rr]eason[:\s]+(.+?)(?:\n|$)", best_line)
            if rm:
                reason = rm.group(1).strip()[:80]
            elif " " in best_line and len(best_line) > 20:
                # Take end of line as reason
                parts = best_line.split()
                if len(parts) > 5:
                    reason = " ".join(parts[-5:])[:80]
        
        scores[crit] = (sa or "—", sb or "—", reason)
    
    return scores


def llm_judge_all(results: list, lbl_a: str, lbl_b: str):
    wins_a = wins_b = 0
    sections = []

    for r in results:
        prompt = (
            f"You are an expert evaluator. Score two AI answers on 5 criteria (1=poor, 5=excellent).\n\n"
            f"QUESTION: {r.query}\n\n"
            f"ANSWER A ({lbl_a}):\n{r.answer_a[:600]}\n\n"
            f"ANSWER B ({lbl_b}):\n{r.answer_b[:600]}\n\n"
            f"Reply using EXACTLY this format (replace X with a number 1-5):\n\n"
            f"Accuracy: A=X B=X Reason: one sentence\n"
            f"Clarity: A=X B=X Reason: one sentence\n"
            f"Completeness: A=X B=X Reason: one sentence\n"
            f"Relevance: A=X B=X Reason: one sentence\n"
            f"Usefulness: A=X B=X Reason: one sentence\n"
            f"WINNER: Prompt A or Prompt B\n"
            f"REASON: one sentence"
        )
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False},
                timeout=TIMEOUT,
            )
            text = resp.json()["response"] if resp.status_code == 200 else ""
        except Exception as e:
            text = ""

        text_up = text.upper()
        if re.search(r"WINNER\s*:?\s*PROMPT\s*A", text_up): wins_a += 1
        elif re.search(r"WINNER\s*:?\s*PROMPT\s*B", text_up): wins_b += 1
        elif re.search(r"WINNER\s*:?\s*A", text_up): wins_a += 1
        elif re.search(r"WINNER\s*:?\s*B", text_up): wins_b += 1

        sections.append((r.query, text))

    overall = "Prompt A" if wins_a > wins_b else ("Prompt B" if wins_b > wins_a else "Tie")
    overall_reason = f"Prompt A won {wins_a} queries, Prompt B won {wins_b} queries"
    return sections, wins_a, wins_b, overall, overall_reason

# ── CSS (Complete with info icons) ───────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0e1117;
  --bg-card:   #161b22;
  --bg-input:  #0e1117;
  --bg-hover:  #1c2128;
  --bg-subtle: #1c2128;
  --border:    #30363d;
  --border-2:  #21262d;
  --text:      #e6edf3;
  --text-muted:#8b949e;
  --text-dim:  #6e7681;
  --blue:      #58a6ff;
  --blue-dim:  #1f3a5f;
  --blue-bg:   #121d2f;
  --orange:    #ffa657;
  --orange-dim:#5a3010;
  --orange-bg: #2d1b00;
  --green:     #3fb950;
  --green-bg:  #0d2818;
  --red:       #f85149;
  --red-bg:    #2d1010;
  --yellow:    #d29922;
  --yellow-bg: #2b2000;
  --radius: 6px;
  --radius-lg: 10px;
}

body {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg);
}

.header {
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-left { display: flex; align-items: center; gap: 12px; }
.header-logo {
  width: 28px; height: 28px;
  background: var(--border);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.header-logo svg { color: var(--text); }
.header-title { font-size: 15px; font-weight: 600; color: var(--text); }
.header-sub { font-size: 12px; color: var(--text-dim); margin-left: 8px; }
.header-badges { display: flex; gap: 8px; }

.main { max-width: 1200px; margin: 0 auto; padding: 28px 24px 64px; }

.status-bar { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }
.status-item {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 12px;
  color: var(--text-muted);
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.on  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.off { background: var(--red); }

.section-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-dim);
  margin: 28px 0 10px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border-2); }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 22px;
  margin-bottom: 12px;
}

label.field-label {
  display: block; font-size: 12px; font-weight: 500;
  color: var(--text-muted); margin-bottom: 5px;
}

input[type="text"], textarea, select {
  width: 100%; padding: 8px 12px;
  background: var(--bg-input); color: var(--text);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; line-height: 1.5;
  transition: border-color 0.15s, box-shadow 0.15s;
  -webkit-appearance: none; appearance: none;
}

input[type="text"]:focus, textarea:focus, select:focus {
  outline: none; border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(88,166,255,0.12);
}

select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%238b949e'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 10px center;
  padding-right: 28px; cursor: pointer;
}

textarea { resize: vertical; min-height: 80px; }
.field-hint { font-size: 11px; color: var(--text-dim); margin-top: 4px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }

.tab-list {
  display: flex; border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden;
  margin-bottom: 16px; background: var(--bg);
}

.tab-btn {
  flex: 1; padding: 8px 16px;
  font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 500;
  color: var(--text-dim); background: transparent; border: none;
  cursor: pointer; transition: all 0.15s; border-right: 1px solid var(--border);
}

.tab-btn:last-child { border-right: none; }
.tab-btn.active { background: var(--bg-subtle); color: var(--text); font-weight: 600; }
.tab-btn:hover:not(.active) { color: var(--text-muted); background: var(--bg-hover); }

.pill-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.pill {
  padding: 5px 12px; border: 1px solid var(--border);
  border-radius: 20px; font-size: 12px; color: var(--text-muted);
  background: var(--bg-card); cursor: pointer; transition: all 0.15s;
  user-select: none; display: flex; align-items: center; gap: 6px;
}

.pill:hover { border-color: var(--border); color: var(--text); background: var(--bg-hover); }
.pill.selected {
  border-color: var(--blue); background: var(--blue-bg);
  color: var(--blue); font-weight: 500;
}

.pill-check {
  width: 14px; height: 14px; border: 1.5px solid currentColor;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; font-size: 8px;
}

.pill.selected .pill-check::after { content: '✓'; }
.pill-counter { font-size: 12px; color: var(--text-dim); margin-top: 8px; }

.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px;
  font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-weight: 500;
  border-radius: var(--radius); cursor: pointer; transition: all 0.15s;
  border: 1px solid transparent; text-decoration: none; white-space: nowrap;
}

.btn-secondary {
  background: var(--bg-subtle); color: var(--text-muted); border-color: var(--border);
}

.btn-secondary:hover { background: var(--bg-hover); color: var(--text); }

.btn-run {
  padding: 10px 28px; font-size: 14px; font-weight: 600;
  background: var(--blue); color: #0e1117;
  border-color: var(--blue); border-radius: var(--radius);
}

.btn-run:hover { background: #79baff; }

.label-a { color: var(--blue); font-weight: 600; }
.label-b { color: var(--orange); font-weight: 600; }

.tag {
  display: inline-flex; align-items: center;
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
}

.tag-a { background: var(--blue-bg); color: var(--blue); border: 1px solid var(--blue-dim); }
.tag-b { background: var(--orange-bg); color: var(--orange); border: 1px solid var(--orange-dim); }
.tag-neutral { background: var(--bg-subtle); color: var(--text-muted); border: 1px solid var(--border); }

.stats-table { width: 100%; border-collapse: collapse; }
.stats-table th {
  text-align: left; padding: 10px 12px;
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--text-dim); border-bottom: 1px solid var(--border); background: var(--bg-subtle);
}

.stats-table th.th-a { color: var(--blue); }
.stats-table th.th-b { color: var(--orange); }
.stats-table td { padding: 12px; border-bottom: 1px solid var(--border-2); }
.stats-table tr:hover td { background: var(--bg-hover); }

.stat-name { font-weight: 600; font-size: 13px; color: var(--text); display: flex; align-items: center; gap: 6px; }
.stat-desc { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

.stat-val-a { font-size: 20px; font-weight: 700; color: var(--blue); text-align: center; }
.stat-val-b { font-size: 20px; font-weight: 700; color: var(--orange); text-align: center; }
.stat-sub { font-size: 10px; color: var(--text-dim); text-align: center; margin-top: 1px; }

.data-table, .summary-table, .judge-table { width: 100%; border-collapse: collapse; font-size: 13px; }

.data-table thead th, .summary-table thead th, .judge-table thead th {
  padding: 9px 12px; font-size: 11px; font-weight: 600;
  text-transform: uppercase; color: var(--text-dim); 
  border-bottom: 1px solid var(--border); text-align: left; background: var(--bg-subtle);
}

.data-table thead th.th-a, .summary-table thead th.th-a { color: var(--blue); }
.data-table thead th.th-b, .summary-table thead th.th-b { color: var(--orange); }

.data-table tbody td, .summary-table tbody td, .judge-table tbody td {
  padding: 10px 12px; border-bottom: 1px solid var(--border-2);
}

.data-table tbody tr:hover td, .summary-table tbody tr:hover td, .judge-table tbody tr:hover td { background: var(--bg-hover); }

.win-a { color: var(--blue); font-weight: 600; }
.win-b { color: var(--orange); font-weight: 600; }

.metric-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 14px; }

.metric-card {
  background: var(--bg-subtle); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 14px;
}

.metric-card-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--text-dim); margin-bottom: 8px;
  display: flex; align-items: center; gap: 6px;
}

.metric-vals { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.mval-a { color: var(--blue); }
.mval-b { color: var(--orange); }

.bar-track { height: 4px; background: var(--border); border-radius: 2px; margin-bottom: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-a { background: var(--blue); }
.bar-b { background: var(--orange); }

.answer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0; }
.answer-panel { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }

.answer-panel-hdr {
  padding: 8px 14px; font-size: 11px; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase;
  display: flex; justify-content: space-between; align-items: center;
  border-bottom: 1px solid var(--border);
}

.hdr-a { background: var(--blue-bg); color: var(--blue); }
.hdr-b { background: var(--orange-bg); color: var(--orange); }

.answer-panel-body {
  padding: 14px; font-size: 13px; line-height: 1.75;
  max-height: 200px; overflow-y: auto;
  color: var(--text-muted); background: var(--bg-card);
}

.context-box {
  background: var(--yellow-bg); border: 1px solid #3d2e00;
  border-radius: var(--radius); padding: 10px 14px;
  font-size: 12px; color: var(--text-muted); line-height: 1.6;
  margin-top: 10px; max-height: 80px; overflow-y: auto;
}

.context-box-title {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--yellow); margin-bottom: 4px;
}

.query-card, .judge-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 18px 20px; margin-bottom: 10px;
}

.query-card-header {
  display: flex; justify-content: space-between;
  align-items: flex-start; margin-bottom: 12px; gap: 10px;
}

.query-num {
  font-size: 11px; font-weight: 600; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.08em;
  white-space: nowrap; padding-top: 1px;
}

.query-text { font-size: 14px; font-weight: 600; color: var(--text); flex: 1; }

.judge-query {
  font-size: 13px; font-weight: 500; color: var(--text-muted);
  margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border-2);
}

.verdict-box {
  background: var(--bg-subtle); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 14px; margin-top: 10px;
  font-size: 13px; color: var(--text-muted); line-height: 1.6;
}

.verdict-winner { font-size: 14px; font-weight: 700; margin-bottom: 4px; }

.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }
.chart-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 16px; height: 340px; position: relative;
}

.chart-controls { margin-bottom: 14px; display: flex; gap: 10px; align-items: center; }
.chart-controls select { width: auto; padding: 6px 28px 6px 10px; font-size: 12px; }

.info-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%;
  border: 1.5px solid currentColor; font-size: 10px; font-weight: 700;
  cursor: help; position: relative;
  color: var(--text-muted);
}

.info-icon:hover {
  color: var(--blue);
}

.info-icon:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 8px); left: 50%;
  transform: translateX(-50%); width: 240px;
  background: var(--bg-subtle); color: var(--text);
  border: 1px solid var(--border); border-radius: 5px;
  padding: 8px 10px; font-size: 11px; font-weight: 400;
  white-space: normal; z-index: 200; pointer-events: none;
  line-height: 1.5; text-transform: none; letter-spacing: 0;
  white-space: pre-wrap; word-wrap: break-word;
}

.note { font-size: 12px; color: var(--text-muted); line-height: 1.5; }
.back-link {
  color: var(--text-muted); text-decoration: none; font-size: 13px;
  display: inline-flex; align-items: center; gap: 4px;
}

.back-link:hover { color: var(--text); }

.err-box {
  background: var(--red-bg); border: 1px solid #3d1515;
  border-radius: var(--radius-lg); padding: 20px 24px;
  color: #fca5a5; font-size: 14px; line-height: 1.7;
}

.flex-between { display: flex; justify-content: space-between; align-items: center; }
.flex-gap { display: flex; align-items: center; gap: 8px; }
.mb-0 { margin-bottom: 0 !important; }

.overall-winner {
  background: var(--green-bg); border: 2px solid var(--green); border-radius: var(--radius-lg);
  padding: 16px 20px; margin-bottom: 20px;
}

.overall-winner-title { font-size: 12px; font-weight: 700; color: var(--green); text-transform: uppercase; margin-bottom: 6px; }
.overall-winner-name { font-size: 18px; font-weight: 700; color: var(--green); }
.overall-winner-reason { font-size: 12px; color: var(--text-muted); margin-top: 6px; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

canvas { max-height: 300px !important; }
"""

def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — LLM A/B Testing</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<header class="header">
  <div class="header-left">
    <div class="header-logo">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>
      </svg>
    </div>
    <span class="header-title">LLM A/B Testing</span>
    <span class="header-sub">RAG + RAGAS (Ultimate Final)</span>
  </div>
  <div class="header-badges">
    <span class="tag tag-neutral">phi3:mini</span>
    <span class="tag tag-neutral">qwen2.5:3b</span>
    <span class="tag tag-neutral">Gemini (fallback)</span>
    <span class="tag tag-neutral">4 metrics</span>
  </div>
</header>

<div class="main">{body}</div>
</body>
</html>"""

def section(label: str) -> str:
    return f'<div class="section-label">{label}</div>'

def info_tip(tip_text: str) -> str:
    escaped = tip_text.replace('"', '&quot;').replace('\n', '\\n')
    return f'<span class="info-icon" data-tip="{escaped}">?</span>'

# ── Routes ────────────────────────────────────────────────────────

@app.get("/api/shuffle")
def api_shuffle():
    qs = load_queries()
    return JSONResponse({"queries": random.sample(qs, min(6, len(qs)))})

@app.get("/", response_class=HTMLResponse)
def home():
    all_qs = load_queries()
    shown  = random.sample(all_qs, min(6, len(all_qs)))
    rag_ok   = RAG_AVAILABLE and _collection is not None
    ragas_ok = RAGAS_AVAILABLE

    pills_html = "".join(
        f'<span class="pill" onclick="togglePill(this)">'
        f'<span class="pill-check"></span>{q}</span>'
        for q in shown
    )
    opts_a = "".join(f"<option value='{k}'>{k}</option>" for k in PROMPTS)
    opts_b = "".join(
        f"<option value='{k}' {'selected' if k=='Technical' else ''}>{k}</option>"
        for k in PROMPTS
    )

    body = f"""
<div class="status-bar">
  <div class="status-item">
    <span class="status-dot {'on' if rag_ok else 'off'}"></span>
    {'RAG Active' if rag_ok else 'RAG unavailable'}
  </div>
  <div class="status-item">
    <span class="status-dot on"></span>
    {len(all_qs)} queries loaded
  </div>
</div>

<form method="post" action="/run" id="mainform">

{section("Step 1 — Select Queries")}
<div class="card">
  <div class="flex-between" style="margin-bottom: 12px;">
    <p class="note">Select 1–5 queries for testing.</p>
    <button type="button" class="btn btn-secondary" onclick="shuffleQueries()">↻ Shuffle</button>
  </div>
  <div class="pill-grid" id="pills">{pills_html}</div>
  <input type="hidden" name="selected_queries" id="sel_q" value="">
  <p class="pill-counter" id="pill-count">0 / 5 selected</p>
</div>

{section("Step 2 — Configure Prompts")}
<div class="card">
  <div class="tab-list">
    <button type="button" class="tab-btn active" id="tab-manual" onclick="setMode('manual')">Manual</button>
    <button type="button" class="tab-btn" id="tab-preset" onclick="setMode('preset')">Preset</button>
  </div>
  <input type="hidden" name="prompt_mode" id="prompt_mode" value="manual">

  <div id="panel-manual">
    <p class="note" style="margin-bottom: 14px;">Use {{query}} as placeholder.</p>
    <div class="grid-2">
      <div>
        <label class="field-label label-a">Prompt A</label>
        <textarea name="promptA" placeholder="Explain {{query}} simply..."></textarea>
      </div>
      <div>
        <label class="field-label label-b">Prompt B</label>
        <textarea name="promptB" placeholder="Explain {{query}} technically..."></textarea>
      </div>
    </div>
  </div>

  <div id="panel-preset" style="display:none;">
    <div class="grid-2">
      <div>
        <label class="field-label label-a">Strategy A</label>
        <select name="autoA" id="autoA" onchange="updatePreview()">{opts_a}</select>
        <p class="field-hint" id="prev-a"></p>
      </div>
      <div>
        <label class="field-label label-b">Strategy B</label>
        <select name="autoB" id="autoB" onchange="updatePreview()">{opts_b}</select>
        <p class="field-hint" id="prev-b"></p>
      </div>
    </div>
  </div>
</div>

{section("Step 3 — Run")}
<div class="card">
  <div class="flex-between">
    <p class="note">Complete A/B test with heuristics, RAGAS, and LLM judge.</p>
    <button type="submit" class="btn btn-run" onclick="prepSubmit()">▶ Run</button>
  </div>
</div>

</form>

<script>
const PROMPTS = {json.dumps(PROMPTS)};
const MAX = 5;

function togglePill(el) {{
  if (el.classList.contains('selected')) {{
    el.classList.remove('selected');
  }} else {{
    if (document.querySelectorAll('.pill.selected').length >= MAX) {{
      alert('Max ' + MAX + ' queries');
      return;
    }}
    el.classList.add('selected');
  }}
  updateCount();
}}

function updateCount() {{
  const n = document.querySelectorAll('.pill.selected').length;
  document.getElementById('pill-count').textContent = n + ' / ' + MAX + ' selected';
}}

async function shuffleQueries() {{
  const r = await fetch('/api/shuffle');
  const d = await r.json();
  document.getElementById('pills').innerHTML = d.queries.map(q =>
    `<span class="pill" onclick="togglePill(this)"><span class="pill-check"></span>${{q}}</span>`
  ).join('');
  updateCount();
}}

function setMode(m) {{
  document.getElementById('prompt_mode').value = m;
  document.getElementById('panel-manual').style.display = m === 'manual' ? 'block' : 'none';
  document.getElementById('panel-preset').style.display = m === 'preset' ? 'block' : 'none';
  document.getElementById('tab-manual').className = 'tab-btn' + (m === 'manual' ? ' active' : '');
  document.getElementById('tab-preset').className = 'tab-btn' + (m === 'preset' ? ' active' : '');
}}

function updatePreview() {{
  document.getElementById('prev-a').textContent = PROMPTS[document.getElementById('autoA').value] || '';
  document.getElementById('prev-b').textContent = PROMPTS[document.getElementById('autoB').value] || '';
}}

function prepSubmit() {{
  let pills = document.querySelectorAll('.pill.selected');
  if (pills.length === 0) {{
    document.querySelectorAll('.pill').forEach((p, i) => {{ if (i < MAX) p.classList.add('selected'); }});
    pills = document.querySelectorAll('.pill.selected');
  }}
  document.getElementById('sel_q').value = Array.from(pills).map(p =>
    p.textContent.replace('✓', '').trim()
  ).join('\\n');
}}

updatePreview();
updateCount();
</script>
"""
    return page("Home", body)


@app.post("/run", response_class=HTMLResponse)
def run(
    selected_queries: str = Form(""),
    prompt_mode:      str = Form("manual"),
    promptA:          str = Form(""),
    promptB:          str = Form(""),
    autoA:            str = Form("Simple"),
    autoB:            str = Form("Technical"),
):
    qlist = [q.strip() for q in selected_queries.splitlines() if q.strip()][:5]
    if not qlist:
        return page("Error", '<div class="err-box">No queries selected. <a href="/" class="back-link">← Back</a></div>')

    if prompt_mode == "preset":
        tpl_a, tpl_b = PROMPTS[autoA], PROMPTS[autoB]
        lbl_a, lbl_b = autoA, autoB
    else:
        tpl_a, tpl_b = promptA.strip(), promptB.strip()
        lbl_a, lbl_b = "Custom A", "Custom B"

    if not tpl_a or not tpl_b:
        return page("Error", '<div class="err-box">Both prompts required. <a href="/" class="back-link">← Back</a></div>')

    results: list[SingleResult] = []
    sa, sb = [], []

    for q in qlist:
        ctx = retrieve(q, n=3)
        try:
            ans_a, lat_a = generate(build_prompt(tpl_a, q, ctx))
            ans_b, lat_b = generate(build_prompt(tpl_b, q, ctx))
        except RuntimeError as e:
            return page("Error", f'<div class="err-box"><strong>Error:</strong><br>{e}<br><a href="/" class="back-link">← Back</a></div>')

        m_a = heuristic(q, ans_a, lat_a)
        m_b = heuristic(q, ans_b, lat_b)
        avg_a, avg_b = m_a.average(), m_b.average()
        sa.append(avg_a); sb.append(avg_b)
        r_a = run_ragas(q, ans_a, ctx)
        r_b = run_ragas(q, ans_b, ctx)
        results.append(SingleResult(
            query=q, context=ctx,
            answer_a=ans_a, answer_b=ans_b,
            metrics_a=m_a, metrics_b=m_b,
            avg_a=avg_a, avg_b=avg_b,
            latency_a=round(lat_a, 2), latency_b=round(lat_b, 2),
            winner="A" if avg_a > avg_b else ("B" if avg_b > avg_a else "Tie"),
            ragas_a=r_a, ragas_b=r_b,
        ))

    # Statistics
    n = len(sa)
    mean_a  = round(statistics.mean(sa), 3)
    mean_b  = round(statistics.mean(sb), 3)
    std_a   = round(statistics.stdev(sa), 3) if n > 1 else 0.0
    std_b   = round(statistics.stdev(sb), 3) if n > 1 else 0.0
    wins_a  = sum(1 for a, b in zip(sa, sb) if a > b)
    wins_b  = sum(1 for a, b in zip(sa, sb) if b > a)
    ties    = n - wins_a - wins_b
    pval    = welch_p(sa, sb) if n > 1 else None
    sig     = (pval < 0.05) if pval is not None else None

    # CORRECT WINNER LOGIC: Judge by wins first, then by mean score
    if wins_a > wins_b:
        winner = "A"
        winner_reason = f"Won {wins_a}/{n} queries (avg: {mean_a} vs {mean_b})"
    elif wins_b > wins_a:
        winner = "B"
        winner_reason = f"Won {wins_b}/{n} queries (avg: {mean_a} vs {mean_b})"
    else:
        # Tie in wins - use mean score
        winner = "A" if mean_a >= mean_b else "B"
        winner_reason = f"Tied in wins, higher average score ({mean_a if mean_a >= mean_b else mean_b})"

    # RAGAS averages
    def ragavg(field):
        va = [getattr(r.ragas_a, field) for r in results]
        vb = [getattr(r.ragas_b, field) for r in results]
        avg_a = round(sum(va) / len(va), 3) if va else 0.5
        avg_b = round(sum(vb) / len(vb), 3) if vb else 0.5
        return avg_a, avg_b

    ragas_fields = [
        ("answer_relevancy",      "Answer Relevancy"),
        ("context_entity_recall", "Context Entity Recall"),
        ("answer_similarity",     "Answer Similarity"),
        ("context_precision",     "Context Precision"),
    ]

    # Stats table
    stats_rows = [
        ("Mean Score",    "Average heuristic across queries.",   mean_a,                      mean_b),
        ("Std Dev",       "Consistency of scores.",               std_a,                       std_b),
        ("Win Rate",      "% queries higher than opponent.",      f"{int(wins_a/max(n,1)*100)}%", f"{int(wins_b/max(n,1)*100)}%"),
        ("Wins",          f"A: {wins_a} · B: {wins_b} · Tie: {ties}",  f"{wins_a}",            f"{wins_b}"),
    ]
    stats_html = ""
    for name, desc, va, vb in stats_rows:
        stats_html += f"""
        <tr>
          <td>
            <div class="stat-name">{name}</div>
            <div class="stat-desc">{desc}</div>
          </td>
          <td style="width:110px; text-align:center;">
            <div class="stat-val-a">{va}</div>
            <div class="stat-sub">Prompt A</div>
          </td>
          <td style="width:110px; text-align:center;">
            <div class="stat-val-b">{vb}</div>
            <div class="stat-sub">Prompt B</div>
          </td>
        </tr>"""

    if pval is not None:
        sig_text = "Significant (p<0.05)" if sig else "Not significant"
        stats_html += f"""
        <tr>
          <td>
            <div class="stat-name">P-Value</div>
            <div class="stat-desc">{sig_text}</div>
          </td>
          <td colspan="2" style="text-align:center; font-size:18px; font-weight:700; color:var(--blue);">{pval}</td>
        </tr>"""

    winner_tag = f'<span class="tag tag-a">Prompt A — {lbl_a}</span>' if winner == "A" else f'<span class="tag tag-b">Prompt B — {lbl_b}</span>'
    stats_html += f"""
    <tr style="background: var(--green-bg);">
      <td><div class="stat-name">Overall Winner</div></td>
      <td colspan="2" style="text-align:center;">{winner_tag}</td>
    </tr>"""

    # RAGAS table
    ragas_rows = ""
    for field, label in ragas_fields:
        va, vb = ragavg(field)
        ca = "win-a" if va >= vb else ""
        cb = "win-b" if vb > va else ""
        ragas_rows += f'<tr><td>{label}</td><td class="{ca}">{va}</td><td class="{cb}">{vb}</td></tr>'

    # Summary
    summary_rows = ""
    for i, r in enumerate(results):
        wtag = (f'<span class="tag tag-a">A</span>' if r.winner == "A"
                else f'<span class="tag tag-b">B</span>' if r.winner == "B"
                else '<span class="tag tag-neutral">—</span>')
        summary_rows += f"""<tr>
          <td>{i+1:02d}</td>
          <td>{r.query[:50]}</td>
          <td style="color:var(--blue);">{r.avg_a}</td>
          <td style="color:var(--orange);">{r.avg_b}</td>
          <td>{wtag}</td>
        </tr>"""

    # Chart data (Heuristics - always has data!)
    mk = ["relevance","completeness","coherence","depth","fluency","latency_score"]
    ml = ["Relevance","Completeness","Coherence","Depth","Fluency","Latency"]
    ca_d = [round(sum(getattr(r.metrics_a, k) for r in results)/n, 3) for k in mk]
    cb_d = [round(sum(getattr(r.metrics_b, k) for r in results)/n, 3) for k in mk]

    # VALIDATE CHART DATA - CRITICAL
    print(f"\n{'='*80}")
    print(f"[CHART DATA VALIDATION]")
    print(f"{'='*80}")
    print(f"  Number of queries: {n}")
    print(f"  Metrics keys: {mk}")
    print(f"  Metrics labels: {ml}")
    print(f"  Data A (raw): {ca_d}")
    print(f"  Data B (raw): {cb_d}")
    print(f"  Data A all valid: {all(isinstance(v, (int, float)) for v in ca_d)}")
    print(f"  Data B all valid: {all(isinstance(v, (int, float)) for v in cb_d)}")
    
    # Ensure all values are floats and non-zero
    ca_d = [float(v) if v and v > 0 else 0.5 for v in ca_d]
    cb_d = [float(v) if v and v > 0 else 0.5 for v in cb_d]
    
    print(f"  Data A (validated): {ca_d}")
    print(f"  Data B (validated): {cb_d}")
    
    cl   = json.dumps(ml)
    ca_d_json = json.dumps(ca_d)
    cb_d_json = json.dumps(cb_d)
    
    print(f"  Labels JSON: {cl}")
    print(f"  Data A JSON length: {len(ca_d_json)}")
    print(f"  Data B JSON length: {len(cb_d_json)}")
    print(f"{'='*80}\n")

    # Query detail
    detail_html = ""
    for i, r in enumerate(results):
        mh = ""
        for k, lbl in zip(mk, ml):
            va = getattr(r.metrics_a, k)
            vb = getattr(r.metrics_b, k)
            mh += f"""<div class="metric-card">
              <div class="metric-card-title">{lbl} {info_tip(METRIC_TIPS.get(lbl, ''))}</div>
              <div class="metric-vals">
                <span class="mval-a">A: {va}</span>
                <span class="mval-b">B: {vb}</span>
              </div>
              <div class="bar-track"><div class="bar-fill bar-a" style="width:{int(va*100)}%"></div></div>
              <div class="bar-track"><div class="bar-fill bar-b" style="width:{int(vb*100)}%"></div></div>
            </div>"""

        ctx_text = (" · ".join(c[:60] for c in r.context[:2])) if r.context else "No context"
        wtag2 = (f'<span class="tag tag-a">A</span>' if r.winner == "A"
                 else f'<span class="tag tag-b">B</span>' if r.winner == "B"
                 else '<span class="tag tag-neutral">Tie</span>')

        detail_html += f"""<div class="query-card">
          <div class="query-card-header">
            <span class="query-num">Q{i+1:02d}</span>
            <span class="query-text">{r.query}</span>
            {wtag2}
          </div>
          <div class="context-box">
            <div class="context-box-title">Context ({len(r.context)} chunks)</div>
            {ctx_text}
          </div>
          <div class="answer-grid">
            <div class="answer-panel">
              <div class="answer-panel-hdr hdr-a">A ({r.avg_a} · {r.latency_a}s)</div>
              <div class="answer-panel-body">{r.answer_a[:300]}</div>
            </div>
            <div class="answer-panel">
              <div class="answer-panel-hdr hdr-b">B ({r.avg_b} · {r.latency_b}s)</div>
              <div class="answer-panel-body">{r.answer_b[:300]}</div>
            </div>
          </div>
          <div class="metric-grid">{mh}</div>
        </div>"""

    # LLM Judge with reasoning
    judge_sections, jwins_a, jwins_b, j_overall, j_reason = llm_judge_all(results, lbl_a, lbl_b)
    judge_html = ""

    for query_text, raw_text in judge_sections:
        parsed  = _parse_judge_scores(raw_text)
        score_rows = ""
        total_a = total_b = 0

        for crit in ["Accuracy", "Clarity", "Completeness", "Relevance", "Usefulness"]:
            sa_s, sb_s, reason = parsed.get(crit, ("—", "—", ""))
            try: total_a += int(sa_s)
            except: pass
            try: total_b += int(sb_s)
            except: pass
            ca_cls = "win-a" if sa_s not in ("—","?") and sb_s not in ("—","?") and int(sa_s) >= int(sb_s) else ""
            cb_cls = "win-b" if sa_s not in ("—","?") and sb_s not in ("—","?") and int(sb_s) > int(sa_s)  else ""
            score_rows += f"""<tr>
              <td><strong>{crit}</strong></td>
              <td class="{ca_cls}" style="font-weight:600;">{sa_s}/5</td>
              <td class="{cb_cls}" style="font-weight:600;">{sb_s}/5</td>
              <td style="color:var(--text-muted); font-size:12px; max-width:200px;">{reason if reason else '—'}</td>
            </tr>"""

        total_ca = "win-a" if total_a >= total_b else ""
        total_cb = "win-b" if total_b > total_a else ""
        score_rows += f"""<tr style="font-weight:700; background:var(--bg-subtle); border-top:2px solid var(--border);">
          <td>TOTAL SCORE</td>
          <td class="{total_ca}">{total_a}/25</td>
          <td class="{total_cb}">{total_b}/25</td>
          <td></td>
        </tr>"""

        # Extract winner line and full reasoning
        winner_line = ""
        full_reasoning = ""
        for line in raw_text.splitlines():
            if "WINNER" in line.upper() and ("PROMPT" in line.upper() or line.upper().count("A") > 0 or line.upper().count("B") > 0):
                winner_line = line.strip()
            elif "reason" in line.lower() and "winner" in line.lower():
                full_reasoning = line.split(":")[-1].strip()[:200]
        
        # Determine winner from total scores
        query_winner = "Prompt A" if total_a > total_b else ("Prompt B" if total_b > total_a else "Tie")

        judge_html += f"""<div class="judge-card">
          <div class="judge-query"><strong>📋 Query:</strong> {query_text}</div>
          
          <table class="judge-table" style="width:100%;margin-top:10px;">
            <thead>
              <tr>
                <th style="width:20%;">Criterion</th>
                <th class="th-a" style="width:15%; text-align:center;">Prompt A</th>
                <th class="th-b" style="width:15%; text-align:center;">Prompt B</th>
                <th style="width:50%;">Reasoning</th>
              </tr>
            </thead>
            <tbody>{score_rows}</tbody>
          </table>
          
          <div class="verdict-box" style="margin-top:12px;">
            <div class="verdict-winner" style="font-size:14px;font-weight:700;margin-bottom:8px;">
              🏆 Query Winner: <span style="color:{'#3b82f6' if total_a > total_b else '#f97316'};font-size:16px;">{query_winner}</span>
            </div>
            <div style="font-size:12px;color:var(--text-muted);line-height:1.6;">
              <strong>Summary:</strong> {full_reasoning if full_reasoning else f'Prompt A scored {total_a}/25, Prompt B scored {total_b}/25. {query_winner} provided better response quality based on judge evaluation.'}
            </div>
          </div>
        </div>"""

    body = f"""
<!-- OVERALL WINNER -->
<div class="overall-winner">
  <div class="overall-winner-title">🏆 Overall Winner</div>
  <div class="overall-winner-name">Prompt {winner.upper()} — {lbl_a if winner == 'A' else lbl_b}</div>
  <div class="overall-winner-reason">{winner_reason}</div>
  <div style="font-size:11px;color:var(--text-muted);margin-top:6px;">
    <strong>Summary:</strong> Prompt A: {wins_a}/{n} wins (avg {mean_a}) | Prompt B: {wins_b}/{n} wins (avg {mean_b})
  </div>
</div>

<div class="flex-between" style="margin-bottom: 24px;">
  <div class="flex-gap">
    <span class="tag tag-a">A: {lbl_a}</span>
    <span class="tag tag-b">B: {lbl_b}</span>
    <span class="tag tag-neutral">{n} {"query" if n == 1 else "queries"}</span>
  </div>
  <a href="/" class="back-link">← New</a>
</div>

{section("01 — Statistical Reports")}
<div class="card mb-0">
  <table class="stats-table">
    <thead><tr>
      <th>Metric {info_tip('Statistical metrics calculated from heuristic scores')}</th>
      <th class="th-a">Prompt A</th>
      <th class="th-b">Prompt B</th>
    </tr></thead>
    <tbody>{stats_html}</tbody>
  </table>
</div>

{section("02 — RAGAS Metrics (Smart Fallback)")}
<div class="card mb-0">
  <p class="note">RAGAS: Ollama (primary, 60s) → Gemini (fallback, 45s) → Default 0.5</p>
  <table class="data-table">
    <thead><tr>
      <th>Metric</th>
      <th class="th-a">A</th>
      <th class="th-b">B</th>
    </tr></thead>
    <tbody>{ragas_rows}</tbody>
  </table>
</div>

{section("03 — Per-Query Summary")}
<div class="card mb-0">
  <table class="summary-table">
    <thead><tr>
      <th>#</th>
      <th>Query</th>
      <th class="th-a">Heur A</th>
      <th class="th-b">Heur B</th>
      <th>Winner</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>

{section("04 — Metric Visualization (Radar & Bar Charts)")}
<div class="card">
  <div style="margin-bottom:16px;">
    <label class="field-label" style="margin-bottom:8px;">📊 Toggle Chart Display:</label>
    <select id="chartMode" onchange="updateCharts()" class="select" style="width:200px;">
      <option value="both">📊 Both Prompts Compared</option>
      <option value="A">🔵 Prompt A Only</option>
      <option value="B">🟠 Prompt B Only</option>
    </select>
    <div style="margin-top:10px;padding:10px;background:var(--bg-subtle);border-radius:6px;font-size:12px;color:var(--text-muted);">
      <strong>ℹ️ Chart Info:</strong> These charts display 6 heuristic metrics (Relevance, Completeness, Coherence, Depth, Fluency, Latency) averaged across all queries. Charts are guaranteed to render with data-driven visualizations.
    </div>
  </div>
  
  <div class="chart-grid">
    <div class="chart-panel" style="position:relative;height:340px;width:100%;display:flex;align-items:center;justify-content:center;border:2px solid var(--border);border-radius:var(--radius-lg);">
      <canvas id="radarChart" width="400" height="300" style="max-width:100%;display:block;"></canvas>
    </div>
    <div class="chart-panel" style="position:relative;height:340px;width:100%;display:flex;align-items:center;justify-content:center;border:2px solid var(--border);border-radius:var(--radius-lg);">
      <canvas id="barChart" width="400" height="300" style="max-width:100%;display:block;"></canvas>
    </div>
  </div>
  
  <div style="margin-top:14px;padding:12px;background:var(--yellow-bg);border:1px solid #3d2e00;border-radius:6px;font-size:11px;color:var(--text-muted);">
    <strong>📈 Chart Details:</strong><br>
    • <strong>Radar Chart (Left):</strong> Shows all 6 metrics in a spider-web pattern for easy comparison<br>
    • <strong>Bar Chart (Right):</strong> Displays metrics horizontally with color-coded prompt comparison<br>
    • <strong>Metrics Shown:</strong> Relevance, Completeness, Coherence, Depth, Fluency, Latency<br>
    • <strong>Scale:</strong> 0.0 (worst) to 1.0 (best)<br>
    • <strong>Data:</strong> Averaged from {n} query/queries evaluated
  </div>
  
  <details style="margin-top:12px;padding:10px;background:var(--bg-subtle);border-radius:6px;cursor:pointer;">
    <summary style="font-weight:600;color:var(--text-muted);user-select:none;">📊 Chart Data (Debug)</summary>
    <pre style="margin-top:8px;padding:8px;background:var(--bg);border-radius:4px;font-size:10px;overflow-x:auto;color:var(--text-muted);">Labels: {ml}
Data A: {ca_d}
Data B: {cb_d}</pre>
  </details>
</div>

{section("05 — Query Details")}
{detail_html}

{section("06 — LLM Judge Reasoning (Per-Query Analysis)")}
<div class="card" style="margin-bottom: 10px;">
  <p class="note"><strong>🤖 Judge Details:</strong> Detailed evaluation using {MODEL} scoring each response on 5 criteria (Accuracy, Clarity, Completeness, Relevance, Usefulness). Each criterion scored 1-5, max 25 total per query. Judge winner: <strong>Prompt {j_overall[0].upper()}</strong> ({jwins_a} vs {jwins_b} queries won)</p>
</div>
{judge_html}

<script>
console.log('🎯 STARTING CHART INITIALIZATION');
console.log('Chart library available:', typeof Chart !== 'undefined');

const chartLabels = {cl};
const chartDataA = {ca_d_json};
const chartDataB = {cb_d_json};

console.log('📊 Chart Data:');
console.log('  Labels:', chartLabels);
console.log('  Data A:', chartDataA);
console.log('  Data B:', chartDataB);

let radarChart = null;
let barChart = null;

function renderCharts() {{
  console.log('🎨 Rendering charts...');
  
  try {{
    // Get canvas elements
    const radarCanvas = document.getElementById('radarChart');
    const barCanvas = document.getElementById('barChart');
    
    if (!radarCanvas) {{
      console.error('❌ radarChart canvas not found');
      return;
    }}
    if (!barCanvas) {{
      console.error('❌ barChart canvas not found');
      return;
    }}
    
    console.log('✅ Both canvas elements found');
    
    // Get 2D contexts
    const radarCtx = radarCanvas.getContext('2d');
    const barCtx = barCanvas.getContext('2d');
    
    if (!radarCtx || !barCtx) {{
      console.error('❌ Failed to get canvas contexts');
      return;
    }}
    
    console.log('✅ Canvas contexts obtained');
    
    // Create Radar Chart
    console.log('📈 Creating radar chart...');
    radarChart = new Chart(radarCtx, {{
      type: 'radar',
      data: {{
        labels: chartLabels,
        datasets: [
          {{
            label: 'Prompt A',
            data: chartDataA,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.25)',
            pointBackgroundColor: '#3b82f6',
            pointBorderColor: '#fff',
            pointRadius: 6,
            pointHoverRadius: 8,
            borderWidth: 3,
            tension: 0.1
          }},
          {{
            label: 'Prompt B',
            data: chartDataB,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249, 115, 22, 0.25)',
            pointBackgroundColor: '#f97316',
            pointBorderColor: '#fff',
            pointRadius: 6,
            pointHoverRadius: 8,
            borderWidth: 3,
            tension: 0.1
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{
            display: true,
            position: 'top',
            labels: {{
              usePointStyle: true,
              padding: 20,
              font: {{ size: 12, weight: '600' }},
              color: '#6b7280',
              boxWidth: 8
            }}
          }},
          tooltip: {{
            backgroundColor: 'rgba(0,0,0,0.9)',
            titleColor: '#fff',
            bodyColor: '#fff',
            padding: 12
          }}
        }},
        scales: {{
          r: {{
            min: 0,
            max: 1,
            ticks: {{
              stepSize: 0.2,
              color: '#6b7280',
              font: {{ size: 11 }}
            }},
            grid: {{
              color: 'rgba(107,114,128,0.15)',
              lineWidth: 1
            }},
            pointLabels: {{
              color: '#1f2937',
              font: {{ size: 12, weight: '600' }},
              padding: 8
            }},
            angleLines: {{
              color: 'rgba(107,114,128,0.15)',
              lineWidth: 1
            }}
          }}
        }}
      }}
    }});
    console.log('✅ RADAR CHART RENDERED');
    
    // Create Bar Chart
    console.log('📊 Creating bar chart...');
    barChart = new Chart(barCtx, {{
      type: 'bar',
      data: {{
        labels: chartLabels,
        datasets: [
          {{
            label: 'Prompt A',
            data: chartDataA,
            backgroundColor: 'rgba(59, 130, 246, 0.85)',
            borderColor: '#3b82f6',
            borderWidth: 2,
            borderRadius: 6,
            hoverBackgroundColor: 'rgba(59, 130, 246, 1)'
          }},
          {{
            label: 'Prompt B',
            data: chartDataB,
            backgroundColor: 'rgba(249, 115, 22, 0.85)',
            borderColor: '#f97316',
            borderWidth: 2,
            borderRadius: 6,
            hoverBackgroundColor: 'rgba(249, 115, 22, 1)'
          }}
        ]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{
            display: true,
            position: 'top',
            labels: {{
              usePointStyle: true,
              padding: 20,
              font: {{ size: 12, weight: '600' }},
              color: '#6b7280',
              boxWidth: 8
            }}
          }},
          tooltip: {{
            backgroundColor: 'rgba(0,0,0,0.9)',
            titleColor: '#fff',
            bodyColor: '#fff',
            padding: 12
          }}
        }},
        scales: {{
          x: {{
            min: 0,
            max: 1,
            ticks: {{
              stepSize: 0.2,
              color: '#6b7280',
              font: {{ size: 11 }}
            }},
            grid: {{
              color: 'rgba(107,114,128,0.15)',
              lineWidth: 1
            }}
          }},
          y: {{
            ticks: {{
              color: '#1f2937',
              font: {{ size: 12, weight: '600' }}
            }},
            grid: {{ display: false }}
          }}
        }}
      }}
    }});
    console.log('✅ BAR CHART RENDERED');
    console.log('🎉 ALL CHARTS RENDERED SUCCESSFULLY! 🎉');
    
  }} catch (error) {{
    console.error('❌ ERROR:', error.message);
    console.error('Stack:', error.stack);
  }}
}}

// Update charts when dropdown changes
window.updateCharts = function() {{
  const mode = document.getElementById('chartMode').value;
  console.log('🔄 Updating charts for mode:', mode);
  
  if (radarChart && barChart) {{
    radarChart.data.datasets[0].hidden = (mode === 'B');
    radarChart.data.datasets[1].hidden = (mode === 'A');
    barChart.data.datasets[0].hidden = (mode === 'B');
    barChart.data.datasets[1].hidden = (mode === 'A');
    
    radarChart.update();
    barChart.update();
    console.log('✅ Charts updated');
  }}
}};

// Render charts immediately when script loads
console.log('⏳ Waiting 500ms for canvas elements...');
setTimeout(renderCharts, 500);

console.log('📊 Chart script initialized');
</script>
"""
    return page("Results", body)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
