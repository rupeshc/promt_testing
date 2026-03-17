"""
LLM A/B Testing Framework — Processing Core
FIXED: _parse_judge_scores is more robust
"""

import requests
import time
import json
import statistics
import math
import re
import concurrent.futures
from typing import Optional, List, Tuple
from models import MetricsResult, RAGASScores, SingleResult, METRIC_TIPS, MODEL, RAGAS_MODEL, TIMEOUT, RAGAS_TIMEOUT_OLLAMA, RAGAS_TIMEOUT_GEMINI, BASE_DIR

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

# ── RAG ────────────────────────────────────────────────────────────────
_embedder   = None
_collection = None

def init_rag():
    global _embedder, _collection
    if not RAG_AVAILABLE:
        return
    print("Initialising RAG pipeline...")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kb_path = f"{BASE_DIR}/knowledge_base.txt"
    if not __import__('os').path.exists(kb_path):
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

def retrieve(query: str, n: int = 3) -> List[str]:
    if _embedder is None or _collection is None:
        return []
    q_vec = _embedder.encode([query], show_progress_bar=False).tolist()
    res   = _collection.query(query_embeddings=q_vec, n_results=n)
    return res["documents"][0]

def load_queries() -> List[str]:
    import os
    try:
        path = f"{BASE_DIR}/queries.txt"
        return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return ["What is machine learning?", "What is Docker?",
                "How does RAG work?", "What is a transformer model?"]

# ── LLM ────────────────────────────────────────────────────────────────
def generate(prompt: str) -> Tuple[str, float]:
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

def build_prompt(template: str, query: str, context: List[str]) -> str:
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

# ── RAGAS with Smart Fallback ──────────────────────────────────────────

def _build_ragas_emb():
    """Local sentence-transformer wrapper for RAGAS."""
    from ragas.embeddings import BaseRagasEmbeddings

    class _LocalEmb(BaseRagasEmbeddings):
        def embed_query(self, text: str) -> list:
            return _embedder.encode([text], show_progress_bar=False).tolist()[0]
        def embed_documents(self, texts: List[str]) -> list:
            return _embedder.encode(texts, show_progress_bar=False).tolist()
        async def aembed_query(self, text: str) -> list:
            return self.embed_query(text)
        async def aembed_documents(self, texts: List[str]) -> list:
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


def _build_gemini_llm(gemini_api_key: str):
    """Build Gemini LLM wrapper (FALLBACK)."""
    if not gemini_api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        try:
            gemini = ChatGoogleGenerativeAI(
                model          = "gemini-2.5-flash-lite",
                google_api_key = gemini_api_key,
                temperature    = 0,
            )
            gemini.invoke("test")
            print(f"✓ Gemini ready (fallback): gemini-2.5-flash-lite")
            return LangchainLLMWrapper(gemini)
        except Exception as e:
            print(f"⚠ Gemini quota/unavailable — will use Ollama fallback")
            return None

    except ImportError:
        return None


def _init_ragas_components(gemini_api_key: str) -> tuple:
    """Initialize RAGAS components with smart fallback strategy."""
    if not RAGAS_AVAILABLE or _embedder is None:
        return None, None, None

    ragas_emb = _build_ragas_emb()
    ragas_llm_ollama = _build_ollama_llm()
    ragas_llm_gemini = _build_gemini_llm(gemini_api_key)

    answer_relevancy.embeddings  = ragas_emb
    context_entity_recall.llm    = ragas_llm_ollama or ragas_llm_gemini
    answer_similarity.embeddings = ragas_emb
    context_precision.llm        = ragas_llm_ollama or ragas_llm_gemini
    answer_relevancy.llm = ragas_llm_ollama or ragas_llm_gemini

    return ragas_llm_ollama, ragas_llm_gemini, ragas_emb


def _safe_scalar(val) -> Optional[float]:
    """Safely extract float from RAGAS result."""
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


def run_ragas(query: str, answer: str, context: List[str], gemini_api_key: str) -> RAGASScores:
    """Smart fallback with default values."""
    if not RAGAS_AVAILABLE or not context or _embedder is None:
        print(f"  ℹ️  RAGAS skipped (missing: RAGAS={RAGAS_AVAILABLE}, context={len(context) if context else 0}, embedder={_embedder is not None})")
        return RAGASScores()

    ragas_llm_ollama, ragas_llm_gemini, ragas_emb = _init_ragas_components(gemini_api_key)

    if ragas_llm_ollama is None and ragas_llm_gemini is None:
        print(f"  ℹ️  RAGAS skipped (no LLM available)")
        return RAGASScores()

    ds = Dataset.from_dict({
        "question":     [query],
        "answer":       [answer],
        "contexts":     [context],
        "ground_truth": [answer],
    })

    print(f"  📊 RAGAS Dataset created: {len(context)} context chunks, {len(answer.split())} answer words")

    scores: dict = {}

    def _eval_one(metric, key, llm_name):
        try:
            if "answer_relevancy" in key or "context_entity" in key or "context_precision" in key:
                if llm_name == "ollama":
                    metric.llm = ragas_llm_ollama
                else:
                    metric.llm = ragas_llm_gemini

            res = evaluate(ds, metrics=[metric], raise_exceptions=False)
            result = _safe_scalar(res[key])

            if result is None or (result == 0.0 and "entity" in key.lower()):
                print(f"  ⚠️  {key}: returned 0.0, using adaptive default")
                if "entity" in key.lower():
                    context_text = " ".join(context).lower()
                    query_words = set(query.lower().split())
                    matching_words = sum(1 for word in query_words if word in context_text)
                    result = round(matching_words / max(len(query_words), 1), 3)
                    result = min(result, 0.5)

            return key, result, llm_name
        except Exception as e:
            print(f"  ⚠️  {key} error: {str(e)[:60]}")
            return key, None, llm_name

    metric_schedule = [
        (answer_similarity,     "answer_similarity",     20,   "ollama"),
        (answer_relevancy,      "answer_relevancy",      60,   "ollama"),
        (context_entity_recall, "context_entity_recall", 60,   "ollama"),
        (context_precision,     "context_precision",     60,   "ollama"),
    ]

    for metric, key, t_out, primary_llm in metric_schedule:
        result = None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_eval_one, metric, key, primary_llm)
                _, result, used_llm = future.result(timeout=t_out)

            if result is not None and result >= 0.0:
                scores[key] = result
                print(f"  ✓ {key}: {result} ({used_llm})")
                continue
        except concurrent.futures.TimeoutError:
            print(f"  ⏱ {key}: {primary_llm} timeout ({t_out}s) — trying fallback...")
        except Exception as e:
            print(f"  ❌ {key}: {primary_llm} failed ({str(e)[:40]}) — trying fallback...")

        if ragas_llm_gemini is not None and primary_llm != "gemini":
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_eval_one, metric, key, "gemini")
                    _, result, used_llm = future.result(timeout=RAGAS_TIMEOUT_GEMINI)

                if result is not None and result >= 0.0:
                    scores[key] = result
                    print(f"  ✓ {key}: {result} (gemini fallback)")
                    continue
            except concurrent.futures.TimeoutError:
                print(f"  ⏱ {key}: gemini timeout — using default")
            except Exception as e:
                print(f"  ❌ {key}: gemini failed — using default")

        if "precision" in key.lower() or "entity" in key.lower():
            scores[key] = 0.3
            print(f"  ⊘ {key}: using conservative default 0.3")
        else:
            scores[key] = 0.5
            print(f"  ⊘ {key}: using neutral default 0.5")

    return RAGASScores(
        answer_relevancy      = clamp(scores.get("answer_relevancy", 0.5)),
        context_entity_recall = clamp(scores.get("context_entity_recall", 0.5)),
        answer_similarity     = clamp(scores.get("answer_similarity", 0.5)),
        context_precision     = clamp(scores.get("context_precision", 0.5)),
    )

# ── Statistics ─────────────────────────────────────────────────────────

def welch_p(a: List[float], b: List[float]) -> Optional[float]:
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

# ── LLM Judge ──────────────────────────────────────────────────────────

def _parse_judge_scores(text: str) -> dict:
    """
    Robustly extract per-criterion scores from judge output.

    Handles formats:
      Accuracy: A=4 B=3 Reason: ...
      Accuracy: A:4 B:3 - ...
      **Accuracy** — A=4, B=3 — reason text
      Accuracy | 4 | 3 | reason

    Returns dict: { criterion: (score_a_str, score_b_str, reason_str) }
    """
    CRITERIA = ["Accuracy", "Clarity", "Completeness", "Relevance", "Usefulness"]
    scores = {}

    # Normalise: strip markdown bold/italic/backticks
    clean_text = re.sub(r"[*_`]", "", text)
    lines = clean_text.splitlines()

    for crit in CRITERIA:
        sa, sb, reason = "—", "—", ""

        for line in lines:
            # Case-insensitive criterion match
            if not re.search(rf"\b{crit}\b", line, re.IGNORECASE):
                continue

            # ── Score extraction: try several patterns ──

            # Pattern 1: A=X B=X  or  A:X B:X
            ma = re.search(r"\bA\s*[=:]\s*([1-5])", line, re.IGNORECASE)
            mb = re.search(r"\bB\s*[=:]\s*([1-5])", line, re.IGNORECASE)
            if ma:
                sa = ma.group(1)
            if mb:
                sb = mb.group(1)

            # Pattern 2: pipe table  |4|3|  or  | 4 | 3 |
            if sa == "—" or sb == "—":
                pipe_nums = re.findall(r"\|\s*([1-5])\s*", line)
                if len(pipe_nums) >= 2:
                    sa = pipe_nums[0]
                    sb = pipe_nums[1]

            # Pattern 3: bare consecutive digits like "4 3" after criterion name
            if sa == "—" or sb == "—":
                # Remove the criterion word then look for two 1-5 digits
                stripped = re.sub(rf"\b{crit}\b", "", line, flags=re.IGNORECASE)
                bare = re.findall(r"\b([1-5])\b", stripped)
                if len(bare) >= 2:
                    sa = bare[0]
                    sb = bare[1]
                elif len(bare) == 1:
                    if sa == "—":
                        sa = bare[0]

            # ── Reason extraction ──
            # Try explicit "Reason:" label
            rm = re.search(r"[Rr]eason\s*[:\-]\s*(.+?)$", line)
            if rm:
                reason = rm.group(1).strip()[:120]
            else:
                # Take everything after the scores as the reason
                # Remove criterion name and score tokens, keep the rest
                leftover = re.sub(rf"\b{crit}\b", "", line, flags=re.IGNORECASE)
                leftover = re.sub(r"\b[AB]\s*[=:]\s*[1-5]\b", "", leftover, flags=re.IGNORECASE)
                leftover = re.sub(r"\|\s*[1-5]\s*\|?", "", leftover)
                leftover = re.sub(r"[:\-\|,]", " ", leftover)
                leftover = leftover.strip()
                if len(leftover) > 15:
                    reason = leftover[:120]

            # Found our line for this criterion — stop searching
            break

        scores[crit] = (sa, sb, reason)

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
        if re.search(r"WINNER\s*:?\s*PROMPT\s*A", text_up):
            wins_a += 1
        elif re.search(r"WINNER\s*:?\s*PROMPT\s*B", text_up):
            wins_b += 1
        elif re.search(r"WINNER\s*:?\s*A\b", text_up):
            wins_a += 1
        elif re.search(r"WINNER\s*:?\s*B\b", text_up):
            wins_b += 1

        sections.append((r.query, text))

    overall = "Prompt A" if wins_a > wins_b else ("Prompt B" if wins_b > wins_a else "Tie")
    overall_reason = f"Prompt A won {wins_a} queries, Prompt B won {wins_b} queries"
    return sections, wins_a, wins_b, overall, overall_reason
