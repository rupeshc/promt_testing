# 🧪 LLM Prompt A/B Testing Framework

> **Evaluate, compare, and decide** — A production-grade framework for testing two LLM prompt strategies head-to-head using RAG, RAGAS metrics, heuristic scoring, and an LLM-as-judge panel.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-phi3%3Amini-black?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [How It Works](#-how-it-works)
- [Evaluation Metrics](#-evaluation-metrics)
- [LLM Judge](#-llm-judge)
- [RAG Pipeline](#-rag-pipeline)
- [API Reference](#-api-reference)
- [Customisation](#-customisation)
- [Troubleshooting](#-troubleshooting)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

This framework lets you pit two prompt strategies against the same set of queries and get a statistically grounded answer to the question: **which prompt performs better?**

It combines:
- **Retrieval-Augmented Generation (RAG)** — so answers are grounded in a local knowledge base
- **Heuristic scoring** — 6 fast metrics computed locally with no external calls
- **RAGAS evaluation** — 4 semantic metrics using embedding models and LLMs
- **LLM-as-Judge** — a third LLM scores both answers on 5 qualitative criteria
- **Statistical analysis** — Welch's t-test, win rates, mean comparisons
- **Visual dashboard** — radar and bar charts rendered in-browser via Chart.js

---

## ✨ Features

| Feature | Description |
|---|---|
| 🅰️🅱️ Side-by-side comparison | Run any two prompt templates against the same queries simultaneously |
| 📚 RAG pipeline | ChromaDB vector store + `all-MiniLM-L6-v2` embeddings over your own knowledge base |
| 📊 Heuristic metrics | Relevance, Completeness, Coherence, Depth, Fluency, Latency — no external API needed |
| 🧬 RAGAS metrics | Answer Relevancy, Context Entity Recall, Answer Similarity, Context Precision |
| 🤖 LLM Judge | `phi3:mini` evaluates each answer pair on Accuracy, Clarity, Completeness, Relevance, Usefulness |
| 📈 Interactive charts | Radar + horizontal bar chart with toggle controls (show A, B, or both) |
| 🔁 Smart fallback | Ollama → Gemini → conservative defaults — the app never crashes on missing API keys |
| 🧮 Statistics | Welch's t-test p-value, mean scores, standard deviation, win rate |
| 🏆 Final verdict | Aggregated judge decision with a rich natural-language explanation |
| 🎨 Dark-mode UI | GitHub-inspired dark theme built with vanilla CSS (zero frontend dependencies) |

---

## 🏗️ Architecture

```
User Browser
     │
     ▼
┌─────────────────────────────────────────────┐
│              FastAPI App (main.py)           │
│                                             │
│  GET /          → Home (query selector +    │
│                   prompt config form)       │
│  POST /run      → Run A/B test, render      │
│                   full results page         │
│  GET /api/shuffle → Random query sample     │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   processors.py     │
          │                     │
          │  init_rag()         │  ◄── knowledge_base.txt
          │  retrieve()         │  ◄── ChromaDB (ephemeral)
          │  generate()         │  ◄── Ollama (phi3:mini)
          │  heuristic()        │
          │  run_ragas()        │  ◄── Ollama (qwen2.5:3b)
          │  llm_judge_all()    │        └── Gemini fallback
          │  welch_p()          │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │     models.py       │
          │                     │
          │  MetricsResult      │
          │  RAGASScores        │
          │  SingleResult       │
          │  PROMPTS config     │
          └─────────────────────┘
```

### Data Flow (per query)

```
Query
  │
  ├─► RAG retrieve(query, n=3)
  │         └─► ChromaDB → top-3 context chunks
  │
  ├─► build_prompt(tpl_A, query, context) → generate() → answer_A  (+ latency)
  ├─► build_prompt(tpl_B, query, context) → generate() → answer_B  (+ latency)
  │
  ├─► heuristic(query, answer_A) → MetricsResult_A
  ├─► heuristic(query, answer_B) → MetricsResult_B
  │
  ├─► run_ragas(query, answer_A, context) → RAGASScores_A
  ├─► run_ragas(query, answer_B, context) → RAGASScores_B
  │
  └─► SingleResult (stored for aggregation)

After all queries:
  ├─► Statistical summary (mean, std, Welch p-value, win rate)
  ├─► llm_judge_all(results) → per-query scores + final verdict
  └─► Render HTML results page
```

---

## 📁 Project Structure

```
llm-ab-testing/
│
├── main.py              # FastAPI app, all routes, HTML rendering
├── processors.py        # RAG, generation, heuristics, RAGAS, judge
├── models.py            # Pydantic models, config constants, prompt templates
│
├── knowledge_base.txt   # Your domain knowledge (plain text, paragraph-separated)
├── queries.txt          # One test query per line
│
├── .env                 # API keys (never commit this)
├── .env.example         # Template for .env
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🔧 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| [Ollama](https://ollama.ai) | Latest | Must be running locally |
| `phi3:mini` model | — | `ollama pull phi3:mini` |
| `qwen2.5:3b` model | — | `ollama pull qwen2.5:3b` (used for RAGAS) |
| Gemini API key | — | Optional — used as RAGAS fallback |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/llm-ab-testing.git
cd llm-ab-testing
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Full requirements.txt</summary>

```
fastapi
uvicorn[standard]
pydantic
requests
python-multipart
python-dotenv

# RAG
chromadb
sentence-transformers

# RAGAS
ragas
datasets
langchain-community
langchain-ollama

# Gemini fallback (optional)
langchain-google-genai
```

</details>

### 4. Pull Ollama models

```bash
# Start Ollama (if not already running as a service)
ollama serve

# In a new terminal
ollama pull phi3:mini
ollama pull qwen2.5:3b
```

### 5. Set up your knowledge base

Edit `knowledge_base.txt` with your domain content. Paragraphs should be separated by blank lines — these become the RAG chunks.

```
Machine learning is a branch of AI...

Neural networks are computational models...

A transformer is a deep learning architecture...
```

### 6. Set up your queries

Edit `queries.txt` — one question per line:

```
What is machine learning?
How does the attention mechanism work?
What is a vector database?
```

---

## ⚙️ Configuration

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# .env
Gemini_API_key=your_gemini_api_key_here
```

> The app also checks `GEMINI_API_KEY` (uppercase) as an alternative variable name.

### Model & timeout settings (`models.py`)

| Constant | Default | Description |
|---|---|---|
| `MODEL` | `phi3:mini` | Main generation + judge model |
| `RAGAS_MODEL` | `qwen2.5:3b` | Model used for RAGAS LLM calls |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini fallback model |
| `TIMEOUT` | `300` | Seconds before Ollama generation times out |
| `RAGAS_TIMEOUT_OLLAMA` | `60` | Per-metric RAGAS timeout via Ollama |
| `RAGAS_TIMEOUT_GEMINI` | `45` | Per-metric RAGAS timeout via Gemini |

### Prompt templates (`models.py`)

Five built-in strategies are available in the UI under "Preset" mode:

| Key | Template |
|---|---|
| `Simple` | `Explain {query} in simple terms.` |
| `Technical` | `Provide a technical explanation of {query}.` |
| `Example` | `Explain {query} with real-world examples.` |
| `StepByStep` | `Explain {query} step by step.` |
| `Comparison` | `Explain {query} and compare it with similar concepts.` |

You can add more by editing the `PROMPTS` dict in `models.py`. Use `{query}` as the placeholder.

---

## ▶️ Running the App

```bash
# Make sure Ollama is running first
ollama serve

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

Or run directly:

```bash
python main.py
```

---

## 🔄 How It Works

### Step 1 — Select Queries

The home page shows a random sample of 6 queries from `queries.txt`. Select up to 5, or click **Shuffle** for a new batch. Selected queries drive the entire A/B test.

### Step 2 — Configure Prompts

Two modes:

- **Manual** — Write your own prompt templates using `{query}` as a placeholder
- **Preset** — Choose from the 5 built-in strategies

### Step 3 — Run

Click **▶ Run**. For each query:

1. Top-3 context chunks are retrieved from ChromaDB
2. Both prompt templates are injected with the query + context
3. `phi3:mini` generates answers for both
4. Heuristic scoring runs locally
5. RAGAS scoring runs via `qwen2.5:3b` (with Gemini fallback)

After all queries complete, the LLM judge evaluates every answer pair and a final verdict is rendered.



## 📏 Evaluation Metrics

### Heuristic Metrics (always available, no API)

These are computed locally from the query text and answer text:

| Metric | Formula | What it measures |
|---|---|---|
| **Relevance** | `len(query_words ∩ answer_words) / len(query_words)` | Keyword overlap — is the answer on-topic? |
| **Completeness** | `len(answer_words) / 150` | Answer length relative to ideal ~150 words |
| **Coherence** | `num_sentences / 10` | Sentence count — is the answer structured? |
| **Depth** | `unique_words / total_words × 2` | Vocabulary diversity — is it rich? |
| **Fluency** | `avg_words_per_sentence / 18` | Sentence length — is it readable? |
| **Latency Score** | `1 - latency_seconds / 60` | Response speed (60s = worst) |

All scores are clamped to `[0.0, 1.0]`. The **average of all 6** is used as the composite heuristic score.

### RAGAS Metrics (require LLM)

| Metric | Method | What it measures |
|---|---|---|
| **Answer Relevancy** | Cosine similarity (embeddings) | Does the answer address the question? |
| **Context Entity Recall** | Entity overlap | Did the context contain the key terms? |
| **Answer Similarity** | Cosine similarity (embeddings) | How close is the answer to the ideal? |
| **Context Precision** | LLM relevance check | Are the retrieved chunks actually useful? |

> **Note on 0.0 scores:** Context Precision and Context Entity Recall can return 0.0 when the retrieved chunks don't contain exact entity matches from the question. This reflects genuine retrieval quality, not a bug.

### Fallback hierarchy for RAGAS

```
Ollama (qwen2.5:3b)
    │ timeout / error
    ▼
Gemini (gemini-2.5-flash-lite)
    │ timeout / error / no key
    ▼
Conservative defaults:
  Context metrics  → 0.3
  Answer metrics   → 0.5
```

---

## ⚖️ LLM Judge

After all heuristic and RAGAS scoring completes, `phi3:mini` is used as an independent judge to evaluate every answer pair. The judge scores on 5 criteria:

| Criterion | Description |
|---|---|
| **Accuracy** | Is the information correct and factually sound? |
| **Clarity** | Is the answer easy to read and understand? |
| **Completeness** | Does it fully address all parts of the question? |
| **Relevance** | Does it stay on topic without irrelevant tangents? |
| **Usefulness** | Would a real user find this genuinely helpful? |

Each criterion is scored **1–5** for both Prompt A and Prompt B, giving a maximum of **25 points per query**.

### Judge prompt format

```
Accuracy: A=X B=X Reason: one sentence
Clarity: A=X B=X Reason: one sentence
Completeness: A=X B=X Reason: one sentence
Relevance: A=X B=X Reason: one sentence
Usefulness: A=X B=X Reason: one sentence
WINNER: Prompt A or Prompt B
REASON: one sentence
```

The parser handles multiple real-world output formats from local models:
- `A=4 B=3` (ideal)
- `A:4 B:3`
- `| 4 | 3 |` (table format)
- Bare consecutive digits after the criterion name

### Final verdict

The bottom of the results page shows a **single comprehensive verdict** (Section 07) combining:

- Judge total scores across all queries
- Heuristic win count and average scores
- Judge win rate (%)
- A multi-sentence natural-language explanation and recommendation

---

## 🗂️ RAG Pipeline

### Indexing (on startup)

1. `knowledge_base.txt` is loaded and split on blank lines into paragraphs
2. Paragraphs longer than 200 words are chunked at 200-word boundaries
3. Each chunk is embedded with `all-MiniLM-L6-v2` via `sentence-transformers`
4. Chunks + embeddings are stored in a ChromaDB **ephemeral** collection (in-memory, rebuilt each run)

### Retrieval (per query)

1. The query is embedded with the same model
2. ChromaDB returns the top-3 most similar chunks by cosine similarity
3. Chunks are injected into the prompt as `[Context 1]`, `[Context 2]`, `[Context 3]`

### Prompt structure with context

```
Use ONLY the context below to answer. Do not use outside knowledge.

[Context 1]: <chunk>
[Context 2]: <chunk>
[Context 3]: <chunk>

Question: <query>

<your prompt template here>
```

---

## 🌐 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Home page — query selector and prompt config |
| `POST` | `/run` | Run the A/B test; returns full results HTML |
| `GET` | `/api/shuffle` | Returns a random sample of 6 queries as JSON |

### POST `/run` — Form parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `selected_queries` | `str` | — | Newline-separated list of selected queries |
| `prompt_mode` | `str` | `manual` | `manual` or `preset` |
| `promptA` | `str` | — | Template for Prompt A (manual mode) |
| `promptB` | `str` | — | Template for Prompt B (manual mode) |
| `autoA` | `str` | `Simple` | Preset key for Prompt A (preset mode) |
| `autoB` | `str` | `Technical` | Preset key for Prompt B (preset mode) |

### GET `/api/shuffle` — Response

```json
{
  "queries": [
    "What is machine learning?",
    "How does RAG work?",
    "..."
  ]
}
```

---

## 🛠️ Customisation

### Add your own knowledge base

Replace `knowledge_base.txt` with any plain-text content. Use blank lines to separate logical sections — each section becomes a retrievable chunk.

### Add prompt strategies

In `models.py`, extend the `PROMPTS` dict:

```python
PROMPTS = {
    "Simple":     "Explain {query} in simple terms.",
    "Technical":  "Provide a technical explanation of {query}.",
    "Socratic":   "Ask three probing questions about {query}, then answer them.",
    "ELI5":       "Explain {query} as if I were five years old.",
}
```

### Change the generation model

In `models.py`:

```python
MODEL = "llama3.2:3b"   # any model available via `ollama list`
```

Make sure to `ollama pull <model>` first.

### Change chunk size

In `processors.py`, inside `init_rag()`:

```python
for i in range(0, len(words), 200):  # change 200 to your preferred chunk size
```

Smaller chunks (100–150 words) improve retrieval precision. Larger chunks (300–500 words) give more context per chunk but reduce precision.

### Persistent ChromaDB

By default the vector store is ephemeral (rebuilt on every startup). To make it persistent:

```python
# processors.py — init_rag()
client = chromadb.PersistentClient(path="./chroma_db")
```

---

## 🔥 Troubleshooting

### `Cannot connect to Ollama`

```bash
# Make sure Ollama is running
ollama serve

# Verify it is accessible
curl http://localhost:11434/api/tags
```

### `Ollama HTTP 404` / model not found

```bash
ollama pull phi3:mini
ollama pull qwen2.5:3b
ollama list     # verify both appear
```

### RAGAS returns all 0.5 defaults

This usually means `qwen2.5:3b` is timing out on RAGAS LLM calls. Try:

1. Increasing `RAGAS_TIMEOUT_OLLAMA` in `models.py` (e.g., `120`)
2. Adding a Gemini API key to `.env` as fallback
3. Using a smaller RAGAS model: `RAGAS_MODEL = "tinyllama"`

### Charts not rendering

Open browser DevTools → Console. If you see `Chart is not defined`, the CDN load failed. Either check your internet connection or host Chart.js locally and update the `<script>` tag in `page()` inside `main.py`.

### `chromadb` import error

```bash
pip install chromadb --upgrade
```

If `chromadb.EphemeralClient()` is not available in your version, the code falls back to `chromadb.Client()` automatically.

### Knowledge base not found

The app looks for `knowledge_base.txt` in the same directory as `main.py`. Check that `BASE_DIR` in `models.py` resolves correctly:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| **Web framework** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Data validation** | [Pydantic v2](https://docs.pydantic.dev/) |
| **LLM inference** | [Ollama](https://ollama.ai) (`phi3:mini`, `qwen2.5:3b`) |
| **LLM fallback** | [Google Gemini](https://ai.google.dev/) via `langchain-google-genai` |
| **Embeddings** | [sentence-transformers](https://sbert.net/) (`all-MiniLM-L6-v2`) |
| **Vector store** | [ChromaDB](https://www.trychroma.com/) (ephemeral, in-memory) |
| **RAG evaluation** | [RAGAS](https://docs.ragas.io/) |
| **LLM orchestration** | [LangChain](https://python.langchain.com/) (thin wrapper for RAGAS) |
| **Charts** | [Chart.js 3.9](https://www.chartjs.org/) (CDN — radar + bar) |
| **Frontend** | Vanilla HTML / CSS / JS (no framework, dark theme) |
| **Font** | IBM Plex Sans + IBM Plex Mono (Google Fonts) |
| **Statistics** | Pure Python (`statistics`, `math`) — Welch's t-test + regularised incomplete beta |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [RAGAS](https://github.com/explodinggradients/ragas) — the open-source RAG evaluation framework
- [Ollama](https://github.com/ollama/ollama) — for making local LLM inference painless
- [ChromaDB](https://github.com/chroma-core/chroma) — the lightweight vector database
- [FastAPI](https://github.com/tiangolo/fastapi) — for the clean, fast API layer

---

*Built for LLM Ops & Eval — RAG · RAGAS · FastAPI · Pydantic*
