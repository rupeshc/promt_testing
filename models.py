"""
LLM A/B Testing Framework — Models & Configuration
"""

from pydantic import BaseModel, Field
from typing import Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .env file loaded")
except ImportError:
    pass

# ── Model Configuration ──────────────────────────────────────────────────
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

# ── Prompt Templates ────────────────────────────────────────────────────
PROMPTS = {
    "Simple":     "Explain {query} in simple terms.",
    "Technical":  "Provide a technical explanation of {query}.",
    "Example":    "Explain {query} with real-world examples.",
    "StepByStep": "Explain {query} step by step.",
    "Comparison": "Explain {query} and compare it with similar concepts.",
}

# ── Info Tooltips ────────────────────────────────────────────────────────
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

# ── Pydantic Models ─────────────────────────────────────────────────────
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
