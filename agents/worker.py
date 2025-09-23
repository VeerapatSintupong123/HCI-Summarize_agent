# worker.py
"""
Worker service:
- POST /process  -> accepts {"headline": "...", "initial_guide": "...", "k": 5}
- Returns JSON with: rich_query, retrieved_count, summary_text, structured_summary
"""

import os
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests

# smolagents
from smolagents import CodeAgent, InferenceClientModel

# LangChain FAISS loader (adjust if you use another vector DB)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    # Fallback imports (depending on your langchain version)
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

# ---------- Config ----------
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", None)  # if None, default inside smolagents will be used
VECTOR_PATH = os.environ.get("VECTOR_PATH", "mock_news_vector_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = 5
# ----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

app = FastAPI(title="news-worker")

# Pydantic models
class ProcessRequest(BaseModel):
    headline: str
    initial_guide: str
    k: int = TOP_K_DEFAULT
    metadata: Dict[str, Any] = {}

class ProcessResponse(BaseModel):
    rich_query: str
    retrieved_count: int
    summary: str
    structured: Dict[str, Any]


# ---------- Helper: load model & vectorstore ----------
def make_agent():
    model = InferenceClientModel(model_id=HF_MODEL_ID) if HF_MODEL_ID else InferenceClientModel()
    agent = CodeAgent(tools=[], model=model, add_base_tools=False)
    return agent

def load_vectorstore(path: str, embedding_name: str):
    # load your FAISS local index - adapt if you use another format
    logger.info("Loading embeddings and vectorstore (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    db = FAISS.load_local(path, embeddings)
    logger.info("Vectorstore loaded.")
    return db

# Initialize (lazy load)
AGENT = None
VECTOR_DB = None

def ensure_initialized():
    global AGENT, VECTOR_DB
    if AGENT is None:
        AGENT = make_agent()
    if VECTOR_DB is None:
        try:
            VECTOR_DB = load_vectorstore(VECTOR_PATH, EMBEDDING_MODEL)
        except Exception as e:
            logger.error("Failed to load vectorstore: %s", e)
            VECTOR_DB = None

# ---------- Core steps ----------
def generate_rich_query(headline: str, initial_guide: str, agent: CodeAgent) -> str:
    prompt = f"""
You are a query engineer. Given the headline and an initial guide for a worker, produce a single concise, high-signal search query suitable for retrieving relevant news chunks
from a RAG system (FAISS). Make it 1-2 short lines, include company/ticker/entity names, and important numeric/temporal keywords if present.
Return only JSON: {{ "query": "<your query>" }}
Headline: \"\"\"{headline}\"\"\"
Initial guide: \"\"\"{initial_guide}\"\"\"
"""
    raw = agent.run(prompt)
    # try to parse JSON; fall back to raw text
    try:
        parsed = json.loads(raw)
        return parsed.get("query", str(parsed))
    except Exception:
        # try to find a JSON object substring
        start = raw.find("{")
        if start >= 0:
            try:
                parsed = json.loads(raw[start:])
                return parsed.get("query", raw)
            except Exception:
                pass
        # last resort: return raw (stripped)
        return raw.strip().strip('"')

def retrieve_chunks(query: str, k: int = TOP_K_DEFAULT):
    if VECTOR_DB is None:
        raise RuntimeError("Vector DB not loaded.")
    docs = VECTOR_DB.similarity_search(query, k=k)
    # Each doc is often a langchain Document; get text + metadata
    results = []
    for d in docs:
        text = getattr(d, "page_content", None) or str(d)
        meta = getattr(d, "metadata", {}) or {}
        results.append({"text": text, "metadata": meta})
    return results

def summarize_chunks(headline: str, initial_guide: str, query: str, chunks: List[Dict[str, Any]], agent: CodeAgent) -> Dict[str, Any]:
    # Concatenate chunk texts but carefully truncate per chunk to avoid token overflow
    def safe_truncate(s: str, max_chars=1200):
        return s if len(s) <= max_chars else s[:max_chars] + " ...[truncated]"

    enumerated = []
    for i, ch in enumerate(chunks, start=1):
        enumerated.append(f"--- CHUNK {i} ---\n{safe_truncate(ch['text'], max_chars=1500)}\nMETADATA: {json.dumps(ch.get('metadata', {}))}\n")
    payload_text = "\n\n".join(enumerated) or "NO CHUNKS_FOUND"

    prompt = f"""
You are a concise news summarizer specialized in financial impact.
Given:
- Headline: {headline}
- Initial guide for the worker: {initial_guide}
- Rich retrieval query used: {query}

Here are the retrieved chunks (each separated). Produce:
1) A short structured JSON object with keys:
   - "topline": one-sentence summary emphasizing *financial impact* (revenue, costs, guidance, stock moves)
   - "entities": list of {{"name":..., "role":..., "evidence": "..."}}
   - "numbers": list of {{"value": "...", "context":"...", "source_chunk": i}}
   - "confidence": "low/medium/high"
   - "recommendation": one-sentence suggested follow-up/action
2) Then produce a short human-readable bullet summary (3-6 bullets) focusing on trend / financial implications.

Return valid JSON for the structured object, followed by the bullets (separated by a newline).
CHUNKS:
{payload_text}
"""
    raw = agent.run(prompt)
    # Attempt to split JSON + bullets
    # Find first JSON object and parse it
    first_brace = raw.find("{")
    last_brace = raw.find("}", first_brace) if first_brace != -1 else -1
    structured = {}
    bullets = raw
    if first_brace != -1:
        # Try to find matching JSON block by searching for the end '}' (simple heuristic)
        # If agent returns JSON then text, try to parse progressively until success
        for end in range(raw.find("}", first_brace) + 1, min(len(raw), first_brace + 2000)):
            try:
                candidate = raw[first_brace:end]
                parsed = json.loads(candidate)
                structured = parsed
                bullets = raw[end:].strip()
                break
            except Exception:
                continue
        if not structured:
            # as a fallback try to load full raw
            try:
                structured = json.loads(raw)
                bullets = ""
            except Exception:
                structured = {"note": "could not parse structured JSON; see raw output"}
                bullets = raw

    return {"structured": structured, "bullets": bullets, "raw": raw}


# ---------- API ----------
@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    try:
        ensure_initialized()
        rich_q = generate_rich_query(req.headline, req.initial_guide, AGENT)
        # retrieve
        chunks = retrieve_chunks(rich_q, k=req.k)
        retrieved_count = len(chunks)
        summary_obj = summarize_chunks(req.headline, req.initial_guide, rich_q, chunks, AGENT)
        structured = summary_obj["structured"]
        # pack a short summary text (bullets if available)
        summary_text = summary_obj.get("bullets") or summary_obj.get("raw", "")
        return ProcessResponse(
            rich_query=rich_q,
            retrieved_count=retrieved_count,
            summary=summary_text if isinstance(summary_text, str) else str(summary_text),
            structured=structured,
        )
    except Exception as e:
        logger.exception("Error in /process")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    ensure_initialized()
    uvicorn.run("worker:app", host="0.0.0.0", port=8001, log_level="info")
