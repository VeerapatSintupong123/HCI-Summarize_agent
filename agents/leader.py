# leader.py
"""
Leader:
- Generates initial guide per headline
- Sends request to worker(s)
- Receives worker summaries, aggregates them, produces final financial-impact trend summary
"""

import os
import json
import time
import logging
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from smolagents import CodeAgent, InferenceClientModel
from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

load_dotenv()  # โหลด .env เข้ามาเป็น environment variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leader")

HF_MODEL_ID = os.environ.get("HF_LEADER_MODEL_ID", None)
WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:8001/process")

langfuse = get_client()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

def make_agent():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model = InferenceClientModel(model_id=HF_MODEL_ID, token=hf_token)
        
    return CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
    )

AGENT = make_agent()

def generate_initial_guide(headline: str) -> str:
    prompt = f"""
You are the Leader (coordinator). Read the headline and "think" about what the worker needs to retrieve and summarize to best assess *financial impact*.
Produce a short initial guide (3-6 short bullets) aimed at the Worker. Include:
- the exact focus (financial items, KPIs, timeframe)
- the desired granularity (high-level vs numbers)
- any entities to prioritize (companies, government bodies)
- desired tone (concise, numbers-first)
Return the guide as a short plain text (no JSON).
Headline: {headline}
"""
    out = AGENT.run(prompt)
    # sanitize
    return out.strip()

def send_to_worker(headline: str, guide: str, worker_url: str = WORKER_URL, k: int = 5):
    payload = {"headline": headline, "initial_guide": guide, "k": k}
    r = requests.post(worker_url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def aggregate_and_summarize(headline: str, guide: str, worker_response: Dict[str, Any]) -> str:
    # build an aggregation prompt for the leader to produce a financial-impact trend
    prompt = f"""
You are the Leader: produce a final short analysis (max 300 words) focusing on financial impact trends based on:
- Headline: {headline}
- Initial guide: {guide}
- Worker structured summary (JSON): {json.dumps(worker_response.get("structured", {}), ensure_ascii=False, indent=2)}
- Worker bullet summary: {worker_response.get("summary", "")}

Produce:
1) One-sentence topline (financial impact)
2) 3 short trend bullets (what's increasing, decreasing, or uncertain)
3) 1 short recommended next action for analysts/investors

Return plain text, clearly separated.
"""
    return AGENT.run(prompt)

def process_headlines(headlines: List[str], worker_url: str = WORKER_URL):
    results = []
    for h in headlines:
        logger.info("Processing headline: %s", h)
        guide = generate_initial_guide(h)
        logger.info("Initial guide:\n%s", guide)
        worker_resp = send_to_worker(h, guide, worker_url)
        logger.info("Worker returned: retrieved_count=%s", worker_resp.get("retrieved_count"))
        # final = aggregate_and_summarize(h, guide, worker_resp)
        # results.append({"headline": h, "guide": guide, "worker": worker_resp, "final_summary": final})
    return guide

if __name__ == "__main__":
    # quick demo with sample headlines
    sample_headlines = [
        "Acme Corp reports Q2 revenue up 12% on stronger cloud sales; raises FY guidance.",
        "Major chipmaker announces plant shutdown after safety incident; production to be delayed 6 weeks.",
    ]
    out = process_headlines(sample_headlines)
    # for r in out:
    #     print("----")
    #     print("Headline:", r["headline"])
    #     print("Final summary:\n", r["final_summary"])
    #     print()
    print(out)