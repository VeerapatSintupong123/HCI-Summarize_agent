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
WORKER_URL = os.environ.get("WORKER_URL")

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
Headline News: {headline}

You are the Leader (coordinator). Read the Headline News and think about what the worker needs to retrieve and summarize to best assess *financial impact*. 

Produce a concise initial guide for the worker that can also serve as a search query for financial news about NVIDIA, AMD, or INTEL. Focus on retrieving content that includes:
- Key financial metrics: revenue, profit, EPS, margins, growth rates
- Timeframes: quarterly, yearly, or recent updates
- Major company announcements affecting finances: earnings reports, guidance, mergers, product launches
- Priority entities: NVIDIA, AMD, INTEL
- Tone: concise, numbers-first, factual
Return output in 1 string only.
Return the guide as short plain text (NO JSON), optimized for retrieving financial-relevant documents.
"""
    out = AGENT.run(prompt)
    # sanitize
    return out.strip()

def send_to_worker(headline: str, guide: str, worker_url: str = WORKER_URL, k: int = 5):
    payload = {"guide": guide} 
    r = requests.post(worker_url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# def aggregate_and_summarize(headline: str, guide: str, worker_response: Dict[str, Any]) -> str:
#     # build an aggregation prompt for the leader to produce a financial-impact trend
#     prompt = f"""
# You are the Leader: produce a final short analysis (max 300 words) focusing on financial impact trends based on:
# - Headline: {headline}
# - Initial guide: {guide}
# - Worker structured summary (JSON): {json.dumps(worker_response.get("structured", {}), ensure_ascii=False, indent=2)}
# - Worker bullet summary: {worker_response.get("summary", "")}

# Produce:
# 1) One-sentence topline (financial impact)
# 2) 3 short trend bullets (what's increasing, decreasing, or uncertain)
# 3) 1 short recommended next action for analysts/investors

# Return plain text, clearly separated.
# """
#     return AGENT.run(prompt)

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
    return worker_resp

if __name__ == "__main__":
    # quick demo with sample headlines
    sample_headlines = [
        "Nvidia's CEO Just Delivered Incredible News for Taiwan Semiconductor Manufacturing Stock Investors",
    ]
    out = process_headlines(sample_headlines)
    # for r in out:
    #     print("----")
    #     print("Headline:", r["headline"])
    #     print("Final summary:\n", r["final_summary"])
    #     print()
    print(out)