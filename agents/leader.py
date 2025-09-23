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

def analyze_impact_trend(summary: str) -> str:
    """
    ใช้ agent วิเคราะห์ว่า summary มีผลกระทบทางการเงินแนวโน้มอย่างไร
    เช่น: Positive, Negative, Neutral หรือ Trend: Upward, Downward, Mixed
    """
    prompt = f"""
You are a financial trend analyst. Read the following summary:

{summary}

Based only on the financial context, categorize the **impact trend** for NVIDIA, AMD, or INTEL as one of:
- Positive (improving financials, growth, strong guidance)
- Negative (decline in revenue, weak profit, poor outlook)
- Neutral/Mixed (uncertain, balanced, no clear trend)

Return just one short label (Positive / Negative / Neutral).
"""
    out = AGENT.run(prompt)
    return out.strip()

def process_headlines(headlines: List[str], worker_url: str = WORKER_URL):
    results = []
    for h in headlines:
        logger.info("Processing headline: %s", h)
        guide = generate_initial_guide(h)
        logger.info("Initial guide:\n%s", guide)
        worker_resp = send_to_worker(h, guide, worker_url)
        worker_summary = worker_resp.get("summary", "")
        impact_trend = analyze_impact_trend(worker_summary)

        result = {
            "headline": h,
            "guide": guide,
            "worker_summary": worker_summary,
            "impact_trend": impact_trend
        }
        results.append(result)
    return results


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