import os
import json
import logging
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel
from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leader")

HF_MODEL_ID = os.environ.get("HF_LEADER_MODEL_ID", None)
WORKER_URL = os.environ.get("WORKER_URL")

langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

def make_agent():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model = InferenceClientModel(model_id=HF_MODEL_ID, token=hf_token)
    return CodeAgent(tools=[], model=model, add_base_tools=False)

AGENT = make_agent()

def generate_initial_guide(content: str) -> str:
    prompt = f"""
Content: {content}

You are the Leader (coordinator). Read the Content and think about what the worker needs to retrieve and summarize to best assess *financial impact*.

Produce a concise initial guide for the worker that can also serve as a search query for financial news about NVIDIA, AMD, or INTEL. Focus on retrieving content that includes:
- Key financial metrics: revenue, profit, EPS, margins, growth rates
- Timeframes: quarterly, yearly, or recent updates
- Major company announcements affecting finances: earnings reports, guidance, mergers, product launches
- Priority entities: NVIDIA, AMD, INTEL
- Tone: concise, numbers-first, factual

Return output in 1 string only and in 1 line and not use bullet.
Return the guide as short plain text (NO JSON).
- Output format example: "NVIDIA AMD INTEL revenue profit EPS margins growth rates quarterly yearly earnings reports guidance mergers product
  launches financial news"

"""
    out = AGENT.run(prompt)
    return out.strip()

def send_to_worker(guide: str, worker_url: str = WORKER_URL):
    payload = {"guide": guide}
    r = requests.post(worker_url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def analyze_impact_trend(summary: str) -> str:
    prompt = f"""
You are a financial trend analyst. Read the following summary:

{summary}

Based only on the financial context, provide a short **financial impact trend analysis**
for NVIDIA, AMD, or INTEL.

Return output in 1 string only and in 1 line and not use bullet.
Return the guide as short plain text (NO JSON).
- Be concise, professional, and numbers/impact oriented.
- Output format example: "Positive – Nvidia revenue growth and strong AI chip demand"
"""
    out = AGENT.run(prompt)
    return out.strip()

def process_items(items: List[Dict[str, Any]], worker_url: str = WORKER_URL, limit: int = None):
    results = []
    if limit:
        items = items[:limit]  # จำกัดจำนวนข่าว
    for item in items:
        headline = item.get("headline", "")
        content = item.get("content", "")
        logger.info("Processing headline: %s", headline)

        guide = generate_initial_guide(content)
        worker_resp = send_to_worker(guide, worker_url)
        worker_summary = worker_resp.get("summary", "")
        impact_trend = analyze_impact_trend(worker_summary)

        result = {
            "headline": headline,
            "content": content,
            "guide": guide,
            "worker_summary": worker_summary,
            "impact_trend": impact_trend,
        }
        results.append(result)
    return results

if __name__ == "__main__":
    # โหลดไฟล์ข่าวจาก JSON
    current_dir = Path(__file__).parent
    json_path = current_dir.parent / "scrape_news" / "22092025.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # รวมข่าวจากทุก entity เป็น list เดียว
    all_items = []
    for company, items in data.items():
        all_items.extend(items)

    # 🔹 เลือกจำนวนข่าวที่จะประมวลผล (None = ทั้งหมด)
    LIMIT = 1  

    out = process_items(all_items, limit=LIMIT)

    # 🔹 บันทึกผลลัพธ์เป็นไฟล์ JSON โดย append ต่อไปเรื่อยๆ
    results_path = current_dir.parent / "results" / "result.json"


    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to {results_path}")


