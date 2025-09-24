#leader.py
import os
import json
import logging
from dotenv import load_dotenv
from smolagents import OpenAIServerModel, tool, ToolCallingAgent
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from pathlib import Path
from langfuse import get_client

# --- Load env ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leader")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

model = OpenAIServerModel(
    model_id="gemini-2.5-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

# --- Import Worker Agent ---
from agents.worker import agent as worker_agent

# --- Leader Agent (Orchestrator) ---
leader = ToolCallingAgent(
    model=model,
    tools=[],                  # leader ไม่มี tool เอง
    managed_agents=[worker_agent],   # จัดการ worker
    name="Leader",
    description="Coordinates tasks and delegates to worker agent",
    stream_outputs=False,
)


# --- Leader Logic (ปรับปรุงใหม่) ---
def process_news_item(item):
    """
    Processes a single news item to get a summary and impact trend analysis.
    """
    headline = item.get("headline", "")
    content = item.get("content", "")
    logger.info("Processing headline: %s", headline)

    # Prompt ที่สั่งให้ทำ 2 งานและตอบเป็น JSON
    # นี่คือส่วนที่สำคัญที่สุด
    query = f"""
    You are a financial analyst agent. Analyze the following news article.
    
    News Headline: "{headline}"
    News Content: "{content}"

    Perform the following two tasks and provide the output as a single JSON object.
    Do not include any text outside of the JSON object.

    1.  **summary**: Summarize the key points of the news article concisely.
    2.  **impact_analysis**: Analyze the potential impact and trend related to this news.
        -   Consider the short-term and long-term effects on the company (e.g., NVIDIA, AMD, INTEL).
        -   Mention the potential impact on stock price, market sentiment, and competitive landscape.
        -   Feel free to use your tools to search for the current date, recent stock performance, or related historical events to enrich your analysis.

    Your response MUST be a valid JSON object with the keys "summary" and "impact_analysis".
    Example format:
    {{
      "summary": "NVIDIA announced a new AI chip...",
      "impact_analysis": "This announcement is expected to positively impact NVIDIA's stock price in the short term..."
    }}
    """
    
    worker_response_str = leader.run(query)
    
    # พยายามแปลงผลลัพธ์จาก Worker ให้เป็น Dictionary
    try:
        # LLM อาจจะตอบกลับมาพร้อมกับ Markdown code block, เราต้อง clean ก่อน
        if worker_response_str.strip().startswith("```json"):
            worker_response_str = worker_response_str.strip()[7:-3]
        
        worker_data = json.loads(worker_response_str)
        summary = worker_data.get("summary", "Failed to get summary.")
        impact_trend = worker_data.get("impact_analysis", "Failed to get impact analysis.")
    except (json.JSONDecodeError, AttributeError):
        logger.error(f"Could not parse JSON response from worker: {worker_response_str}")
        summary = "Error: Worker returned an invalid format."
        impact_trend = worker_response_str # เก็บคำตอบดิบไว้ในกรณีที่เกิดข้อผิดพลาด

    # ประกอบผลลัพธ์สุดท้าย
    result = {
        "headline": headline,
        "content": content,
        "worker_summary": summary,
        "impact_trend": impact_trend,
    }
    return result

if __name__ == "__main__":
    # --- Configuration ---
    # ระบุชื่อบริษัทที่ต้องการวิเคราะห์ใน List นี้
    TARGET_COMPANIES = ["Nvidia"] 
    
    # กำหนดจำนวนข่าวสูงสุดที่ต้องการประมวลผล (ใส่ None ถ้าต้องการทั้งหมด)
    NEWS_LIMIT = 1
    # ---------------------

    # โหลดข่าว
    current_dir = Path(__file__).parent
    json_path = current_dir.parent / "scrape_news" / "22092025.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: News file not found at {json_path}")
        exit() # ออกจากโปรแกรมถ้าไม่เจอไฟล์

    all_items = []
    # รวบรวมข่าวจากบริษัทที่ระบุใน TARGET_COMPANIES
    print(f"Filtering news for companies: {', '.join(TARGET_COMPANIES)}")
    for company, items in data.items():
        if company in TARGET_COMPANIES:
            all_items.extend(items)

    # จำกัดจำนวนข่าวตามที่กำหนดใน NEWS_LIMIT
    if NEWS_LIMIT is not None and NEWS_LIMIT > 0:
        print(f"Limiting to the first {NEWS_LIMIT} articles.")
        all_items = all_items[:NEWS_LIMIT]
    
    if not all_items:
        print("No news items found for the specified companies. Exiting.")
        exit()

    print(f"\nStarting processing for {len(all_items)} news article(s)...\n")
    
    final_results = []
    # ประมวลผลทีละข่าว
    for item in all_items:
        processed_result = process_news_item(item)
        final_results.append(processed_result)

    # ตั้งชื่อไฟล์ผลลัพธ์แบบไดนามิก
    company_str = "_".join(TARGET_COMPANIES).lower()
    results_filename = f"results_{company_str}_{len(final_results)}items.json"
    results_path = current_dir.parent / "results" / results_filename
    
    # สร้างโฟลเดอร์ results ถ้ายังไม่มี
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed analysis results saved to {results_path}")
