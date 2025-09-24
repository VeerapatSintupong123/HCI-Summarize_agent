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
    2.  **impact_analysis**: Briefly analyze the potential impact and trend related to this news.
        -   Consider the short-term and long-term effects on the company (e.g., NVIDIA, AMD, INTEL).
        -   Mention the potential impact on stock price, market sentiment, and competitive landscape.
        -   Feel free to use your tools to search for the current date, recent stock performance, or related historical events to enrich your analysis.

    Perform the following two tasks and provide the output as a single JSON object.
    Your final response MUST be a single, compact line of valid JSON.
    All special characters, especially newlines within the text values, must be properly escaped (e.g., use \\n instead of a literal newline).
    Do not include markdown code blocks like ```json or any text outside of the JSON object itself.

    Example of a valid, single-line response:
    {{"summary": "NVIDIA announced a new AI chip...", "impact_analysis": "This announcement is expected to positively impact NVIDIA's stock price in the short term by demonstrating continued innovation.\\nIn the long term, it could solidify their market leadership."}}
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
    # ระบุบริษัทและจำนวนข่าวที่ต้องการในรูปแบบ Dictionary
    # Key คือชื่อบริษัท, Value คือจำนวนข่าว (ใช้ None หากต้องการทั้งหมด)
    COMPANY_CONFIG = {
        "Nvidia": 1,
        "Intel": 1,
        "AMD": 1,
    }
    # ---------------------

    # 1. โหลดข้อมูลข่าวจากไฟล์
    current_dir = Path(__file__).parent
    json_path = current_dir.parent / "scrape_news" / "22092025.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_news_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: News file not found at {json_path}")
        exit()

    # 2. รวบรวมข่าวตาม Configuration
    items_to_process = []
    for company, limit in COMPANY_CONFIG.items():
        company_news = all_news_data.get(company, []) # ดึงข่าวของบริษัท, ถ้าไม่เจอก็ได้ List ว่าง
        
        # จำกัดจำนวนข่าวตาม limit (ถ้า limit ไม่ใช่ None)
        if limit is not None and limit > 0:
            company_news = company_news[:limit]
            
        items_to_process.extend(company_news)
        print(f"Collected {len(company_news)} articles for {company}.")

    if not items_to_process:
        print("No news items found for the specified configuration. Exiting.")
        exit()

    # 3. ประมวลผลข่าวที่รวบรวมมา
    print(f"\nStarting processing for a total of {len(items_to_process)} news article(s)...\n")
    final_results = [process_news_item(item) for item in items_to_process]

    # 4. บันทึกผลลัพธ์
    results_filename = f"result.json"
    results_path = current_dir.parent / "results" / results_filename
    
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed analysis results saved to {results_path}")
