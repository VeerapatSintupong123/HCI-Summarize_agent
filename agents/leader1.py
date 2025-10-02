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
HF_LEADER_MODEL_ID = os.getenv("HF_LEADER_MODEL_ID", "gemini-2.5-flash")

config_json = os.getenv("COMPANY_CONFIG")
COMPANY_CONFIG = json.loads(config_json)
print(COMPANY_CONFIG)

langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

model = OpenAIServerModel(
    model_id=HF_LEADER_MODEL_ID,
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

# --- Import Worker Agent ---
from agents.worker import summary_worker_agent,analysis_worker_agent

# --- Leader Agent (Orchestrator) ---
leader = ToolCallingAgent(
    model=model,
    tools=[],                  # leader ไม่มี tool เอง
    managed_agents=[summary_worker_agent, analysis_worker_agent],   # จัดการ worker
    name="Leader1",
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

    # --- Start: Recommended Prompt (Updated) ---
    query = f"""
    You are the Leader Agent, an expert orchestrator. Your primary goal is to manage a team of specialist agents to process a news article and produce a combined JSON output.

    **Available Agents:**
    - `Summary_Worker_Agent`: Specializes in summarizing text.
    - `Analysis_Worker_Agent`: Specializes in analyzing financial impact and trends.

    **Input Data:**
    - News Headline: "{headline}"
    - News Content: "{content}"

    **Your Task Instructions (The Plan):**
    1.  **Delegation for Summary:** First, delegate the task of summarizing the provided news content to the `Summary_Worker_Agent`.
    **Crucially, you MUST instruct it to first use the `local_retriever_tool` to search for other relevant news articles published on the same day.** The goal is to identify related events or announcements. The final summary must then integrate the content of the main article with the context from any related same-day news it finds, providing a holistic overview.
    - **MANDATORY DELAY:** After it has generated the summary, you **MUST** instruct it to call the `delay_tool` with `seconds=90` before finishing its turn. This is a critical step for rate limit management.
    
    2.  **Delegation for Comprehensive Analysis:** Second, delegate the analysis task to the `Analysis_Worker_Agent`. Instruct it to provide a **comprehensive yet accessible analysis of the financial impact and resulting trends.
    **Crucially, you MUST instruct it to also use the `local_retriever_tool` to find related financial news, market trends, or competitor announcements from the same day.** After retrieving this vital same-day context, it must perform a comprehensive analysis. The analysis should explain how the main news item, when viewed alongside other events of the day, impacts financial trends and market sentiment.
        - Identify all key financial implications (both positive and negative).
        - Consider potential short-term and long-term effects on the company's market position and stock value.
        - Be written in clear, professional language, ensuring the insights are easy to understand and can be utilized for strategic decision-making.
    
    3.  **Final Output Generation:** After receiving the results from both agents, combine them into a single, final JSON object.

    **Strict Output Requirements:**
    Your final response MUST be a single, compact line of valid JSON. Do not include any text, explanations, or markdown code blocks. All special characters must be properly escaped.

    **Example of a valid, single-line response:**
    {{"summary": "NVIDIA announced a new AI chip...", "financial_impact_trend": "The introduction of the new AI chip is expected to positively impact NVIDIA's stock by reinforcing its technological leadership. Short-term, this could lead to a stock price increase due to investor confidence. Long-term, it solidifies their competitive advantage against rivals, potentially leading to sustained revenue growth in the data center segment."}}
    """
    worker_response_str = leader.run(query)
    
    # พยายามแปลงผลลัพธ์จาก Worker ให้เป็น Dictionary
    try:
        # LLM อาจจะตอบกลับมาพร้อมกับ Markdown code block, เราต้อง clean ก่อน
        if worker_response_str.strip().startswith("```json"):
            worker_response_str = worker_response_str.strip()[7:-3]
        
        worker_data = json.loads(worker_response_str)
        summary = worker_data.get("summary", "Failed to get summary.")
        financial_impact_trend = worker_data.get("financial_impact_trend", "Failed to get impact analysis.")
    except (json.JSONDecodeError, AttributeError):
        logger.error(f"Could not parse JSON response from worker: {worker_response_str}")
        summary = "Error: Worker returned an invalid format."
        financial_impact_trend = worker_response_str # เก็บคำตอบดิบไว้ในกรณีที่เกิดข้อผิดพลาด

    # ประกอบผลลัพธ์สุดท้าย
    result = {
        "headline": headline,
        "content": content,
        "summary": summary,
        "financial_impact_trend": financial_impact_trend,
    }
    return result

if __name__ == "__main__":
    # --- Configuration ---
    # ระบุบริษัทและจำนวนข่าวที่ต้องการในรูปแบบ Dictionary
    # Key คือชื่อบริษัท, Value คือจำนวนข่าว (ใช้ 0 หากต้องการทั้งหมด)
    # COMPANY_CONFIG = {
    #     "Nvidia": 1,
    #     "Intel": 1,
    #     "AMD": 1,
    # }
    # ---------------------

    # 1. โหลดข้อมูลข่าวจากไฟล์
    current_dir = Path(__file__).parent
    input_filename = os.getenv("TODAY_NEWS_FILENAME")
    json_path = current_dir.parent / "scrape_news" / input_filename
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
    results_filename = f"lab1.json"
    results_path = current_dir.parent / "results" / results_filename
    
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed analysis results saved to {results_path}")
