#leader.py
import os
import json
from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from agents.worker import summary_worker_agent,analysis_worker_agent
from langfuse import observe, get_client

load_dotenv()
NEWS_DATE_FILE = os.getenv("NEWS_DATE_FILE", "") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_LEADER_MODEL_ID = os.getenv("HF_LEADER_MODEL_ID", "gemini-2.5-flash")

headlines = ""
data_path = os.path.join("data/query", NEWS_DATE_FILE)

langfuse = get_client()
if langfuse.auth_check():
    print("✅ Langfuse client is authenticated and ready!")
else:
    print("❌ Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

model = OpenAIServerModel(
    model_id=HF_LEADER_MODEL_ID,
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

leader = ToolCallingAgent(
    model=model,
    tools=[],                 
    managed_agents=[summary_worker_agent, analysis_worker_agent],
    name="Leader1",
    description="Coordinates tasks and delegates to worker agent",
    stream_outputs=False,
)
print("✅ Leader Agent initialized.")

with open(data_path, 'r', encoding='utf-8') as f:
    news_data = json.load(f)
    chipmakers = list(news_data.keys())
    for chipmaker in chipmakers:
        headlines += f"\n\n### {chipmaker} ###\n"
        articles = news_data[chipmaker]
        for article in articles:
            headlines += article.get("headline", "") + "\n"

print("✅ News data loaded.")

today = NEWS_DATE_FILE.split('.')[0]


initial_guide = f"""
**Input Data:**
- News Headline: "{headlines}"

**Initial Context Setting:**
1. Carefully read and understand the headline. This is your primary subject.
2. Before doing any summarization or analysis, you MUST call the `local_retriever_tool` with the headline (or related keywords) to fetch same-day related articles or context.
3. Treat the retrieved context as essential background knowledge. Use it together with the headline to provide a richer, more accurate output.
"""

query = f"""
You are the Leader Agent, an expert orchestrator. Your primary goal is to manage a team of specialist agents to process a news article and produce a combined JSON output.

**Today is {today}.**

**Available Agents:**
- `Summary_Worker_Agent`: Specializes in summarizing text.
- `Analysis_Worker_Agent`: Specializes in analyzing financial impact and trends.

{initial_guide}

**Your Task Instructions (The Plan):**
1.  **Delegation for Summary:** First, delegate the task of summarizing the provided news content to the `Summary_Worker_Agent`.
**Crucially, you MUST instruct it to first use the `local_retriever_tool` to search for other relevant news articles published on the same day.** The goal is to identify related events or announcements. The final summary must then integrate the content of the main article with the context from any related same-day news it finds, providing a holistic overview.
    - **MANDATORY DELAY:** Before Start this step, you **MUST** instruct it to call the `delay_tool` with `seconds=90`. This is a critical step for rate limit management.

2.  **Delegation for Comprehensive Analysis:** Second, delegate the analysis task to the `Analysis_Worker_Agent`. Instruct it to provide a **comprehensive yet accessible analysis of the financial impact and resulting trends.
**Crucially, you MUST instruct it to also use the `local_retriever_tool` to find related financial news, market trends, or competitor announcements from the same day.** After retrieving this vital same-day context, it must perform a comprehensive analysis. The analysis should explain how the main news item, when viewed alongside other events of the day, impacts financial trends and market sentiment.
    - Identify all key financial implications (both positive and negative).
    - Consider potential short-term and long-term effects on the company's market position and stock value.
    - Be written in clear, professional language, ensuring the insights are easy to understand and can be utilized for strategic decision-making.

3. Final Output Generation:
    The final output must be written in easy word and natural language, suitable for a decision-maker who needs to quickly grasp the situation and its implications.

After collecting the results from both agents, generate a cohesive report in the following format:
- Title: Summary Report of Financial News ({today})
- Paragraph: A well-structured paragraph that integrates the summary from the Summary_Worker_Agent and the analysis from the Analysis_Worker_Agent. This should highlight the main events and their financial implications in clear, natural language.
- Key Insight (short paragraph): A concise, standalone paragraph that synthesizes the most important takeaway from both the summary and the analysis, emphasizing the broader financial significance of the news. It should be slightly more detailed than a single sentence but remain compact and impactful.
- Bullet Points: A concise list of the most significant implications or insights, making it easy to understand the critical points at a glance.

Example Output:
### Summary Report of Financial News (26/01/2025)

### Summary Paragraph
Paragraph

### Key Insight
Key Insight

### Key Implications
Bullet Points

"""

trace_name = f"Leader1_{today}"
@observe()
def process_request(query):
    # Add to the current trace
    langfuse.update_current_trace(session_id="1", name=trace_name)
    return leader.run(query)
response = process_request(query)

with open('summary_leader1.md', 'w', encoding='utf-8') as f:
    f.write(response)