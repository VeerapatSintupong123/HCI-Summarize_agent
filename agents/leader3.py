#leader.py
import os
import json
from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from agents.worker import summary_worker_agent,analysis_worker_agent
from agents.graph_retriever import graph_retriever
from agents.enhanced_searcher import enhanced_search_agent
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
    managed_agents=[summary_worker_agent, analysis_worker_agent, graph_retriever, enhanced_search_agent],
    name="Leader3",
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
today = today[:2] + '/' + today[2:4] + '/' + today[4:]

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
- `graph_retriever`: Specializes in analyzing knowledge graphs and extracting relationships from historical news (7 day later).
- `enhanced_search_agent`: A powerful search specialist that finds recent, high-impact business and financial news. It automatically expands queries, filters by source credibility, and analyzes the significance of search results, returning structured JSON.

{initial_guide}

**Importance**
- Use information from all four agents to create a comprehensive and insightful final report. The summary provides the factual basis, the analysis offers depth and implications, and the graph_retriever adds context and relationships that enhance understanding. Together, they ensure the final output is well-rounded, informative, and actionable.
- Keep in mind date: {today} is the date of the news article.

**Your Task Instructions (The Plan):**
1.  **Extract Graph Context (Detailed Sub-plan):** Delegate to the `graph_retriever` (graph_retriever use in 7 day ago news to get context.) with the following logical steps:
    a. **Identify Primary Company:** First, determine the primary chipmaker (NVIDIA, AMD, or Intel) from the news headline.
    b. **Get Specific Summary:** Use `get_7day_summary` for that primary chipmaker to get focused recent context.
    c. **Find All Related Entities:** Use `get_entities_from_chipmaker` to list all known associated entities (companies, products, people).
    d. **Deep Dive on Relationships:** This is crucial. Identify other key entities mentioned *in the news content*. For each of these secondary entities, use `get_relations_between_entities` to find the precise relationship between the primary company and the secondary entity. This will uncover the direct implications of the news.
    e. **Consolidate Findings:** Combine all retrieved information (summary, entity list, and specific relationships) into a structured context report.    
    - **MANDATORY DELAY:** Before Start this step, you **MUST** instruct it to call the `delay_tool` with `seconds=90`. This is a critical step for rate limit management.
    f. **Output Format:** add relation of triplets like "(Entity A) --[Relationship]--> (Entity B)"

2.  **Gather High-Impact Market Context (Detailed Sub-plan):** Delegate to the `enhanced_search_agent` to gather broader market and competitive context. Your goal is to find other significant, recent news that helps understand the landscape surrounding the main article.
    a. **Identify Key Search Terms:** First, extract the core companies, products, and concepts from the news content (e.g., "NVIDIA H200", "data center revenue", "AMD MI300X", "Intel foundry").
    b. **Formulate Strategic Queries:** For each key term, formulate a concise query for the `enhanced_internet_search` tool. The tool will automatically expand these queries for business context. Focus on finding related business news, not encyclopedic definitions. For example:
        - "NVIDIA H200 data center market"
        - "AMD financial guidance 2025"
        - "Intel competition TSMC"
    c. **Specify Search Focus:** Instruct the agent to use a `focus` of "financial" or "business". This leverages the tool's ability to filter out consumer-focused content and prioritize high-credibility business news sources.
    d. **Consolidate Findings:** The tool will return a structured JSON list of pre-analyzed news articles. Consolidate this JSON output into a "market context briefing". This briefing, containing a list of relevant, high-impact articles with their significance rating, will provide critical external context for the summary and analysis agents.
    - **MANDATORY DELAY:** Before Start this step, you **MUST** instruct it to call the `delay_tool` with `seconds=90`. This is a critical step for rate limit management.
    
3.  **Delegation for Summary:** Delegate the task of summarizing the provided news content to the `Summary_Worker_Agent`.
    - It must use the **context report from `graph_retriever`(historical 7 day context) and data from `local_retriever_tool`(current-day context) and MUST use the 'market context briefing' from the `enhanced_search_agent` to enrich its summary.**
    Then, instruct it to use the `local_retriever_tool` to search for other relevant news articles published on the same day. The final summary must integrate the main article's content with the context from the graph, the market briefing, and any related same-day news it finds, providing a truly holistic overview.
    - **MANDATORY DELAY:** Before Start this step, you **MUST** instruct it to call the `delay_tool` with `seconds=90`. This is a critical step for rate limit management.

4.  **Delegation for Comprehensive Analysis:** Delegate the analysis task to the `Analysis_Worker_Agent`. Instruct it to provide a comprehensive yet accessible analysis of the financial impact and resulting trends.
    **To perform its analysis, it MUST Use the context report (historical 7 day) from `graph_retriever` + current-day news (from `local_retriever_tool`) and MUST integrate insights from the 'market context briefing' (from `enhanced_search_agent`) with the main news article.**
    Then, instruct it to also use the `local_retriever_tool` to find related financial news, market trends, or competitor announcements from the same day. The final analysis should explain how the main news item, when viewed alongside the graph context, market briefing, and other events of the day, impacts financial trends and market sentiment.
    - Identify all key financial implications (both positive and negative).
    - Consider potential short-term and long-term effects on the company's market position and stock value.
    - Be written in clear, professional language, ensuring the insights are easy to understand and can be utilized for strategic decision-making.
    - **MANDATORY DELAY:** Before Start this step, you **MUST** instruct it to call the `delay_tool` with `seconds=90`. This is a critical step for rate limit management.

5. Final Output Generation:
    After collecting the outputs from `graph_retriever`,`enhanced_search_agent`, `Summary_Worker_Agent`, and `Analysis_Worker_Agent`, your goal is to compose a clear, narrative-style report — not JSON. 
    The final output must be written in easy word and natural language, suitable for a decision-maker who needs to quickly grasp the situation and its implications.
    
    Structure the final response as follows:
    - Title: Summary Report of Financial News ({today})

    - **Summary Paragraph** A well-structured paragraph that integrates the summary from `Summary_Worker_Agent`, the analysis from `Analysis_Worker_Agent`, insights and relationships from `graph_retriever`, and the high-impact market context from `enhanced_search_agent`. This paragraph should clearly highlight:
        - Main events of the news article
        - Financial implications
        - Key relationships between entities
        - Relevant external market events or competitor news that affect the interpretation of the main news
      Ensure the paragraph reads naturally and can be used for quick leader-level decision making.
    - **Key Insight (Reasoned Narrative)**
        Write one short paragraph (4–5 sentences) that **connects the past, present, and future**. 
        Focus on:
        - Start with what **the historical data (7 days ago)** from `graph_retriever` revealed.  
        - Then describe what is **happening now (today)** based on the current news and retrieved context.  
        - End with what is **likely to happen next (future outlook)** based on ** trends, risks and opportunities** revealed by the news and relationships.
        - A single actionable insight that captures the broader significance in the financial or strategic context
        This should show clear cause–effect reasoning.
    - **Bullet List of Implications** A structured list that clearly presents:
        - Key relationships between entities extracted by `graph_retriever`
        - Risks and opportunities derived from the main news and additional context
        - Notable market and competitor events identified by `enhanced_search_agent`
      Each bullet should be concise but specific, allowing a reader to quickly grasp critical insights. 
    - Ensure all outputs properly reference the sources of insights:
        - Graph relationships are marked as from `graph_retriever`
        - Market and competitor insights are marked as from `enhanced_search_agent`
      This ensures transparency of context and credibility of decision-making information.

Example Output:
### Summary Report of Financial News (26/01/2025)

### Summary Paragraph
Summary Paragraph

### Key Insight
Key Insight

### Key Implications
Bullet List of Implications
"""

trace_name = f"Leader3_{today}"
@observe()
def process_request(query):
    # Add to the current trace
    langfuse.update_current_trace(session_id="3", name=trace_name)
    return leader.run(query)
response = process_request(query)

with open('summary_leader3.md', 'w', encoding='utf-8') as f:
    f.write(response)