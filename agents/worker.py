import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, InferenceClientModel, tool, OpenAIServerModel
from chunk_news.vector_db import get_retriever
from datetime import datetime
from tavily import TavilyClient

# --- Load env ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_WORKER_MODEL_ID = os.getenv("HF_WORKER_MODEL_ID", "gemini-2.5-flash")
# --- Model ---
model = OpenAIServerModel(
    model_id=HF_WORKER_MODEL_ID,
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)


# --- Tools ---
retriever = get_retriever()
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
def local_retriever_tool(query: str) -> str:
    """
    Searches the LOCAL news vector database (from 2025) to find historical context.
    Use this for information known before October 2025.
    Returns the combined text of the most relevant document chunks.

    Args:
        query (str): The search query string used to retrieve relevant document chunks.

    Returns:
        str: Combined text of the most relevant document chunks.
    """
    print(f"-> Tool 'local_retriever_tool' called with query: '{query}'")
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    print(f"-> Found {len(docs)} relevant document chunks.")
    return context


# @tool
# def web_search_tool(query: str) -> str:
#     """
#     Searches the web for real-time, up-to-date information.
#     Use this for recent events, current stock prices, or information not found in the local database.
#     Returns a concise answer from the web.

#     Args:
#         query (str): The search query string for the web search.

#     Returns:
#         str: A concise answer from the web search results.
#     """
#     print(f"-> Tool 'web_search_tool' called with query: '{query}'")
#     response = tavily_client.search(query=query, search_depth="basic")
#     if response and response.get('results'):
#         return response['results'][0]['content']
#     return "No results found from web search."


@tool
def get_current_date_tool() -> str:
    """
    Returns the current date.
    Use this to understand the timeliness of information.

    Returns:
        str: Today's date in YYYY-MM-DD format.
    """
    print("-> Tool 'get_current_date_tool' called.")
    return datetime.now().strftime("%Y-%m-%d")

# --- Summary Worker Agent ---
summary_worker_agent = ToolCallingAgent(
    model=model,
    tools=[local_retriever_tool, get_current_date_tool],
    name="Summary_Worker_Agent",
    description="Finds financial news using retriever and date tool",
    stream_outputs=False,
)

# --- Analysis Worker Agent ---
analysis_worker_agent = ToolCallingAgent(
    model=model,
    tools=[local_retriever_tool, get_current_date_tool],
    name="Analysis_Worker_Agent",
    description="Analyze financial impact trends from financial news using retriever and date tool",
    stream_outputs=False,
)

# --- (Optional) Standalone Run ---
if __name__ == "__main__":
    query = "Summarize financial impact for NVIDIA Q2 earnings"
    result = agent.run(query)
    print("\nWorker Result:\n", result)
