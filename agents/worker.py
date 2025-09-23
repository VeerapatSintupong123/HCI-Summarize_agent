import os
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, tool
from chunk_news.vector_db import get_retriever
from datetime import datetime

# API specific imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Tool specific imports
from tavily import TavilyClient


# --- Initialization ---

# Load environment variables
load_dotenv()

# Initialize HF model via token
model = InferenceClientModel(
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Initialize clients for our tools
retriever = get_retriever()
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


# --- Tool Definitions ---

@tool
def local_retriever_tool(query: str) -> str:
    """
    Searches the LOCAL news vector database (from 2025) to find historical context.
    Use this for information known before September 2025.
    Returns the combined text of the most relevant document chunks.

    Args:
        query (str): The search query to find relevant news articles.
    """
    print(f"-> Tool 'local_retriever_tool' called with query: '{query}'")
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    print(f"-> Found {len(docs)} relevant document chunks.")
    return context

@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web for real-time, up-to-date information.
    Use this for recent events, current stock prices, or information not found in the local database.
    Returns a concise answer from the web.

    Args:
        query (str): The search query for the web search.
    """
    print(f"-> Tool 'web_search_tool' called with query: '{query}'")
    response = tavily_client.search(query=query, search_depth="basic")
    # It's safer to check if results exist before accessing them
    if response and response.get('results'):
        return response['results'][0]['content']
    return "No results found from web search."

@tool
def get_current_date_tool() -> str:
    """
    Returns the current date.
    Use this to understand the timeliness of information.
    """
    print("-> Tool 'get_current_date_tool' called.")
    return datetime.now().strftime("%Y-%m-%d")

# Note: The smolagents CodeAgent can do math natively, so a specific calculator tool is not needed
# unless you require complex scientific calculations.


# --- Agent Setup ---

# Initialize your agent with a full suite of tools
all_tools = [local_retriever_tool, web_search_tool, get_current_date_tool]
agent = CodeAgent(tools=all_tools, model=model)


# --- API Setup ---

# Initialize FastAPI app
app = FastAPI(
    title="Worker Agent API",
    description="An API that takes a guide and returns a summary based on a local vector database and web search.",
    version="2.0.0" # Version bump!
)

# Define the request model for input validation
class TaskRequest(BaseModel):
    guide: str

# --- Core Logic ---

def handle_task(guide: str) -> str:
    """
    Handles a task by giving a high-level guide to an agent, which can then
    use its suite of tools to find information and generate a summary.

    Args:
        guide: The initial guide text from the leader.

    Returns:
        A summary of the findings.
    """
    print(f"-> Handling task with Agent: '{guide}'")

    # This prompt encourages the agent to think and use the best tool for the job.
    agent_prompt = f"""
    You are a financial research assistant. Your goal is to provide a concise summary that addresses the user's guide.
    
    You have access to the following tools:
    1. `local_retriever_tool`: For historical data from your 2025 news archive.
    2. `web_search_tool`: For the most current, real-time information.
    3. `get_current_date_tool`: To check today's date.

    Think step-by-step. First, check the current date to understand the context.
    Then, decide if the query is best answered with historical data (local_retriever_tool) or if it requires fresh, up-to-the-minute information (web_search_tool).
    You may use multiple tools if needed. Synthesize the information you find into a final answer.

    User Guide: "{guide}"
    """
    
    summary = agent.run(agent_prompt).strip()
    return summary

# --- API Endpoint ---

@app.post("/handle-task/", response_model=dict)
def handle_task_endpoint(request: TaskRequest):
    """
    API endpoint to process a guide and return a summary.
    """
    summary = handle_task(request.guide)
    return {"summary": summary}

# --- Main Execution Block ---

if __name__ == "__main__":
    # This block now starts the API server
    print("--- Starting Worker API Server with Tools ---")
    uvicorn.run(app, host="0.0.0.0", port=8001)


