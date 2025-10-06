import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, InferenceClientModel, tool, OpenAIServerModel
from chunk_news.vector_db import get_retriever
from datetime import datetime
import time

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_WORKER_MODEL_ID = os.getenv("HF_WORKER_MODEL_ID", "gemini-2.5-flash")

model = OpenAIServerModel(
    model_id=HF_WORKER_MODEL_ID,
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

retriever_instance = get_retriever()
print("✅ Retriever instance initialized.")

@tool
def local_retriever_tool(queries: list[str]) -> str:
    """
    Performs multiple rounds of retrieval, each with a different query.
    
    Args:
        queries (list[str]): A list of queries for each retrieval round.
        top_k (int): Number of top results per round.

    Returns:
        str: Combined text of retrieved document chunks from all rounds.
    """
    
    combined_docs = []
    
    for i, query in enumerate(queries):
        print(f"  🔹 Retrieval round {i+1}: query='{query}'")
        docs = retriever_instance.invoke(query)
        combined_docs.extend(d.page_content for d in docs)
    
    seen = set()
    unique_docs = []
    for doc in combined_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)
    print(f"-> Total unique chunks retrieved: {len(unique_docs)}")
    return context

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

@tool
def delay_tool(seconds: int) -> str:
    """
    Pauses the execution for a specified number of seconds.
    Use this tool to manage rate limits when instructed by the orchestrator.

    Args:
        seconds (int): The number of seconds to wait.

    Returns:
        str: A confirmation message that the delay has completed.
    """
    print(f"-> Tool 'delay_tool' called. Waiting for {seconds} seconds...")
    time.sleep(seconds)
    print("-> Delay finished.")
    return f"Successfully delayed for {seconds} seconds."

summary_worker_agent = ToolCallingAgent(
    model=model,
    tools=[local_retriever_tool, delay_tool],
    name="Summary_Worker_Agent",
    description=(
        "You are a Summary Worker Agent. "
        "Your role is to carefully read financial news articles, including the headlines, "
        "and gather additional context using the local_retriever_tool for related news of the same day. "
        "You then create a clear and concise summary that captures the key facts, events, and context, "
        "so that the leader agent can make informed decisions quickly. "
        "Focus on presenting the news in a structured, easy-to-understand way, highlighting the most important points."
    ),
    stream_outputs=False
)
print("✅ Summary Worker Agent initialized.")

analysis_worker_agent = ToolCallingAgent(
    model=model,
    tools=[local_retriever_tool, delay_tool],
    name="Analysis_Worker_Agent",
    description=(
        "You are an Analysis Worker Agent. "
        "Your role is to read the provided news article and any additional context retrieved via local_retriever_tool. "
        "You analyze the financial impact, market trends, and strategic implications of the news. "
        "Provide a professional yet accessible analysis that identifies key risks, opportunities, "
        "short-term and long-term effects on the company, competitors, and market sentiment. "
        "Your insights help the leader agent make strategic decisions, so focus on clarity and actionable takeaways."
    ),
    stream_outputs=False
)
print("✅ Analysis Worker Agent initialized.")