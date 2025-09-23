# worker.py
import os
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel
from chunk_news.vector_db import get_retriever

# API specific imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# --- Initialization ---

# Load environment variables
load_dotenv()

# Initialize HF model via token
# Make sure your .env file has HUGGINGFACEHUB_API_TOKEN set
model = InferenceClientModel(
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Initialize your agent with no external tools
agent = CodeAgent(tools=[], model=model)

# Load your vector DB retriever
retriever = get_retriever()

# --- API Setup ---

# Initialize FastAPI app
app = FastAPI(
    title="Worker Agent API",
    description="An API that takes a guide and returns a summary based on a local vector database.",
    version="1.0.0"
)

# Define the request model for input validation
class TaskRequest(BaseModel):
    guide: str

# --- Core Logic ---

def handle_task(guide: str) -> str:
    """
    Handles a task by retrieving relevant context from the local vector database
    and generating a summary based on it.

    Args:
        guide: The initial guide text from the leader (used as the search query).

    Returns:
        A summary of the findings.
    """
    print(f"-> Handling task: '{guide}'")

    # 1. Retrieve relevant documents directly using the guide as the query.
    print("-> Retrieving relevant documents from the vector database...")
    docs = retriever.invoke(guide)
    
    # 2. Create a context string from the retrieved documents.
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    print(f"-> Found {len(docs)} relevant document chunks.")

    # 3. Create a single, focused prompt for the agent.
    summary_prompt = f"""
    Based *only* on the following context, please provide a concise summary that addresses the user's guide.

    User Guide: "{guide}"

    Context from documents:
    ---
    {context}
    ---

    Summary:
    """

    # 4. Run the agent once with the complete context to get the final summary.
    print("-> Generating summary...")
    summary = agent.run(summary_prompt).strip()
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
    print("--- Starting Worker API Server ---")
    # To run this, you will need to install uvicorn and fastapi:
    # pip install "fastapi[all]"
    uvicorn.run(app, host="0.0.0.0", port=8001)

