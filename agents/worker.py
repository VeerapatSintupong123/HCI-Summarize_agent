# worker.py
from smolagents import CodeAgent, InferenceClientModel
from chunk_news.vector_db import get_retriever
from dotenv import load_dotenv
import os

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
    #    CORRECTED: Use the .invoke() method for retriever objects.
    print("-> Retrieving relevant documents from the vector database...")
    docs = retriever.invoke(guide)
    
    # 2. Create a context string from the retrieved documents.
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    print(f"-> Found {len(docs)} relevant document chunks.")

    # 3. Create a single, focused prompt for the agent.
    #    This prompt explicitly tells the agent to use ONLY the provided context.
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


if __name__ == "__main__":
    guide = "Check the financial implications of new AI regulations in Europe"
    result = handle_task(guide)
    print("\n--- Worker Summary ---\n")
    print(result)
