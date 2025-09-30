from smolagents import OpenAIServerModel, tool, ToolCallingAgent
from ddgs import DDGS
from dotenv import load_dotenv
import os, json

# --- Load env ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Internet Search Tool ---
@tool
def internet_search(query: str, max_results: int = 20) -> str:
    """
    Perform an internet search for financial or company-related information 
    and return structured results in JSON format.

    Args:
        query (str): A keyword or phrase to search for.
            Examples:
                - "Tesla Q3 2025 earnings report"
                - "Amazon stock price today"
                - "Federal Reserve interest rate hike September 2025"
        max_results (int): Number of search results to return (default = 5).

    Returns:
        str: A JSON array of search results. Each object contains:
            - title (str): The title of the webpage.
            - link (str): The URL of the source.
            - snippet (str): A short summary or preview text.

    Usage:
        - Use this tool when up-to-date external information is required.
        - Focus on financial data, company statements, and economic indicators.
        - Always return structured JSON so that other agents can parse it.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        
        if not results:
            return json.dumps([{"error": f"No search results found for query: {query}"}], indent=2)
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps([{"error": f"Search failed: {str(e)}"}], indent=2)

# --- Model ---
model = OpenAIServerModel(
    model_id="gemini-2.5-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

# --- Internet Search Agent ---
search_agent = ToolCallingAgent(
    model=model,
    tools=[internet_search],
    name="Search_Agent",
    description=(
        "Search_Agent retrieves financial and company-related information from the internet. "
        "It is used to understand keywords, retrieve stock prices, related stock movements, "
        "or official company statements to support financial news summarization. "
        "It always returns results in JSON format (title, link, snippet)."
    ),
    stream_outputs=False,
)

# --- Run standalone test ---
if __name__ == "__main__":
    query = "H200"
    max_results_to_test = [5, 10, 20]  # Different max_results values to compare

    search_agent.run(f"Search for: {query} with default max_results")

    # answers = []

    # for max_res in max_results_to_test:
    #     print(f"\n{'='*20} Testing with max_results={max_res} {'='*20}")
        
    #     # Create a new agent instance for each run to avoid state issues
    #     agent = ToolCallingAgent(
    #         model=model,
    #         tools=[internet_search],
    #         name=f"Search_Agent_max{max_res}",
    #         description=(
    #             "Search_Agent retrieves financial and company-related information from the internet. "
    #             "It is used to understand keywords, retrieve stock prices, related stock movements, "
    #             "or official company statements to support financial news summarization. "
    #             "It always returns results in JSON format (title, link, snippet)."
    #         ),
    #         stream_outputs=False,
    #     )
        
    #     result = agent.run(f"Search for: {query} with max_results={max_res}")
    #     answers.append((max_res, result))
    
    # for max_res, res in answers:
    #     print(f"\n{'='*15} Final Answer (max_results={max_res}) {'='*15}")
    #     print(res)
    #     print("="*60 + "\n")