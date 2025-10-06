from smolagents import tool, ToolCallingAgent, OpenAIServerModel
import networkx as nx
import json
from dotenv import load_dotenv
import os

with open(r"graph_news\23092025_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.MultiDiGraph()

for chipmaker in data:
    for d in data[chipmaker]:
        date = d['date']
        for triplet in d['triplets']:
            subject = triplet['subject']
            relation = triplet['relation']
            object_ = triplet['object']

            verb = relation.get("verb")
            detail = relation.get("detail")

            G.add_edge(
                subject,
                object_,
                key=verb,
                verb=verb,
                detail=detail,
                date=date,
                chipmaker=chipmaker
            )

print("✅ Graph loaded with nodes and edges.")

@tool
def get_7day_summary(chipmaker: str) -> str:
    """
    return a 7-day summary of news for the given chipmaker

    Args:
        chipmaker (str): the name of the chipmaker, e.g., "Nvidia", "AMD", "Intel"
    """
    dict_file = {"AMD":"amd_7day.md", "INTEL":"intel_7day.md", "NVIDIA":"nvidia_7day.md"}
    with open(r"graph_news\\" + dict_file[chipmaker.upper()], "r", encoding="utf-8") as f:
        summary = f.read()
        return summary

    return ""

@tool
def get_across_summary() -> str:
    """
    return a summary of news across all chipmakers in the past 7 days

    Args:
        None
    """
    with open(r"graph_news\summary.md", mode="r", encoding="utf-8") as f:
        summary = f.read()
        return summary

@tool
def get_entities_from_chipmaker(chipmaker: str) -> list[str]:
    """
    return a list of entities related to the given chipmaker

    Args:
        chipmaker (str): the name of the chipmaker, e.g., "Nvidia", "AMD", "Intel"
    """
    nodes = set()
    for u, v, data in G.edges(data=True):
        if data.get("chipmaker", "").lower() == chipmaker.lower():
            nodes.add(u)
            nodes.add(v)
    return list(nodes)

@tool
def get_relations_from_chipmaker(chipmaker: str) -> list[str]:
    """
    return a list of relations related to the given chipmaker

    Args:
        chipmaker (str): the name of the chipmaker, e.g., "Nvidia", "AMD", "Intel"
    """
    relations = set()
    for _, _, data in G.edges(data=True):
        if data.get("chipmaker", "").lower() == chipmaker.lower():
            relations.add(data.get("verb", ""))
    return list(relations)

@tool
def get_relations_between_entities(chipmaker: str, entity1: str, entity2: str) -> list[str]:
    """
    return a list of relations between two entities related to the given chipmaker

    Args:
        chipmaker (str): the name of the chipmaker, e.g., "Nvidia", "AMD", "Intel"
        entity1 (str): the first entity
        entity2 (str): the second entity
    """
    relations = []
    for u, v, data in G.edges(data=True):
        if data.get("chipmaker", "").lower() == chipmaker.lower():
            if (u == entity1 and v == entity2) or (u == entity2 and v == entity1):
                verb = data.get("verb", "")
                detail = data.get("detail", "")
                date = data.get("date", "")
                relations.append(f"{verb} ({detail}, {date})")
    return relations


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

model = OpenAIServerModel(
    model_id="gemini-2.5-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

graph_retriever = ToolCallingAgent(
    model=model,
    tools=[get_7day_summary, 
           get_across_summary, 
           get_entities_from_chipmaker, 
           get_relations_from_chipmaker, 
           get_relations_between_entities
    ],
    name="graph_retriever",
    description="Handles graph queries with graph retrieval tools.",
    stream_outputs=False,
)

print("✅ Graph Retriever Agent initialized.")

# graph_retriever.run("Give me a 7-day summary of news for Nvidia. List 5 entities related to Nvidia. List 5 relations related to Nvidia. What are the relations between Nvidia and ARM?")