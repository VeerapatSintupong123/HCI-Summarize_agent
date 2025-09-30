import json
import re
from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# LangChain and HuggingFace Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# This is the new, recommended import path for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
# The name of the folder where the vector database is stored.
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER")
# The HuggingFace model used for creating text embeddings.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    """
    Removes URLs and collapses excess whitespace from a string.
    Handles None or empty inputs gracefully.

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string.
    """
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Reduce multiple whitespace characters to a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove newlines
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def load_news(path: Path) -> list[str]:
    """
    Loads news articles from a JSON file, cleans them, and formats them into
    a list of strings.

    Args:
        path: The Path object pointing to the JSON file.

    Returns:
        A list of formatted strings, where each string is a news document.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for category, items in data.items():
        for item in items:
            content = clean_text(item.get("content", ""))
            full_text = (
                f"Category: {category}\n"
                f"Headline: {item.get('headline','')}\n"
                f"Source: {item.get('source','')}\n"
                f"Content: {content}\n"
                f"Timestamp: {item.get('timestamp','')}"
            )
            docs.append(full_text)
    return docs


def build_vector_db(docs: list[str], save_path: Path):
    """
    Builds a FAISS vector database from a list of documents and saves it locally.

    Args:
        docs: A list of documents (strings) to process.
        save_path: The directory path where the vector database will be saved.
    """
    print("Building vector database...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )
    
    # CORRECTED LINE: Use create_documents for lists of strings.
    # This method correctly handles turning your raw text into Document objects before splitting.
    chunks = splitter.create_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    vectordb.save_local(str(save_path))
    print(f"✅ Vector DB saved successfully to: {save_path}")


def load_vector(path: Path):
    """
    Loads an existing FAISS vector database from a local path.

    Args:
        path: The directory path where the vector database is stored.

    Returns:
        The loaded FAISS database object.
    """
    print(f"Loading vector database from: {path}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # The allow_dangerous_deserialization flag is needed for FAISS with pickle
    db = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded successfully.")
    return db


def get_retriever():
    """
    Constructs the path to the vector DB, loads it, and returns a retriever.
    This is the primary function to be imported by other scripts like worker.py.
    
    Returns:
        A LangChain retriever object.
    """
    # Get the directory where this current file (vector_db.py) is located
    current_dir = Path(__file__).parent
    
    # Build the full, absolute path to the vector database folder
    db_path = current_dir / VECTOR_DB_FOLDER
    
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_path}. "
            f"Please run the build script first."
        )

    db = load_vector(db_path)
    
    # Configure the database as a retriever to find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# --- Main execution block ---
# This part of the script will only run when you execute `python vector_db.py` directly.
# It's useful for building the database for the first time.
if __name__ == "__main__":
    print("--- Running vector_db.py as a standalone script ---")
    
    # Define the path to the source JSON relative to this script's location
    # Assumes 'scrape_news' and 'chunk_news' are sibling directories
    current_dir = Path(__file__).parent
    json_path = current_dir.parent / "scrape_news" / os.getenv("TODAY_NEWS_FILENAME")
    save_path = current_dir / VECTOR_DB_FOLDER

    if not json_path.exists():
        print(f"❌ Error: JSON file not found at {json_path}")
    else:
        # 1. Load the news documents
        news_docs = load_news(json_path)
        print(f"✅ Loaded {len(news_docs)} news items.")
        
        # 2. Build and save the vector database
        build_vector_db(news_docs, save_path)
        
        # 3. Test the retriever
        print("\n--- Testing the retriever ---")
        retriever = get_retriever()
        query = "Nvidia acquisition"
        results = retriever.invoke(query)
        print(f"🔍 Query: '{query}'")
        print("✅ Retriever test successful. Found results.")
        # print("Top result content snippet:")
        print(results[0].page_content[:200] + "...")

