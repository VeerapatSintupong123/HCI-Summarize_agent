import json
import os
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AI_Evaluator")

# ---------------------------
# Environment & API Setup
# ---------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in .env file or environment variables.")

# Use environment variables to define which models to use, with defaults
ground_truth_model_name = os.environ.get("GROUND_TRUTH_MODEL", "gemini-2.5-flash")
evaluation_model_name = os.environ.get("EVALUATION_MODEL", "gemini-2.5-flash")

logger.info(f"Using Ground Truth Model: {ground_truth_model_name}")
logger.info(f"Using Evaluation Model: {evaluation_model_name}")

try:
    # Model for evaluation (e.g., scoring, justification)
    client = genai.GenerativeModel(model_name=evaluation_model_name)
    # Model for generating ground truth (can be a faster model)
    gt_model = genai.GenerativeModel(model_name=ground_truth_model_name)
except Exception as e:
    raise RuntimeError(f"❌ Error configuring GenAI Client: {e}")

# ---------------------------
# Load Data
# ---------------------------
current_dir = Path(__file__).parent
result_path = current_dir.parent / "results" / "ex2_result.json"

if not result_path.exists():
    raise FileNotFoundError(f"❌ File not found: {result_path}")

with open(result_path, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# ---------------------------
# Load Models
# ---------------------------
logger.info("Loading SentenceTransformer model...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge_evaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
logger.info("✅ Models loaded successfully.")

# ---------------------------
# Ground Truth Cache
# ---------------------------
cache_path = current_dir / "ground_truth_cache.json"
if cache_path.exists():
    with open(cache_path, "r", encoding="utf-8") as f:
        ground_truth_cache = json.load(f)
else:
    ground_truth_cache = {}


def save_cache():
    """Saves the cache to a file."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_cache, f, ensure_ascii=False, indent=2)


# ---------------------------
# Helper Functions
# ---------------------------

def _extract_text_from_response(response) -> str:
    """Helper to safely extract text from Gemini API response."""
    try:
        return response.text.strip()
    except Exception:
        # Fallback for older response structures if needed
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return ""


def generate_ground_truth(headline: str, content: str) -> Tuple[str, str]:
    """Generates or loads Ground Truth from cache."""
    if headline in ground_truth_cache:
        logger.info(f"🗂️ Using cached Ground Truth for: {headline}")
        gt = ground_truth_cache[headline]
        return gt["summary"], gt["impact"]

    logger.info(f"⚡ Generating new Ground Truth for: {headline}")

    summary_prompt = f"Summarize the following news article clearly and concisely.\n\nHeadline: {headline}\nContent: {content}"
    impact_prompt = f"Analyze the short-term and long-term impacts of the following news.\n\nHeadline: {headline}\nContent: {content}"

    try:
        summary_resp = gt_model.generate_content(summary_prompt)
        impact_resp = gt_model.generate_content(impact_prompt)

        summary = _extract_text_from_response(summary_resp)
        impact = _extract_text_from_response(impact_resp)

        ground_truth_cache[headline] = {"summary": summary, "impact": impact}
        save_cache()

        return summary, impact
    except Exception as e:
        logger.error(f"Error during Ground Truth generation: {e}")
        return "Error", "Error"


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculates ROUGE scores for a prediction against a reference."""
    scores = rouge_evaluator.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def calculate_semantic_similarity(prediction: str, reference: str) -> float:
    """Calculates cosine similarity between two text strings."""
    embeddings = semantic_model.encode([prediction, reference])
    return round(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item(), 4)

# --- REFACTORED FUNCTIONS ---

def evaluate_summary(
    agent_summary: str, gt_summary: str, original_content: str
) -> Dict[str, Dict[str, str]]:
    """
    Uses an LLM to evaluate the 'summary' output.
    """
    logger.info("🤖 Getting AI evaluation for Summary...")

    evaluation_prompt = f"""
You are an expert AI evaluator. Your task is to score the AI agent's summary based on the provided criteria (1-5 scale).
Compare the agent's summary to the ground truth summary and the original content.

**Original Content:**
{original_content}
---
**Ground Truth Summary:**
{gt_summary}
---
**Agent's Summary:**
{agent_summary}
---
Output ONLY JSON with this schema:
{{
  "summary_clarity": {{"score": 5, "justification": "The summary is extremely clear and easy to understand."}},
  "summary_completeness": {{"score": 4, "justification": "The summary covers most of the key points but missed one minor detail."}},
  "summary_accuracy": {{"score": 5, "justification": "The summary accurately reflects the information in the original content without any distortion."}}
}}
"""
    try:
        response = client.generate_content(evaluation_prompt)
        text_output = _extract_text_from_response(response)
        
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        json_text = match.group(0) if match else text_output
        
        return json.loads(json_text)
        
    except Exception as e:
        logger.error(f"Error during Summary AI evaluation or JSON parsing: {e}")
        return {
            "summary_clarity": {"score": 0, "justification": "Evaluation failed"},
            "summary_completeness": {"score": 0, "justification": "Evaluation failed"},
            "summary_accuracy": {"score": 0, "justification": "Evaluation failed"},
        }

def evaluate_impact(
    agent_impact: str, gt_impact: str, original_content: str
) -> Dict[str, Dict[str, str]]:
    """
    Uses an LLM to evaluate the 'impact analysis' output.
    """
    logger.info("🤖 Getting AI evaluation for Impact Analysis...")

    evaluation_prompt = f"""
You are an expert AI evaluator. Your task is to score the AI agent's impact analysis based on the provided criteria (1-5 scale).
Compare the agent's impact analysis to the ground truth and consider its relevance to the original content.

**Original Content:**
{original_content}
---
**Ground Truth Impact Analysis:**
{gt_impact}
---
**Agent's Impact Analysis:**
{agent_impact}
---
Output ONLY JSON with this schema:
{{
  "impact_relevance": {{"score": 5, "justification": "The analysis is highly relevant to the news." }},
  "impact_depth": {{"score": 3, "justification": "The analysis is somewhat superficial and could explore deeper consequences." }},
  "impact_coherence": {{"score": 4, "justification": "The arguments are logical and well-structured." }},
  "impact_completeness": {{"score": 5, "justification": "The analysis covers both short-term and long-term impacts comprehensively." }}
}}
"""
    try:
        response = client.generate_content(evaluation_prompt)
        text_output = _extract_text_from_response(response)
        
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        json_text = match.group(0) if match else text_output

        return json.loads(json_text)
        
    except Exception as e:
        logger.error(f"Error during Impact AI evaluation or JSON parsing: {e}")
        return {
            "impact_relevance": {"score": 0, "justification": "Evaluation failed"},
            "impact_depth": {"score": 0, "justification": "Evaluation failed"},
            "impact_coherence": {"score": 0, "justification": "Evaluation failed"},
            "impact_completeness": {"score": 0, "justification": "Evaluation failed"},
        }


# ---------------------------
# Main Process
# ---------------------------
def main():
    all_results: List[Dict] = []

    for i, item in enumerate(news_data, start=1):
        logger.info(f"\n{'='*20} Evaluating News Item #{i} {'='*20}")
        logger.info(f"HEADLINE: {item['headline']}")

        summary_gt, impact_gt = generate_ground_truth(item["headline"], item["content"])

        rouge_results = calculate_rouge_scores(item["worker_summary"], summary_gt)
        semantic_sim_summary = calculate_semantic_similarity(item["worker_summary"], summary_gt)
        semantic_sim_impact = calculate_semantic_similarity(item["impact_trend"], impact_gt)
        
        # --- UPDATED SECTION: Call evaluation functions separately ---
        summary_scores = evaluate_summary(
            agent_summary=item["worker_summary"],
            gt_summary=summary_gt,
            original_content=item["content"]
        )
        
        impact_scores = evaluate_impact(
            agent_impact=item["impact_trend"],
            gt_impact=impact_gt,
            original_content=item["content"]
        )
        
        # Combine results from the two separate evaluations
        ai_scores = {**summary_scores, **impact_scores}
        # --- END OF UPDATED SECTION ---

        flat_scores = {f"{key}_score": val.get("score", 0) for key, val in ai_scores.items()}
        flat_justifications = {f"{key}_justification": val.get("justification", "N/A") for key, val in ai_scores.items()}

        all_results.append(
            {
                "headline": item["headline"],
                **rouge_results,
                "summary_semantic_similarity": semantic_sim_summary,
                "impact_semantic_similarity": semantic_sim_impact,
                **flat_scores,
                **flat_justifications,
            }
        )

    results_df = pd.DataFrame(all_results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    logger.info("\n📊 FINAL AUTOMATED EVALUATION RESULTS 📊\n" + "=" * 50)
    score_cols = [c for c in results_df.columns if any(x in c for x in ["score", "similarity", "rouge", "headline"])]
    print(results_df[score_cols].to_string())

    try:
        out_path = current_dir / "ai_evaluation_results.csv"
        results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Results saved to {out_path}")
    except Exception as e:
        logger.error(f"❌ Could not save CSV: {e}")


if __name__ == "__main__":
    main()