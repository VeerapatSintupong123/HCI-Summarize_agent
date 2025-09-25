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

# --- UPDATED SECTION: Load model names from .env ---
# Load model names from environment variables with default fallbacks
ground_truth_model = os.environ.get("GROUND_TRUTH_MODEL", "gemini-1.5-flash")
evaluation_model = os.environ.get("EVALUATION_MODEL", "gemini-1.5-pro-latest")

logger.info(f"Using Ground Truth Model: {ground_truth_model}")
logger.info(f"Using Evaluation Model: {evaluation_model}")
# --- END OF UPDATED SECTION ---

try:
    # The client is configured once, and model names are passed in each call
    client = genai.Client(api_key=api_key)
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
    """บันทึก cache ลงไฟล์"""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_cache, f, ensure_ascii=False, indent=2)


# ---------------------------
# Helper Functions
# ---------------------------

def _extract_text_from_response(response) -> str:
    """Helper to safely extract text from Gemini API response."""
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        return getattr(response, "text", "").strip()


def generate_ground_truth(headline: str, content: str) -> Tuple[str, str]:
    """สร้างหรือโหลด Ground Truth จาก cache"""
    if headline in ground_truth_cache:
        logger.info(f"🗂️ Using cached Ground Truth for: {headline}")
        gt = ground_truth_cache[headline]
        return gt["summary"], gt["impact"]

    logger.info(f"⚡ Generating new Ground Truth for: {headline}")

    summary_prompt = f"Summarize the following news article clearly and concisely.\n\nHeadline: {headline}\nContent: {content}"
    impact_prompt = f"Analyze the short-term and long-term impacts of the following news.\n\nHeadline: {headline}\nContent: {content}"

    try:
        # --- UPDATED SECTION: Use model variable ---
        summary_resp = client.models.generate_content(
            model=ground_truth_model,
            contents=summary_prompt
        )
        impact_resp = client.models.generate_content(
            model=ground_truth_model,
            contents=impact_prompt
        )
        # --- END OF UPDATED SECTION ---

        summary = _extract_text_from_response(summary_resp)
        impact = _extract_text_from_response(impact_resp)

        # เก็บลง cache
        ground_truth_cache[headline] = {"summary": summary, "impact": impact}
        save_cache()

        return summary, impact
    except Exception as e:
        logger.error(f"Error during Ground Truth generation: {e}")
        return "Error", "Error"


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    scores = rouge_evaluator.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def calculate_semantic_similarity(prediction: str, reference: str) -> float:
    embeddings = semantic_model.encode([prediction, reference])
    return round(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item(), 4)


def get_ai_evaluation_scores(
    agent_summary: str,
    gt_summary: str,
    agent_impact: str,
    gt_impact: str,
    original_content: str,
) -> Dict[str, Dict[str, str]]:
    """ใช้ LLM เพื่อทำการประเมินผลลัพธ์"""
    logger.info("🤖 Getting evaluation scores from AI Judge...")

    evaluation_prompt = f"""
You are an expert AI evaluator. Your task is to score the output of another AI agent based on criteria (1-5 scale).
Compare the agent's output to the ground truth and original content.

**Original Content:**
{original_content}
---
**Ground Truth Summary:**
{gt_summary}
---
**Agent's Summary:**
{agent_summary}
---
**Ground Truth Impact Analysis:**
{gt_impact}
---
**Agent's Impact Analysis:**
{agent_impact}
---
Output ONLY JSON with this schema:
{{
  "summary_clarity": {{"score": 5, "justification": "..." }},
  "summary_completeness": {{"score": 4, "justification": "..." }},
  "summary_accuracy": {{"score": 5, "justification": "..." }},
  "impact_relevance": {{"score": 5, "justification": "..." }},
  "impact_depth": {{"score": 3, "justification": "..." }},
  "impact_coherence": {{"score": 4, "justification": "..." }},
  "impact_completeness": {{"score": 5, "justification": "..." }}
}}
"""

    try:
        # --- UPDATED SECTION: Use model variable ---
        response = client.models.generate_content(model=evaluation_model, contents=evaluation_prompt)
        # --- END OF UPDATED SECTION ---
        
        text_output = _extract_text_from_response(response)

        # Try to extract JSON
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        json_text = match.group(0) if match else text_output

        return json.loads(json_text)

    except Exception as e:
        logger.error(f"Error during AI evaluation or JSON parsing: {e}")
        return {
            key: {"score": 0, "justification": "Evaluation failed"}
            for key in [
                "summary_clarity",
                "summary_completeness",
                "summary_accuracy",
                "impact_relevance",
                "impact_depth",
                "impact_coherence",
                "impact_completeness",
            ]
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

        ai_scores = get_ai_evaluation_scores(
            item["worker_summary"], summary_gt, item["impact_trend"], impact_gt, item["content"]
        )

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