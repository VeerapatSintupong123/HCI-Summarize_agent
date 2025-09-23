import json
import os
import requests
from dotenv import dotenv_values

# --- 🛠️ Gemini API Setup ---
# Load API keys from .env file
# โหลดค่าคอนฟิกจากไฟล์ .env
config = dotenv_values(".env")
GEMINI_API_KEY = config.get("GEMINI_API_KEY")

# --- การแก้ไข: แก้ไข URL ของ API Endpoint ให้ถูกต้อง ---
# ผมได้เปลี่ยนจาก generativeai.googleapis.com เป็น generativelanguage.googleapis.com
# ซึ่งเป็น endpoint ที่ถูกต้องสำหรับ Gemini API ครับ
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def get_gemini_response(prompt: str) -> str:
    """Sends a prompt to the Gemini API and returns the text response."""
    if not GEMINI_API_KEY:
        error_message = "Error: GEMINI_API_KEY not found. Please create a .env file and add your API key."
        print(error_message)
        return ""

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(GEMINI_API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        
        # เพิ่มการตรวจสอบเพื่อความปลอดภัยว่ามี candidate และ part ที่ต้องการอยู่จริง
        if 'candidates' in result and result['candidates'][0]['content']['parts'][0]['text']:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print("Error: Invalid response format from Gemini API.")
            print("Raw response:", result)
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        # พิมพ์รายละเอียดของ response หากมีข้อผิดพลาดจากฝั่งเซิร์ฟเวอร์
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        return ""

# --- 🧠 Ground Truth Generation (Using Gemini as the 'leader') ---
def generate_ground_truth(headline: str, content: str, guide: str):
    """Generates ground truth summary and impact using the Gemini API."""
    worker_summary_prompt = f"""
    Based on the following news headline and content, write a concise financial news summary and a financial analysis.

    Headline: "{headline}"
    Content: "{content}"

    Your output MUST be a valid JSON object with two keys:
    - "worker_summary": A summary of the news.
    - "impact_trend": A concise analysis of the financial impact (e.g., "Positive," "Negative," or "Neutral") and the reasoning.
    """
    
    try:
        response_text = get_gemini_response(worker_summary_prompt)
        if not response_text:
            return None
        
        # Attempt to parse the response as JSON. Gemini might add markdown, so we clean it.
        # พยายามทำความสะอาด response ที่อาจมี markdown ปนมา
        cleaned_response = response_text.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from Gemini response: {e}")
        print(f"Raw response received: '{response_text}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_ground_truth: {e}")
        return None

# --- 📝 Ground Truth Data (Initial Set) ---
# สร้าง Ground Truth เมื่อสคริปต์เริ่มทำงาน
print("Generating ground truth from Gemini API...")
GROUND_TRUTH = {
    "If you're looking for a gaming laptop on a budget, this is it":
        generate_ground_truth(
            "If you're looking for a gaming laptop on a budget, this is it",
            "Save $290 off retail for a limited time",
            "NVIDIA AMD INTEL revenue profit EPS margins growth rates quarterly yearly earnings reports guidance mergers product launches financial news"
        )
}
print("Ground truth generated successfully.")

# --- 📊 Quantitative Evaluation ---
def evaluate_quantitative(results):
    """
    Evaluates worker summaries and impact trends quantitatively.
    This function uses ROUGE for summarization and a simple string comparison for the impact trend.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    except ImportError:
        print("Please install the rouge-score library: pip install rouge-score")
        return []

    scores = []
    for r in results:
        headline = r["headline"]
        worker_summary = r["worker_summary"]
        worker_impact = r["impact_trend"]
        
        # Get ground truth from the pre-generated set
        gold_data = GROUND_TRUTH.get(headline)
        if not gold_data:
            print(f"Warning: No ground truth found for headline: '{headline}'")
            continue

        gold_summary = gold_data.get("worker_summary", "")
        gold_impact_raw = gold_data.get("impact_trend", "") # อ่านค่าดิบที่อาจเป็น dict
        
        # --- FIX for AttributeError ---
        # ข้อผิดพลาดเกิดจาก Gemini API อาจ trả lời JSON object (dict ใน Python)
        # สำหรับฟิลด์ "impact_trend" แทนที่จะเป็น string ธรรมดา
        # โค้ดส่วนนี้จะตรวจสอบประเภทของ gold_impact_raw ก่อน ถ้าเป็น dictionary
        # จะถูกแปลงเป็น string เพื่อป้องกันการเรียกใช้เมธอด .strip() บน dict
        gold_impact = ""
        if isinstance(gold_impact_raw, str):
            gold_impact = gold_impact_raw
        elif isinstance(gold_impact_raw, dict):
            # หากเป็น dictionary ให้รวมค่า value ทั้งหมดเพื่อสร้างเป็น string เดียว
            # นี่เป็นวิธีที่ยืดหยุ่นในการจัดการโครงสร้างที่ไม่คาดคิดจาก LLM
            gold_impact = " - ".join(str(v) for v in gold_impact_raw.values())

        # Calculate ROUGE score for the summary
        rouge_scores = scorer.score(gold_summary, worker_summary) # gold standard is the first argument
        
        # Calculate a simple binary score for the impact trend
        # ตอนนี้ gold_impact เป็น string ที่ปลอดภัยแล้ว และ .strip() จะทำงานได้
        impact_match_score = 1.0 if worker_impact.strip().lower() == gold_impact.strip().lower() else 0.0

        scores.append({
            "headline": headline,
            "rouge_score": {
                "rouge1": rouge_scores['rouge1'].fmeasure,
                "rouge2": rouge_scores['rouge2'].fmeasure,
                "rougeL": rouge_scores['rougeL'].fmeasure
            },
            "impact_match_score": impact_match_score,
            "worker_summary": worker_summary,
            "gold_summary": gold_summary,
            "worker_impact": worker_impact,
            "gold_impact": gold_impact
        })
    return scores

# --- 🔎 Qualitative Evaluation ---
def evaluate_qualitative(results, n=3):
    """
    Performs a qualitative inspection by printing a sample of the results.
    """
    print("\n=== Qualitative Evaluation (Sample outputs) ===")
    for r in results[:n]:
        headline = r['headline']
        worker_summary = r['worker_summary']
        worker_impact = r['impact_trend']
        
        gold_data = GROUND_TRUTH.get(headline)
        if not gold_data:
            continue
            
        gold_summary = gold_data.get("worker_summary", "N/A")
        gold_impact = gold_data.get("impact_trend", "N/A")
        
        print(f"\nHeadline: {headline}")
        print("-" * (len(headline) + 10))
        print(f"Predicted Summary: {worker_summary}")
        print(f"Gold Standard Summary: {gold_summary}")
        print(f"Predicted Impact: {worker_impact}")
        print(f"Gold Standard Impact: {gold_impact}")
    print("\n")

# --- 🚀 Main Execution ---
if __name__ == "__main__":
    # This is a sample result from your worker process.
    worker_results = [
        {
            "headline": "If you're looking for a gaming laptop on a budget, this is it",
            "content": "Save $290 off retail for a limited time",
            "guide": "NVIDIA AMD INTEL revenue profit EPS margins growth rates quarterly yearly earnings reports guidance mergers product launches financial news",
            # เนื้อหาส่วนนี้มาจาก Worker ของคุณ ซึ่งเราจะนำมาเปรียบเทียบกับ Ground Truth ที่สร้างโดย Gemini
            "worker_summary": "A gaming laptop is available at a discounted price, offering a saving of $290 for a limited period. This deal targets budget-conscious gamers.",
            "impact_trend": "Neutral - This is a product-specific discount and does not directly reflect the financial performance of major chip manufacturers like NVIDIA, AMD, or Intel."
        }
    ]

    # ตรวจสอบว่า Ground Truth ถูกสร้างสำเร็จหรือไม่ก่อนดำเนินการต่อ
    if GROUND_TRUTH.get("If you're looking for a gaming laptop on a budget, this is it"):
        # Run quantitative evaluation
        q_scores = evaluate_quantitative(worker_results)
        print("\n=== Quantitative Evaluation Results ===")
        print(json.dumps(q_scores, indent=2, ensure_ascii=False))

        # Run qualitative evaluation
        evaluate_qualitative(worker_results)
    else:
        print("\nCould not perform evaluation because ground truth generation failed.")

