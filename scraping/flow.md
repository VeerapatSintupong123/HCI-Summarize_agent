# 🔹 Guideline for Collecting Financial News (for Model Training)

## 1. Sources to Include (Financially Reliable)

Since your focus is **financial/business news** (not consumer tech reviews), prioritize **established financial and tech-industry outlets**:

### Recommended Sources

* **Financial & Market Outlets**

  * Bloomberg
  * Reuters
  * Financial Times (FT)
  * Wall Street Journal (WSJ)
  * CNBC
  * MarketWatch
  * Yahoo Finance

* **Tech-Focused but Business-Oriented**

  * TechCrunch (business coverage, funding, partnerships)
  * The Verge (business/industry sections)
  * Wired (business/AI/semiconductors sections)
  * VentureBeat (AI & business news)

* **Other Aggregated Sources**

  * Seeking Alpha (finance/investor analysis)
  * Business Insider (when filtered for finance/markets)

👉 In **NewsAPI**, you can use `domains` parameter to whitelist these sources so you avoid random blogs.
Example:

```text
domains=bloomberg.com,reuters.com,ft.com,cnbc.com,marketwatch.com
```

---

## 2. Filtering Strategy (Laptop Ads, Specs, etc.)

* **Keywords to filter out**:
  `"laptop"`, `"notebook"`, `"gaming"`, `"specs"`, `"configuration"`, `"deal"`, `"discount"`, `"sale"`, `"shopping"`

* **Keywords to keep (business context)**:
  `"stock"`, `"market"`, `"earnings"`, `"quarterly"`, `"partnership"`, `"revenue"`, `"supply chain"`, `"AI"`, `"chip production"`, `"export ban"`, `"semiconductor"`

---

## 3. Cleaning Content

> You asked: would using **NLP for cleaning** be overengineered?

* ✅ **For initial pipeline** → Yes, it’s overkill. A **rule-based filter** (keywords + HTML cleanup + regex) should be enough.
* 🔮 Later → If you want to scale or improve accuracy, you can fine-tune a **binary classifier** (business-related vs. product ad/news) with something lightweight like `scikit-learn` or `transformers`.

**Recommendation:**

* Start **rule-based**.
* Collect enough training samples.
* If misclassification is high, move to **NLP classifier**.

---

## 4. Data Storage Format

You want:

```json
{
    "NVIDIA": [
        {
            "headline": "",
            "content": "",
            "source": "",
            "url": "",
            "timestamp": ""
        }
    ],
    "INTEL": [
        ...
    ],
    "AMD": [
        ...
    ]
}
```

### Notes

* JSON with **company as root key** makes querying simpler.
* Each list item = one cleaned article.
* `timestamp` should always be UTC (`YYYY-MM-DDTHH:MM:SSZ`) for consistency.

---

## 5. Query Tool

Instead of external tools (jq, SQL), just build a **Python query script**:

### Example Usage

```bash
python query_news.py --company NVIDIA --since "2025-09-15" --until "2025-09-22"
```

### Example Output

```json
[
  {
    "headline": "NVIDIA Expands AI Chip Production",
    "content": "NVIDIA announced new AI hardware production...",
    "source": "Reuters",
    "url": "https://reuters.com/example",
    "timestamp": "2025-09-20T10:00:00Z"
  }
]
```

---

## 6. Suggested File/Script Structure

```
news_pipeline/
│
├── get_news.py        # Fetch from API
├── filter_news.py     # Apply whitelist + blacklist
├── scrape_content.py  # Use newspaper3k + lxml_html_clean
├── clean_content.py   # Remove ads, normalize text
├── save_dataset.py    # Save to JSON format (your schema)
├── query_news.py      # Query script
│
├── data/
│   ├── raw/           # Raw API responses
│   ├── scraped/       # Scraped + cleaned content
│   └── dataset.json   # Final structured dataset
│
└── requirements.txt
```