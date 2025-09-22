```mermaid
Get News -> Filter Irrelevant News -> Get Contents -> Clean Content -> Save in one big file -> Query Tools
```

# Details in each step
- Get News: 2 providers only get you a URL and a brief description
- Filter Irrelevant News: We get all news about the NVIDIA, INTEL, AMD which sometime the laptop advertisement gonna be involved. So get rid of the news doesn't talk about their business
- Get Contents: from step one, we use "newspaper3k" and "lxml_html_clean" to scrape the news content from URL
- Clean Content: The news we got still something like the advertise banners, text on button so we have to remove it
- Save in one big file: the file gonna separate 3 big index NVIDIA, INTEL, AMD for store every thing have got in there
- Query tools: For later use, just run command to get the news e.g. Get only NVIDIA news, Get all news from 3 company that within a week


## Steps inside each component
### **1. Get News**
### [News API org](https://newsapi.org)

- Endpoint: https://newsapi.org/v2/everything
- query
    - date range
    - searchIn: title, description, content
    - language
    - date range
    - page: (default/max 100)
    - domain
- Authorize
    - "apiKey" in querystring parameter
    - "X-Api-Key" HTTP header

- response
    - status
    - totalResults -> page to go through
    - article [array]
        - source.name
        - author (, delimiter)
        - title
        - description
        - url
        - publishedAt

### **2. Filter news**
- Not all articles with NVIDIA/Intel/AMD keywords are relevant (e.g., laptop ads, hardware specs in consumer blogs).
- filter out: "laptop", "gaming", "specs", "deal", "discount", "sale", "shopping"
- Keep: "market", "stock", "revenue", "earnings"

### **3. Get and Cleaing Content**
Use `newspaper3k` and `lxml_html_clean` to get news content from given URL

A rule-based filter
- HTML cleanup
- regex

### **4. Data storage format**
before save check dupplicating of news

```json
{
    "NVIDIA": [
        {
            "headline": "",
            "content": "",
            "source": "",
            "url": "",
            "tiimestamp": ""
        }
    ],
    "AMD": [],
    "Intel": []
}
```

### **5. Query Tool**

```
python query_news.py --company NVIDIA --since "2025-09-15" --until "2025-09-22"
```

Argument
- `--company`:
    - NVIDIA
    - INTEL
    - AMD
    - all/ALL (default if not specify)
- `--since`:
    - no (data within 7 days)
    - oldest (get the oldest data)
    - `YYYY-MM-DD` format
- `--until`:
    - no (default is today)
    - `YYYY-MM-DD` format

Output

file nomenclature: date_time_[company-included]_[since]_[to]

```json
{
    "NVIDIA": [

    ]
}
```