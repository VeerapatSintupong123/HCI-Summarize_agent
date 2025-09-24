import requests
import json 
from datetime import datetime, timedelta
import os
import math

class NewsAPIScraper:
    def __init__(self):
        self.api_key = '7eb571b97f8440118c46dc8c74279e0e'
        self.base_url = "https://newsapi.org/v2/everything"
API_KEY = '7eb571b97f8440118c46dc8c74279e0e'
QUERY = ["NVIDIA", "AMD", "Intel"]

def get_date_range(days=7):
    dates = []
    for i in range(1, days + 1):  # Start from 1 to exclude today
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        dates.append(date)
    return dates

def fetch_news_for_single_day(api_key, query, target_date, max_articles=20):
    """
    Fetch news for a specific date with limited articles
    """
    if not is_valid_date(target_date):
        raise ValueError("Date must be in YYYY-MM-DD format")

    # Use the same date for both from and to for single day
    url = f"https://newsapi.org/v2/everything?q={query}&from={target_date}&to={target_date}&language=en&apiKey={api_key}&pageSize={max_articles}&page=1"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        total_articles = data.get('totalResults', 0)
        articles = data.get('articles', [])
        
        # Limit to max_articles
        limited_articles = articles[:max_articles]
        
        return {
            "articles": limited_articles,
            "total_available": total_articles,
            "fetched_count": len(limited_articles),
            "date": target_date,
            "query": query
        }
    else:
        return {"error": f"Failed to fetch news: {response.text}"}

def is_valid_date(date_string):
    """
    Checks if a string can be parsed into a date according to a specific format.
    """
    try:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                datetime.strptime(date_string, fmt)
                return True
            except ValueError:
                continue
        return False
    except ValueError:
        return False

def main():
    print("Starting day-by-day news scraping...")
    print(f"Target dates: {date_list}")
    print(f"Companies: {', '.join(QUERY)}")
    print(f"Max articles per company per day: 20")
    print("-" * 50)
    
    # Create directory if it doesn't exist
    if not os.path.exists(r'../data/raw/news_api'):
        os.makedirs(r'../data/raw/news_api')
    
    # Dictionary to store articles for each company
    company_articles = {company: [] for company in QUERY}
    
    for target_date in date_list:
        print(f"\n📅 Scraping news for {target_date}")
        
        for company in QUERY:
            print(f"  🔍 Fetching {company} news...")
            
            # Fetch news for this specific day and company
            result = fetch_news_for_single_day(API_KEY, company, target_date, max_articles=20)
            
            if "error" in result:
                print(f"  ❌ Error fetching {company} news: {result['error']}")
                continue
            
            articles = result["articles"]
            total_available = result["total_available"]
            fetched_count = result["fetched_count"]
            
            print(f"  📊 {company}: {fetched_count} articles fetched (out of {total_available} available)")
            
            if not articles:
                print(f"  ⚠️  No articles found for {company} on {target_date}")
                continue
            
            # Map articles to your desired format and add to company collection
            for article in articles:
                source = article.get('source', {}).get('name', '')
                author = article.get('author', '')
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')
                published_at = article.get('publishedAt', '')

                arc = {
                    "source": source,
                    "author": author,
                    "headline": title,
                    "description": description,
                    "url": url,
                    "timestamp": published_at,
                    "scraped_date": target_date  # Add which date this was scraped from
                }
                company_articles[company].append(arc)
            
            # Small delay to be respectful to the API
            import time
            time.sleep(0.5)
    
    # Save one file per company with all articles from 7 days
    # Get today's date for filename
    scrape_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"raw_{scrape_date}.json"
    path = os.path.join(r'../data/raw/news_api', file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(company_articles, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Saved to: {file_name}")
    
    print(f"\n🎉 Scraping completed!")
    print(f"📍 Files saved in: ../data/raw/news_api/")
    
    # Summary
    total_articles = sum(len(articles) for articles in company_articles.values())
    print(f"📈 Total articles collected: {total_articles}")
    for company in QUERY:
        print(f"  • {company}: {len(company_articles[company])} articles")

if __name__ == "__main__":
    date_list = get_date_range(7)

    print("🚀 Starting News Scraping Process")
    print("=" * 50)
    print(f"Date range: {date_list[0]} to {date_list[-1]} (7 days)")
    print(f"Companies: {', '.join(QUERY)}")
    print(f"Max articles per company per day: 20")
    print("=" * 50)

    confirm = input("Proceed with scraping? (y/n): ")
    if confirm.lower() == 'y':
        main()
    else:
        print("Scraping cancelled.")