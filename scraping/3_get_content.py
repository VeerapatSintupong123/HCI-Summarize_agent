from newspaper import Article
import json
import os
import time
import ssl
from datetime import datetime
from tqdm import tqdm

class ContentFetcher:
    def __init__(self, file=None):
        # To handle SSL certificate issues
        ssl._create_default_https_context = ssl._create_unverified_context
        self.PATH_NEWS = r'../data/raw/news_api'
        self.PATH_SAVE_NEWS = r'../data/scraped/news_api'
        self.PATH_DATASET = r'../data'

        os.makedirs(self.PATH_SAVE_NEWS, exist_ok=True)
        os.makedirs(self.PATH_NEWS, exist_ok=True)

        if file is None or os.path.exists(file) is False:
            raise ValueError("Please provide a JSON file with articles to process.")

        self.companies_articles = self.load_article_json(file)

    def load_article_json(self, file):
        with open(file, 'rb') as f:
            data = json.load(f)

        return data

    def load_content_article(self, url, retries=3):
        """Load article content with simple retry for SSL errors"""
        for attempt in range(retries + 1):
            try:
                article = Article(url)
                article.download()
                article.parse()
                return article.text
            except Exception as e:
                error_str = str(e)
                # Only retry for SSL errors, skip others permanently
                if attempt < retries and "SSL" in error_str:
                    print(f"SSL error, retrying in {10 * (attempt + 1)}s... ({attempt + 1}/{retries + 1})")
                    time.sleep(10 * (attempt + 1))  # Exponential backoff
                    continue
                elif "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "404" in error_str or "403" in error_str:
                    # These are permanent failures - don't retry and return None to indicate skip
                    print(f"⚠️  Skipping URL (permanent failure): {error_str}")
                    return None
                else:
                    raise e
        return None  # If all retries failed
                
    def run(self):
        output_file = os.path.join(self.PATH_SAVE_NEWS, f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        failed_fetching_file = os.path.join(self.PATH_SAVE_NEWS, f"failed_fetching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        if os.path.exists(output_file):
            output_file = output_file.replace(".json", "_v2.json")

        self.failed_fetching = []
        
        # Initialize the structured output format
        self.structured_data = {}
        for company in self.companies_articles.keys():
            self.structured_data[company] = []

        for company, articles in self.companies_articles.items():
            print(f"\nCompany: {company} - Total Articles: {len(articles)}")
            for i, article in tqdm(enumerate(articles), total=len(articles)):
                if 'content' not in article or not article['content']:
                    try:
                        content = self.load_content_article(article['url'])
                        if content is None:
                            # Skip this article entirely (permanent failure)
                            print(f"⏭️  Skipping article: {article.get('url', 'No URL')}")
                            continue
                        article['content'] = content
                    except Exception as e:
                        print(f"✗ Failed: {e}")
                        article['content'] = article.get('description', None)
                        self.failed_fetching.append(article)
                        continue

                # Structure the article data
                structured_article = {
                    "headline": article.get('headline', ''),
                    "content": article.get('content', ''),
                    "source": article.get('source', {}).get('name', '') if isinstance(article.get('source'), dict) else str(article.get('source', '')),
                    "url": article.get('url', ''),
                    "timestamp": article.get('timestamp', '')
                }
                
                # Add to the company's list
                self.structured_data[company].append(structured_article)

                # Save every 10 articles or for testing (i == 3)
                if (i + 1) % 10 == 0 or i == 3:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(self.structured_data, f, ensure_ascii=False, indent=4)
                
                if i == 3:  # For testing - remove this condition when ready for full run
                    break

                time.sleep(2)  # Be nice to servers
        
        # Final save with all accumulated data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.structured_data, f, ensure_ascii=False, indent=4)

        with open(failed_fetching_file, "w", encoding="utf-8") as f:
            json.dump(self.failed_fetching, f, ensure_ascii=False, indent=4)

        print(f"\n✅ Successfully saved {sum(len(articles) for articles in self.structured_data.values())} articles to {output_file}")
        print(f"❌ Failed to fetch {len(self.failed_fetching)} articles")

    def collect_in_dataset(self):
        with open(os.path.join(self.PATH_DATASET, "dataset.json"), "rb") as f:
            dataset = json.load(f)

        

        for company in self.structured_data.keys():
            if company not in dataset:
                dataset[company] = []
        
        for company, articles in self.structured_data.items():
            added_count = 0
            existing_articles = dataset[company]
            existing_urls = {article['url'] for article in existing_articles}

            for article in articles:
                if article['url'] not in existing_urls:
                    existing_articles.append(article)
                    existing_urls.add(article['url'])
                    added_count += 1

            print(f"\n✅ Added {added_count} new articles to the {company} dataset.")

        with open(os.path.join(self.PATH_DATASET, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    fetcher = ContentFetcher(file=r'../data/raw/news_api/raw_20250924_075304.json')
    fetcher.run()
    fetcher.collect_in_dataset()