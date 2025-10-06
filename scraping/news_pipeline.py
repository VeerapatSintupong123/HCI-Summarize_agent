#!/usr/bin/env python3
"""
Hybrid News Processing Pipeline

Supports both file-based and object-based data passing:
- File-based: Traditional checkpoint approach with fault tolerance
- Object-based: Fast in-memory processing
- Hybrid: Combines both for optimal performance and reliability
"""

import requests
import json
import os
import time
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from newspaper import Article
from tqdm import tqdm
import re

class NewsAPIScraper:
    def __init__(self, api_key: str = 'a5e0c9cbf67f45e3980f7e9723cffb90'):
        self.api_key = api_key
        self.companies = ["NVIDIA", "AMD", "Intel"]  # Keep API names for scraping
    
    def scrape_all_companies(self, days: int = 7) -> Dict[str, List[Dict]]:
        print("🚀 Starting news scraping...")
        company_articles = {company: [] for company in self.companies}
        
        for i in range(1, days + 1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            print(f"📅 {date}")
            
            for company in self.companies:
                url = f"https://newsapi.org/v2/everything?q={company}&from={date}&to={date}&language=en&apiKey={self.api_key}&pageSize=20"
                
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        articles = [{
                            "source": a.get('source', {}).get('name', ''),
                            "headline": a.get('title', ''),
                            "description": a.get('description', ''),
                            "url": a.get('url', ''),
                            "timestamp": a.get('publishedAt', ''),
                        } for a in response.json().get('articles', [])]
                        
                        company_articles[company].extend(articles)
                        print(f"  {company}: {len(articles)} articles")
                    else:
                        print(f"  ❌ {company}: API error {response.status_code}")
                except Exception as e:
                    print(f"  ❌ {company}: {str(e)}")
                
                time.sleep(0.5)
        
        return company_articles

class NewsFilter:
    def __init__(self):
        self.keep_keywords = ['investment', 'revenue', 'earnings', 'stock', 'financial', 'ceo', 'billion', 'ai chip', 'business']
        self.filter_out = ['gaming', 'deal', 'fps', 'benchmark', 'price', 'buy', 'sale', 'discount']
    
    def filter_company_data(self, company_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        filtered_data = {}
        
        for company, articles in company_data.items():
            original_count = len(articles)
            filtered_articles = []
            
            for article in articles:
                text = f"{article.get('headline', '')} {article.get('description', '')}".lower()
                
                # Keep business articles
                if any(keyword in text for keyword in self.keep_keywords):
                    filtered_articles.append(article)
                # Filter out consumer/gaming articles
                elif not any(keyword in text for keyword in self.filter_out):
                    filtered_articles.append(article)
            
            filtered_data[company] = filtered_articles
            print(f"🔍 {company}: {original_count} -> {len(filtered_articles)} articles")
        
        return filtered_data

class ContentFetcher:
    def __init__(self):
        ssl._create_default_https_context = ssl._create_unverified_context
    
    def enrich_articles(self, company_data: Dict[str, List[Dict]], max_per_company: int = None) -> Dict[str, List[Dict]]:
        enriched_data = {}
        
        for company, articles in company_data.items():
            print(f"📄 {company}: fetching content...")
            articles_to_process = articles[:max_per_company] if max_per_company else articles
            
            for i, article in enumerate(tqdm(articles_to_process), 1):
                if not article.get('content'):
                    try:
                        news_article = Article(article['url'])
                        news_article.download()
                        news_article.parse()
                        article['content'] = news_article.text or None
                    except:
                        article['content'] = None

                if hasattr(article, 'description'):
                    article.delattr('description', None)  # Remove description
                
                if i % 10 == 0:
                    time.sleep(1)
                else:
                    time.sleep(0.3)
            
            enriched_data[company] = articles_to_process
            print(f"✅ {company}: {len(articles_to_process)} articles processed")
        
        return enriched_data

class NewsPipeline:
    def __init__(self, data_dir: str = None):
        # Auto-detect data directory based on current location
        if data_dir is None:
            # Find the HCI-Summarize_agent directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up until we find the project root (contains data directory)
            while current_dir and not os.path.exists(os.path.join(current_dir, "data")):
                parent = os.path.dirname(current_dir)
                if parent == current_dir:  # Reached filesystem root
                    break
                current_dir = parent
            data_dir = os.path.join(current_dir, "data")
        
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.filtered_dir = os.path.join(data_dir, "filtered")
        self.final_dir = os.path.join(data_dir, "final")
        self.dataset_file = os.path.join(data_dir, "dataset.json")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.filtered_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        
        # Initialize components
        self.scraper = NewsAPIScraper()
        self.filter = NewsFilter()
        self.content_fetcher = ContentFetcher()
    
    def transform_to_dataset_format(self, company_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Transform data to match the 19092025.json format"""
        formatted_data = {}
        
        company_name_map = {
            "NVIDIA": "Nvidia",
            "AMD": "AMD", 
            "Intel": "Intel"
        }
        
        for company, articles in company_data.items():
            formatted_company_name = company_name_map.get(company, company)
            formatted_articles = []
            
            for article in articles:
                # Parse timestamp to extract just the date
                timestamp = article.get('timestamp', '')
                if timestamp:
                    try:
                        # Parse the timestamp and extract just the date
                        if 'T' in timestamp:  # ISO format like "2025-09-18T22:23:04Z"
                            date_part = timestamp.split('T')[0]
                        else:
                            date_part = timestamp[:10]  # Take first 10 characters (YYYY-MM-DD)
                    except:
                        date_part = timestamp
                else:
                    date_part = ''
                
                # Only keep the required fields in the correct format
                formatted_article = {
                    "headline": article.get('headline', ''),
                    "content": article.get('content', ''),
                    "source": article.get('source', ''),
                    "url": article.get('url', ''),
                    "timestamp": date_part
                }
                formatted_articles.append(formatted_article)
            
            formatted_data[formatted_company_name] = formatted_articles
        
        return formatted_data
    
    def save_checkpoint(self, stage: str, data: Dict, timestamp: str = None, transform_format: bool = False) -> str:
        """Save data checkpoint"""
        if not timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d')
        
        # Choose directory based on stage
        if stage == "raw":
            target_dir = self.raw_dir
        elif stage == "filtered":
            target_dir = self.filtered_dir
        elif stage == "final":
            target_dir = self.final_dir
        else:
            target_dir = self.raw_dir  # fallback
        
        filename = f"{stage}_{timestamp}.json"
        filepath = os.path.join(target_dir, filename)
        
        # Transform data to dataset format if requested
        data_to_save = self.transform_to_dataset_format(data) if transform_format else data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Checkpoint saved: {filepath}")
        return filepath
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load data from checkpoint"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_hybrid_pipeline(self, days: int = 7, max_content_per_company: int = 50) -> str:
        """Hybrid approach: fast operations in memory, slow operations with checkpoints"""
        print("⚡ Running HYBRID pipeline...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Stage 1: Scraping (fast, but save checkpoint)
        print("\n=== Stage 1: Scraping ===")
        raw_data = self.scraper.scrape_all_companies(days)
        self.save_checkpoint("raw", raw_data, timestamp)
        
        # Stage 2: Filtering (fast, in-memory)
        print("\n=== Stage 2: Filtering (in-memory) ===")
        filtered_data = self.filter.filter_company_data(raw_data)
        
        # Stage 3: Content Fetching (slow, with progress checkpoints)
        print("\n=== Stage 3: Content Fetching (with checkpoints) ===")
        filtered_file = self.save_checkpoint("filtered", filtered_data, timestamp)
        enriched_data = self.content_fetcher.enrich_articles(filtered_data, max_content_per_company)
        final_file = self.save_checkpoint("final", enriched_data, timestamp, transform_format=True)
        
        print(f"\n⚡ Hybrid pipeline complete! Final data: {final_file}")
        return final_file
    
    def resume_from_stage(self, stage: str, filepath: str, max_content_per_company: int = 50) -> str:
        """Resume pipeline from a specific stage"""
        print(f"🔄 Resuming from {stage} stage...")
        data = self.load_checkpoint(filepath)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if stage == "raw":
            # Continue with filtering and content fetching
            filtered_data = self.filter.filter_company_data(data)
            self.save_checkpoint("filtered", filtered_data, timestamp)
            enriched_data = self.content_fetcher.enrich_articles(filtered_data, max_content_per_company)
            return self.save_checkpoint("final", enriched_data, timestamp, transform_format=True)
        
        elif stage == "filtered":
            # Continue with content fetching
            enriched_data = self.content_fetcher.enrich_articles(data, max_content_per_company)
            return self.save_checkpoint("final", enriched_data, timestamp, transform_format=True)
        
        else:
            print("❌ Invalid stage. Use 'raw' or 'filtered'")
            return None
    
    def update_dataset(self, data_source: Union[str, Dict]):
        """Update master dataset with new data"""
        if isinstance(data_source, str):
            # Load from file
            new_data = self.load_checkpoint(data_source)
        else:
            # Use provided data
            new_data = data_source
        
        # Transform data to the correct format
        new_data = self.transform_to_dataset_format(new_data)
        
        # Load existing dataset or create new one
        if os.path.exists(self.dataset_file):
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        else:
            # Initialize with empty arrays for each company
            dataset = {"Nvidia": [], "AMD": [], "Intel": []}
        
        # Update dataset
        total_added = 0
        for company, articles in new_data.items():
            if company not in dataset:
                dataset[company] = []
            
            existing_urls = {article['url'] for article in dataset[company] if article.get('url')}
            added_count = 0
            
            for article in articles:
                if article.get('url') and article['url'] not in existing_urls:
                    dataset[company].append(article)
                    existing_urls.add(article['url'])
                    added_count += 1
            
            total_added += added_count
            print(f"📊 {company}: +{added_count} new articles")
        
        # Save updated dataset
        with open(self.dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset updated! Total new articles: {total_added}")

def main():
    """Example usage of the hybrid pipeline"""
    pipeline = NewsPipeline()
    
    print("🌟 News Processing Pipeline")
    print("=" * 50)
    print("1. News Scraping and Processing")
    print("2. Resume from checkpoint")

    days = int(input("Enter number of days to scrape (default 30): ").strip() or 30)
    max_content = int(input("Enter max content per company (default 50): ").strip() or 50)
    
    choice = input("\nSelect mode (1-2): ").strip()
    
    if choice == "1":
        result = pipeline.run_hybrid_pipeline(days=days, max_content_per_company=max_content)
        pipeline.update_dataset(result)
    
    elif choice == "2":
        stage = input("Resume from stage (raw/filtered): ")
        filepath = input("Checkpoint file path: ")
        result = pipeline.resume_from_stage(stage, filepath, max_content_per_company=max_content)
        if result:
            pipeline.update_dataset(result)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()