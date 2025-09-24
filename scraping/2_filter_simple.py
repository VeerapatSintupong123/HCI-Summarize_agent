#!/usr/bin/env python3
"""
Simple news filtering system to remove irrelevant articles.
Filters out consumer/gaming content, keeps business/financial news.
"""

import json
import re
from typing import Dict, List

class NewsFilter:
    def __init__(self):
        # Keywords to filter OUT (consumer/gaming content)
        self.filter_out_keywords = [
            'laptop', 'gaming', 'specs', 'deal', 'discount', 'sale', 'shopping',
            'fps', 'frame rate', 'benchmark', 'game', 'gamer', 'graphics card',
            'gpu review', 'cpu review', 'rtx', 'geforce', 'unboxing', 'hands-on'
        ]
        
        # Keywords to KEEP (business/financial content)
        self.keep_keywords = [
            'investment', 'revenue', 'earnings', 'profit', 'stock', 'market',
            'partnership', 'acquisition', 'merger', 'quarterly', 'financial',
            'ceo', 'executive', 'billion', 'million', 'datacenter', 'ai chip'
        ]
        
        # Regex patterns for deals/gaming
        self.deal_pattern = re.compile(r'\$\d+.*off|save.*\$|\d+% off', re.IGNORECASE)
        self.gaming_pattern = re.compile(r'\d+fps|frame.*rate|benchmark', re.IGNORECASE)
    
    def is_business_relevant(self, article: Dict) -> bool:
        """Check if article is business-relevant (True = keep, False = filter out)"""
        text = f"{article.get('headline', '')} {article.get('description', '')}".lower()
        
        # Strong business indicators - always keep
        if any(keyword in text for keyword in self.keep_keywords):
            return True
        
        # Deal/gaming patterns - filter out
        if self.deal_pattern.search(text) or self.gaming_pattern.search(text):
            return False
            
        # Consumer keywords - filter out
        if any(keyword in text for keyword in self.filter_out_keywords):
            return False
        
        # Default: keep if nothing matches filter-out criteria
        return True
    
    def filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter a list of articles, return only business-relevant ones"""
        return [article for article in articles if self.is_business_relevant(article)]
    
    def filter_news_data(self, data: Dict) -> Dict:
        """Filter news data for all companies"""
        filtered_data = {}
        
        for company, articles in data.items():
            if not isinstance(articles, list):
                filtered_data[company] = articles
                continue
                
            original_count = len(articles)
            filtered_articles = self.filter_articles(articles)
            filtered_count = len(filtered_articles)
            
            filtered_data[company] = filtered_articles
            print(f"{company}: {original_count} -> {filtered_count} articles")
        
        return filtered_data
    
    def load_and_filter(self, input_file: str, output_file: str):
        """Load, filter, and save news data"""
        print(f"Loading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Filtering articles...")
        filtered_data = self.filter_news_data(data)
        
        print(f"Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print("Done!")

def main():
    """Main function"""
    filter = NewsFilter()
    
    # Adjust these paths as needed
    input_file = '../data/raw/news_api/raw_20250924_075304.json'
    output_file = '../data/raw/news_api/filtered_news_api.json'
    
    filter.load_and_filter(input_file, output_file)

if __name__ == "__main__":
    main()