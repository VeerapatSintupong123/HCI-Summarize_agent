#!/usr/bin/env python3
"""
News filtering system to remove irrelevant articles that are not about 
company business or financial activities.

This system filters out:
- Consumer product reviews and specifications
- Gaming hardware and laptop deals
- Shopping deals and discounts
- Product announcements unrelated to business strategy

This system keeps:
- Business partnerships and deals
- Financial reports and earnings
- Market analysis and stock news
- Strategic business decisions
- Investment and funding news
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FilterMode(Enum):
    """Filtering modes with different aggressiveness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

@dataclass
class FilterResult:
    """Result of filtering process with metadata."""
    is_relevant: bool
    confidence_score: float
    matched_keywords: List[str]
    filter_reasons: List[str]

@dataclass
class FilterStats:
    """Statistics from filtering process."""
    original_count: int
    filtered_count: int
    removed_count: int
    percentage_kept: float

class NewsFilter:
    """
    A configurable news filtering system for business-relevant articles.
    
    Limitations addressed by this class:
    - Configurable filtering modes (strict/moderate/lenient)
    - Confidence scoring for articles
    - Extensible keyword management
    - Better error handling and logging
    - Separation of concerns
    """
    
    def __init__(self, mode: FilterMode = FilterMode.MODERATE):
        """Initialize the news filter with specified mode."""
        self.mode = mode
        self._init_keywords()
        self._init_patterns()
        
    def _init_keywords(self):
        """Initialize keyword lists based on filtering mode."""
        # Base filter-out keywords
        self.filter_out_keywords = [
            # Consumer/shopping related
            'laptop', 'gaming', 'specs', 'deal', 'discount', 'sale', 'shopping',
            'consumer', 'product review', 'benchmark', 'price', 'buy', 'purchase',
            'save', 'offer', 'promotion', 'black friday', 'cyber monday',
            
            # Gaming specific
            'fps', 'frame rate', 'performance test', 'game', 'gamer', 'esports',
            'handheld', 'steam deck', 'rog ally', 'gaming pc', 'gaming laptop',
            
            # Consumer hardware
            'rtx 5080', 'rtx 5060', 'geforce', 'graphics card', 'gpu review',
            'cpu review', 'motherboard', 'ram', 'storage', 'ssd', 'cooling',
            
            # Deal sites and retailers
            'slickdeals', 'amazon', 'walmart', 'best buy', 'newegg', 'microcenter',
            
            # Product specifications
            'specifications', 'tech specs', 'unboxing', 'hands-on', 'first look',
            'leaked', 'rumor', 'release date', 'launch', 'announcement'
        ]
        
        # Business/financial keywords
        self.keep_keywords = [
            # Business/financial
            'investment', 'invest', 'funding', 'revenue', 'earnings', 'profit',
            'loss', 'market cap', 'stock', 'share', 'dividend', 'acquisition',
            'merger', 'partnership', 'strategic', 'business', 'enterprise',
            
            # Financial metrics
            'quarterly', 'annual', 'financial', 'fiscal', 'guidance', 'outlook',
            'forecast', 'analyst', 'wall street', 'nasdaq', 'trading',
            
            # Corporate actions
            'ceo', 'executive', 'leadership', 'board', 'shareholder', 'ipo',
            'public offering', 'private equity', 'venture capital', 'valuation',
            
            # Industry/market
            'industry', 'sector', 'competition', 'competitor', 'market share',
            'data center', 'datacenter', 'cloud', 'enterprise', 'corporate',
            
            # AI/Tech business context
            'artificial intelligence', 'machine learning', 'ai chip', 'server',
            'infrastructure', 'computing', 'semiconductor', 'technology'
        ]
        
        # Strong business indicators
        self.strong_business_keywords = [
            'billion', 'million', 'investment', 'partnership', 'deal', 
            'revenue', 'earnings', 'acquisition', 'merger'
        ]
        
        # Consumer tech sources
        self.consumer_sources = [
            'slickdeals', 'techradar', 'pcgamer', 'tomshardware', 'anandtech',
            'engadget', 'the verge', 'ars technica', 'wccftech', 'techpowerup'
        ]
        
        # Adjust keywords based on mode
        if self.mode == FilterMode.STRICT:
            # Add more aggressive filtering in strict mode
            self.filter_out_keywords.extend(['review', 'test', 'comparison', 'vs'])
        elif self.mode == FilterMode.LENIENT:
            # Remove some keywords in lenient mode
            self.filter_out_keywords = [k for k in self.filter_out_keywords 
                                      if k not in ['announcement', 'launch', 'technology']]
    
    def _init_patterns(self):
        """Initialize regex patterns for filtering."""
        self.deal_pattern = re.compile(r'\$\d+.*off|save.*\$|deal.*\$|\d+% off|discount', re.IGNORECASE)
        self.gaming_performance_pattern = re.compile(r'\d+fps|\d+ fps|frame.*rate|benchmark|performance.*test', re.IGNORECASE)
    
    def load_news_data(self, file_path: str) -> Dict:
        """Load news data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    
    def save_news_data(self, data: Dict, file_path: str):
        """Save filtered news data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to save data to {file_path}: {e}")

    def _calculate_confidence_score(self, article: Dict[str, Any], text: str) -> float:
        """Calculate confidence score for article relevance (0-1)."""
        score = 0.5  # Start neutral
        
        # Positive indicators
        business_matches = sum(1 for keyword in self.keep_keywords if keyword in text)
        strong_business_matches = sum(1 for keyword in self.strong_business_keywords if keyword in text)
        
        # Weight strong business indicators more heavily
        score += (business_matches * 0.05) + (strong_business_matches * 0.15)
        
        # Negative indicators
        consumer_matches = sum(1 for keyword in self.filter_out_keywords if keyword in text)
        score -= consumer_matches * 0.1
        
        # Source credibility adjustment
        source = article.get('source', '').lower()
        if any(cs in source for cs in self.consumer_sources):
            score -= 0.2
        
        # Pattern-based adjustments
        if self.deal_pattern.search(text):
            score -= 0.3
        if self.gaming_performance_pattern.search(text):
            score -= 0.3
            
        return max(0.0, min(1.0, score))  # Clamp to 0-1 range
    
    def _get_matched_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """Get lists of matched positive and negative keywords."""
        positive_matches = [kw for kw in self.keep_keywords if kw in text]
        negative_matches = [kw for kw in self.filter_out_keywords if kw in text]
        return positive_matches, negative_matches
    
    def evaluate_article(self, article: Dict[str, Any]) -> FilterResult:
        """
        Evaluate a single article and return detailed filtering result.
        
        Args:
            article: Article dictionary with headline, description, source, etc.
            
        Returns:
            FilterResult with relevance decision, confidence, and metadata
        """
        # Combine headline and description for analysis
        text_to_analyze = f"{article.get('headline', '')} {article.get('description', '')}".lower()
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(article, text_to_analyze)
        
        # Get matched keywords
        positive_matches, negative_matches = self._get_matched_keywords(text_to_analyze)
        
        # Determine relevance based on mode and confidence
        threshold = {
            FilterMode.STRICT: 0.7,
            FilterMode.MODERATE: 0.5,
            FilterMode.LENIENT: 0.3
        }[self.mode]
        
        is_relevant = confidence >= threshold
        
        # Build filter reasons
        filter_reasons = []
        
        # Check for filter-out keywords
        for keyword in self.filter_out_keywords:
            if keyword in text_to_analyze:
                # Double check - if it also contains strong business keywords, might still be relevant
                business_context = any(bkw in text_to_analyze for bkw in self.strong_business_keywords)
                if not business_context:
                    filter_reasons.append(f"Contains consumer keyword: '{keyword}'")
        
        # Check source credibility
        source = article.get('source', '').lower()
        if any(cs in source for cs in self.consumer_sources):
            if not any(keyword in text_to_analyze for keyword in self.strong_business_keywords):
                filter_reasons.append(f"Consumer tech source: '{source}'")
        
        # Additional pattern checks
        if self.deal_pattern.search(text_to_analyze):
            filter_reasons.append("Contains deal/discount patterns")
            
        if self.gaming_performance_pattern.search(text_to_analyze):
            filter_reasons.append("Contains gaming performance patterns")
        
        return FilterResult(
            is_relevant=is_relevant,
            confidence_score=confidence,
            matched_keywords=positive_matches + negative_matches,
            filter_reasons=filter_reasons
        )
    
    def filter_articles(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[FilterResult]]:
        """
        Filter a list of articles and return filtered articles with results.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (filtered_articles, filter_results)
        """
        filtered_articles = []
        filter_results = []
        
        for article in articles:
            result = self.evaluate_article(article)
            filter_results.append(result)
            
            if result.is_relevant:
                filtered_articles.append(article)
        
        return filtered_articles, filter_results
    
    def filter_news_data(self, data: Dict, verbose: bool = True) -> Tuple[Dict, Dict[str, FilterStats]]:
        """
        Filter news data by removing irrelevant articles.
        
        Args:
            data: Dictionary with company names as keys and article lists as values
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (filtered_data, statistics_dict)
        """
        filtered_data = {}
        stats = {}
        
        for company, articles in data.items():
            if not isinstance(articles, list):
                filtered_data[company] = articles
                continue
                
            original_count = len(articles)
            filtered_articles, _ = self.filter_articles(articles)
            filtered_count = len(filtered_articles)
            removed_count = original_count - filtered_count
            percentage_kept = round((filtered_count / original_count) * 100, 1) if original_count > 0 else 0
            
            filtered_data[company] = filtered_articles
            stats[company] = FilterStats(
                original_count=original_count,
                filtered_count=filtered_count,
                removed_count=removed_count,
                percentage_kept=percentage_kept
            )
            
            if verbose:
                print(f"{company}: {original_count} -> {filtered_count} articles "
                      f"({removed_count} removed, {percentage_kept}% kept)")
        
        return filtered_data, stats
    
    def add_custom_keywords(self, filter_out: Optional[List[str]] = None, 
                           keep: Optional[List[str]] = None):
        """Add custom keywords to the filter lists."""
        if filter_out:
            self.filter_out_keywords.extend(filter_out)
        if keep:
            self.keep_keywords.extend(keep)
    
    def remove_keywords(self, filter_out: Optional[List[str]] = None, 
                       keep: Optional[List[str]] = None):
        """Remove keywords from the filter lists."""
        if filter_out:
            self.filter_out_keywords = [kw for kw in self.filter_out_keywords if kw not in filter_out]
        if keep:
            self.keep_keywords = [kw for kw in self.keep_keywords if kw not in keep]

def main():
    """Main function to run the news filtering process."""
    input_file = 'data/raw/news_api/raw_news_api.json'
    output_file = f'data/raw/news_api/filtered_{input_file.split("/")[-1].replace("raw_", "")}'
    
    # Initialize news filter with moderate mode
    news_filter = NewsFilter(mode=FilterMode.MODERATE)
    
    print("Loading news data...")
    try:
        data = news_filter.load_news_data(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("Filtering news articles...")
    filtered_data, stats = news_filter.filter_news_data(data)
    
    print("\nFiltering completed. Statistics:")
    total_original = sum(s.original_count for s in stats.values())
    total_filtered = sum(s.filtered_count for s in stats.values())
    total_removed = total_original - total_filtered
    
    for company, stat in stats.items():
        print(f"  {company}: {stat.original_count} -> {stat.filtered_count} ({stat.removed_count} removed)")
    
    print(f"\nOverall: {total_original} -> {total_filtered} articles ({total_removed} removed)")
    print(f"Total retention rate: {round((total_filtered / total_original) * 100, 1)}%")
    
    print(f"\nSaving filtered data to {output_file}...")
    try:
        news_filter.save_news_data(filtered_data, output_file)
        print("Filtering complete!")
    except IOError as e:
        print(f"Error saving file: {e}")

def create_filter_example():
    """Example of how to use the NewsFilter class with different configurations."""
    
    # Example 1: Basic usage with different modes
    print("=== Example 1: Different Filter Modes ===")
    
    sample_articles = [
        {
            "headline": "NVIDIA Reports Record Revenue in Q3 Earnings",
            "description": "NVIDIA announced quarterly earnings with record revenue driven by AI chip demand",
            "source": "Reuters"
        },
        {
            "headline": "Best Gaming Laptop Deals This Week",
            "description": "Save up to $500 on gaming laptops with RTX 4080 graphics cards",
            "source": "TechRadar"
        },
        {
            "headline": "AMD Partners with Microsoft for AI Infrastructure",
            "description": "Strategic partnership announced to develop next-generation datacenter solutions",
            "source": "Financial Times"
        }
    ]
    
    for mode in FilterMode:
        print(f"\n--- {mode.value.upper()} Mode ---")
        filter_instance = NewsFilter(mode=mode)
        
        for i, article in enumerate(sample_articles, 1):
            result = filter_instance.evaluate_article(article)
            print(f"Article {i}: {'KEEP' if result.is_relevant else 'FILTER'} "
                  f"(confidence: {result.confidence_score:.2f})")
            if result.filter_reasons:
                print(f"  Reasons: {', '.join(result.filter_reasons)}")
    
    # Example 2: Custom keyword management
    print("\n=== Example 2: Custom Keywords ===")
    custom_filter = NewsFilter(mode=FilterMode.MODERATE)
    
    # Add custom keywords
    custom_filter.add_custom_keywords(
        filter_out=["cryptocurrency", "bitcoin"],
        keep=["sustainability", "renewable energy"]
    )
    
    crypto_article = {
        "headline": "NVIDIA GPU Mining Cryptocurrency Trends",
        "description": "Analysis of bitcoin mining using NVIDIA graphics cards",
        "source": "CryptoNews"
    }
    
    result = custom_filter.evaluate_article(crypto_article)
    print(f"Crypto article: {'KEEP' if result.is_relevant else 'FILTER'} "
          f"(confidence: {result.confidence_score:.2f})")

if __name__ == "__main__":
    # Run main filtering process
    main()
    
    # Show usage examples
    print("\n" + "="*50)
    create_filter_example()