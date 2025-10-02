from smolagents import OpenAIServerModel, tool, ToolCallingAgent
from ddgs import DDGS
from dotenv import load_dotenv
import os
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TECH_CONTEXT_MAP = {
    # GPU/Hardware Terms
    "H200": "Nvidia H200 GPU artificial intelligence machine learning data center tensor core",
    "H100": "Nvidia H100 GPU AI accelerator data center computing hopper architecture",
    "A100": "Nvidia A100 GPU artificial intelligence machine learning ampere data center",
    "RTX": "Nvidia RTX graphics processing unit ray tracing gaming professional",
    "GeForce": "Nvidia GeForce gaming graphics card consumer market",
    "Radeon": "AMD Radeon graphics processing unit gaming RDNA architecture",
    "RDNA": "AMD RDNA graphics architecture gaming performance efficiency",
    "CDNA": "AMD CDNA compute DNA data center AI acceleration instinct",
    "Xeon": "Intel Xeon server processor data center enterprise scalable",
    "Core": "Intel Core processor consumer desktop laptop performance",
    
    # Business/Financial Terms
    "foundry": "semiconductor manufacturing fabrication services contract manufacturing",
    "fab": "semiconductor fabrication facility manufacturing process technology",
    "node": "semiconductor manufacturing process technology nanometer transistor density",
    "wafer": "semiconductor silicon wafer chip manufacturing substrate",
    "TSMC": "Taiwan Semiconductor Manufacturing Company foundry advanced process",
    "earnings": "quarterly earnings financial results revenue profit performance",
    "guidance": "financial guidance forecast outlook revenue expectations",
    "margin": "gross margin profit margin financial performance profitability",
    "capex": "capital expenditure investment spending infrastructure equipment",
    "opex": "operational expenditure operating expenses business costs"
}

TRUSTED_SOURCES = {
    "tier_1": {
        "domains": [
            "bloomberg.com", "reuters.com", "wsj.com", "ft.com",
            "cnbc.com", "marketwatch.com", "finance.yahoo.com"
        ],
        "weight": 1.0,
        "reliability_score": 95
    },
    "tier_2": {
        "domains": [
            "techcrunch.com", "theverge.com", "arstechnica.com",
            "seekingalpha.com", "businessinsider.com", "forbes.com",
            "venturebeat.com", "wired.com"
        ],
        "weight": 0.8,
        "reliability_score": 85
    },
    "tier_3": {
        "domains": [
            "anandtech.com", "tomshardware.com", "electronicsweekly.com",
            "eetimes.com", "semiconductor-digest.com", "techpowerup.com"
        ],
        "weight": 0.7,
        "reliability_score": 80
    }
}

BUSINESS_KEYWORDS = {
    "high_priority": {
        "keywords": [
            "earnings", "revenue", "profit", "quarterly", "financial results",
            "stock price", "market cap", "valuation", "ipo", "acquisition",
            "merger", "partnership", "investment", "funding", "venture capital",
            "ceo", "executive", "board", "shareholder", "dividend",
            "data center", "datacenter", "enterprise", "artificial intelligence",
            "machine learning", "semiconductor", "chip manufacturing"
        ],
        "weight": 2.0
    },
    "medium_priority": {
        "keywords": [
            "cloud computing", "server", "manufacturing", "supply chain",
            "export", "regulation", "competition", "market share",
            "industry analysis", "technology roadmap", "innovation",
            "research development", "patent", "licensing"
        ],
        "weight": 1.5
    },
    "filter_out": {
        "keywords": [
            "gaming laptop", "gaming pc", "fps test", "benchmark score",
            "graphics settings", "game review", "laptop deal", "consumer review",
            "unboxing", "hands-on review", "price drop", "shopping deal",
            "affiliate discount", "black friday", "cyber monday"
        ],
        "weight": -2.0
    }
}

IMPACT_INDICATORS = {
    "financial_impact": {
        "high": ["billion", "merger", "acquisition", "bankruptcy", "ipo", "guidance cut", "guidance raise"],
        "medium": ["million", "partnership", "expansion", "layoffs", "restructuring"],
        "low": ["thousand", "update", "announcement", "conference"]
    },
    "market_impact": {
        "high": ["stock split", "dividend increase", "major contract", "regulatory approval"],
        "medium": ["product launch", "new facility", "executive change", "strategic shift"],
        "low": ["minor update", "press conference", "interview", "analyst meeting"]
    },
    "technology_impact": {
        "high": ["breakthrough", "new architecture", "major upgrade", "paradigm shift"],
        "medium": ["product refresh", "performance improvement", "new feature"],
        "low": ["minor update", "bug fix", "patch", "optimization"]
    }
}

@dataclass
class EnhancedSearchResult:
    title: str
    url: str
    snippet: str
    source_tier: str  # "premium", "established", "specialized", "unknown"
    content_type: str  # "financial_news", "business_analysis", "tech_business", "consumer_content"
    significance: str  # "high_impact", "moderate_impact", "minor_update", "routine_news"
    opinion_summary: str  # Brief qualitative assessment
    search_metadata: Dict[str, Any]

class SearchCache:
    """Simple in-memory cache for search results"""
    def __init__(self, ttl_minutes: int = 30):
        self.cache = {}
        self.ttl = ttl_minutes * 60
    
    def get_cache_key(self, query: str, max_results: int) -> str:
        return hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
    
    def get(self, key: str) -> List[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: List[Dict]):
        self.cache[key] = (data, time.time())

class ContextExpander:
    """Expands search queries with relevant business and technical context"""
    
    def __init__(self):
        self.last_expansion = ""
    
    def expand_query(self, query: str) -> str:
        """Expand query with relevant technical and business context"""
        expanded_terms = []
        words = query.lower().split()
        original_query = query
        
        # Add original terms
        expanded_terms.extend(words)
        
        # Add context for technical terms
        for word in words:
            if word in TECH_CONTEXT_MAP:
                context_terms = TECH_CONTEXT_MAP[word].split()
                expanded_terms.extend(context_terms)
        
        # Add business context for company names
        companies = ["nvidia", "amd", "intel"]
        for company in companies:
            if company in query.lower():
                if any(financial_term in query.lower() for financial_term in ["earnings", "revenue", "stock", "financial"]):
                    expanded_terms.extend(["quarterly", "financial", "results", "business", "performance"])
                else:
                    expanded_terms.extend(["corporation", "company", "business", "technology"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        expanded_query = " ".join(unique_terms)
        self.last_expansion = expanded_query
        return expanded_query

class SourceReliabilityFilter:
    """Filters and scores search results based on source reliability"""
    
    def assess_source_quality(self, url: str) -> Tuple[str, str]:
        """Assess source quality and return tier and description"""
        domain = urlparse(url).netloc.lower()
        
        tier_descriptions = {
            "tier_1": ("premium", "Premium financial/business source with high credibility"),
            "tier_2": ("established", "Established tech/business publication"),
            "tier_3": ("specialized", "Industry-specialized publication")
        }
        
        for tier, info in TRUSTED_SOURCES.items():
            for trusted_domain in info["domains"]:
                if trusted_domain in domain:
                    return tier_descriptions[tier]
        
        # Unknown source
        return "unknown", "Unverified or less established source"
    
    def is_business_source(self, url: str) -> bool:
        """Check if the source is business/financial focused"""
        business_domains = [
            "bloomberg", "reuters", "wsj", "cnbc", "marketwatch", 
            "seekingalpha", "businessinsider", "forbes"
        ]
        domain = urlparse(url).netloc.lower()
        return any(business_domain in domain for business_domain in business_domains)

class RelevanceScorer:
    """Scores search results for business/financial relevance"""
    
    def classify_content_type(self, title: str, snippet: str) -> Tuple[str, str]:
        """Classify content type and provide reasoning"""
        text = f"{title} {snippet}".lower()
        
        # Check for high-priority financial content
        financial_indicators = ["earnings", "revenue", "quarterly", "financial results", "stock price", "valuation"]
        if any(indicator in text for indicator in financial_indicators):
            return "financial_news", "Contains financial/earnings information"
        
        # Check for business analysis content
        business_indicators = ["partnership", "acquisition", "market share", "competition", "strategy"]
        if any(indicator in text for indicator in business_indicators):
            return "business_analysis", "Business strategy or competitive analysis"
        
        # Check for tech business content
        tech_business_indicators = ["data center", "enterprise", "artificial intelligence", "semiconductor"]
        if any(indicator in text for indicator in tech_business_indicators):
            return "tech_business", "Technology business or enterprise focus"
        
        # Check for consumer content (to filter out)
        consumer_indicators = ["gaming", "laptop deal", "review", "fps", "benchmark"]
        if any(indicator in text for indicator in consumer_indicators):
            return "consumer_content", "Consumer-focused or product review content"
        
        return "general_news", "General news or announcement"

class ImpactAnalyzer:
    """Analyzes the potential business impact of news items"""
    
    def assess_significance(self, title: str, snippet: str) -> Tuple[str, str, List[str]]:
        """Assess news significance and provide reasoning"""
        text = f"{title} {snippet}".lower()
        
        high_impact_indicators = ["billion", "merger", "acquisition", "bankruptcy", "ipo", "guidance cut", "guidance raise"]
        medium_impact_indicators = ["million", "partnership", "expansion", "layoffs", "restructuring", "product launch"]
        
        found_indicators = []
        
        # Check for high impact
        high_found = [indicator for indicator in high_impact_indicators if indicator in text]
        if high_found:
            found_indicators.extend(high_found)
            return "high_impact", "Major business event with significant financial implications", found_indicators
        
        # Check for medium impact
        medium_found = [indicator for indicator in medium_impact_indicators if indicator in text]
        if medium_found:
            found_indicators.extend(medium_found)
            return "moderate_impact", "Notable business development worth monitoring", found_indicators
        
        # Check for minor updates
        minor_indicators = ["update", "announcement", "conference", "interview"]
        minor_found = [indicator for indicator in minor_indicators if indicator in text]
        if minor_found:
            found_indicators.extend(minor_found)
            return "minor_update", "Regular business update or announcement", found_indicators
        
        return "routine_news", "Standard news item without clear business impact indicators", []

class EnhancedSearchEngine:
    """Main enhanced search engine combining all improvements"""
    
    def __init__(self):
        self.context_expander = ContextExpander()
        self.source_filter = SourceReliabilityFilter()
        self.relevance_scorer = RelevanceScorer()
        self.impact_analyzer = ImpactAnalyzer()
        self.cache = SearchCache()
    
    def search(self, query: str, max_results: int = 20, min_relevance: float = 0.5) -> List[EnhancedSearchResult]:
        """Perform enhanced search with all improvements"""
        
        # Check cache first
        cache_key = self.cache.get_cache_key(query, max_results)
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return [EnhancedSearchResult(**result) for result in cached_results]
        
        # Step 1: Expand query with context
        expanded_query = self.context_expander.expand_query(query)
        
        # Step 2: Perform base search with more results for filtering
        raw_results = self._perform_base_search(expanded_query, max_results * 3)
        
        # Step 3: Process and enhance results
        enhanced_results = []
        
        for result in raw_results:
            # Assess source quality
            source_tier, source_description = self.source_filter.assess_source_quality(result["link"])
            
            # Classify content type
            content_type, content_reasoning = self.relevance_scorer.classify_content_type(result["title"], result["snippet"])
            
            # Skip consumer content if looking for business focus
            if content_type == "consumer_content" and min_relevance > 0.3:
                continue
            
            # Assess significance
            significance, significance_reasoning, key_indicators = self.impact_analyzer.assess_significance(result["title"], result["snippet"])
            
            # Create opinion summary
            opinion_parts = []
            if source_tier in ["premium", "established"]:
                opinion_parts.append(f"From {source_description.lower()}")
            if content_type in ["financial_news", "business_analysis"]:
                opinion_parts.append(f"{content_reasoning.lower()}")
            if significance in ["high_impact", "moderate_impact"]:
                opinion_parts.append(f"{significance_reasoning.lower()}")
            
            opinion_summary = "; ".join(opinion_parts) if opinion_parts else "Standard news item"
            
            # Create enhanced result
            enhanced_result = EnhancedSearchResult(
                title=result["title"],
                url=result["link"],
                snippet=result["snippet"],
                source_tier=source_tier,
                content_type=content_type,
                significance=significance,
                opinion_summary=opinion_summary,
                search_metadata={
                    "expanded_query": expanded_query,
                    "original_query": query,
                    "source_description": source_description,
                    "content_reasoning": content_reasoning,
                    "significance_reasoning": significance_reasoning,
                    "key_indicators": key_indicators,
                    "is_business_source": self.source_filter.is_business_source(result["link"])
                }
            )
            
            enhanced_results.append(enhanced_result)
        
        # Step 4: Sort by qualitative priority
        def get_priority_score(result):
            priority = 0
            
            # Source tier priority
            if result.source_tier == "premium":
                priority += 3
            elif result.source_tier == "established":
                priority += 2
            elif result.source_tier == "specialized":
                priority += 1
            
            # Content type priority
            if result.content_type == "financial_news":
                priority += 3
            elif result.content_type == "business_analysis":
                priority += 2
            elif result.content_type == "tech_business":
                priority += 1
            
            # Significance priority
            if result.significance == "high_impact":
                priority += 3
            elif result.significance == "moderate_impact":
                priority += 2
            elif result.significance == "minor_update":
                priority += 1
            
            return priority
        
        enhanced_results.sort(key=get_priority_score, reverse=True)
        
        # Step 5: Return top results and cache them
        final_results = enhanced_results[:max_results]
        
        # Cache results for future use
        cache_data = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "source_tier": r.source_tier,
                "content_type": r.content_type,
                "significance": r.significance,
                "opinion_summary": r.opinion_summary,
                "search_metadata": r.search_metadata
            }
            for r in final_results
        ]
        self.cache.set(cache_key, cache_data)
        
        return final_results
    
    def _perform_base_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform the base DuckDuckGo search"""
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            return results
        except Exception as e:
            print(f"Base search error: {e}")
            return []

enhanced_search_engine = EnhancedSearchEngine()

@tool
def enhanced_internet_search(
    query: str, 
    max_results: int = 20,
    focus: str = "financial",
    min_relevance: float = 0.5
) -> str:
    """
    Perform enhanced internet search with context expansion, source filtering,
    relevance scoring, and impact analysis for financial/business information.

    Args:
        query (str): A keyword or phrase to search for. Will be automatically 
                    expanded with relevant business and technical context.
            Examples:
                - "Tesla Q3 2025 earnings report"
                - "H200 GPU performance data center"
                - "AMD Radeon competition NVIDIA"
        max_results (int): Number of search results to return (default = 20).
        focus (str): Search focus area ("financial", "technical", "business").
        min_relevance (float): Minimum relevance score to include results (0.0-3.0).

    Returns:
        str: Enhanced JSON array of search results. Each object contains:
            - title (str): The title of the webpage.
            - url (str): The URL of the source.
            - snippet (str): A short summary or preview text.
            - source_tier (str): Source quality tier (premium/established/specialized/unknown).
            - content_type (str): Content classification (financial_news/business_analysis/tech_business/consumer_content).
            - significance (str): Business significance (high_impact/moderate_impact/minor_update/routine_news).
            - opinion_summary (str): Qualitative assessment of the article's value.
            - assessment (dict): Detailed reasoning for classifications.
            - search_metadata (dict): Additional context and analysis data.

    Usage:
        - Automatically expands technical terms (e.g., "H200" → "Nvidia H200 GPU AI...")
        - Prioritizes trusted financial and technology sources
        - Filters out consumer/gaming content in favor of business news
        - Provides impact analysis for business decision making
        - Returns structured JSON for easy parsing by other agents
    """
    try:
        # Adjust min_relevance based on focus
        if focus == "financial":
            min_relevance = max(min_relevance, 1.0)  # Higher threshold for financial focus
        elif focus == "technical":
            min_relevance = max(min_relevance, 0.7)
        
        # Perform enhanced search
        results = enhanced_search_engine.search(query, max_results, min_relevance)
        
        if not results:
            return json.dumps([{
                "error": f"No relevant results found for query: {query}",
                "suggestion": "Try a broader search term or lower the min_relevance threshold"
            }], indent=2)
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source_tier": result.source_tier,
                "content_type": result.content_type,
                "significance": result.significance,
                "opinion_summary": result.opinion_summary,
                "assessment": {
                    "source_quality": result.search_metadata["source_description"],
                    "content_analysis": result.search_metadata["content_reasoning"],
                    "business_significance": result.search_metadata["significance_reasoning"],
                    "key_indicators": result.search_metadata["key_indicators"]
                },
                "search_metadata": {
                    "expanded_query": result.search_metadata["expanded_query"],
                    "original_query": result.search_metadata["original_query"],
                    "search_focus": focus,
                    "total_results_processed": len(results)
                }
            })
        
        return json.dumps(serializable_results, indent=2)
        
    except Exception as e:
        return json.dumps([{"error": f"Enhanced search failed: {str(e)}"}], indent=2)

@tool
def delay_tool(seconds: int) -> str:
    """
    Pauses the execution for a specified number of seconds.
    Use this tool to manage rate limits when instructed by the orchestrator.

    Args:
        seconds (int): The number of seconds to wait.

    Returns:
        str: A confirmation message that the delay has completed.
    """
    print(f"-> Tool 'delay_tool' called. Waiting for {seconds} seconds...")
    time.sleep(seconds)
    print("-> Delay finished.")
    return f"Successfully delayed for {seconds} seconds."

model = OpenAIServerModel(
    model_id="gemini-2.5-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/",
    api_key=GEMINI_API_KEY,
)

enhanced_search_agent = ToolCallingAgent(
    model=model,
    tools=[delay_tool,enhanced_internet_search],
    name="Enhanced_Search_Agent",
    description=(
        "You are a specialized search tool that ONLY performs internet searches and returns "
        "the raw JSON search results. Do NOT provide summaries, analysis, or final answers. "
        "Simply call the enhanced_internet_search tool with the user's query and return the "
        "JSON results directly. Your job is to retrieve and return search data, not to "
        "interpret or summarize it."
    ),
    stream_outputs=False,
)

if __name__ == "__main__":
    query = "H200"
    print(f"Testing agent with query: '{query}'")
    
    try:
        result = enhanced_search_agent.run(f"Please search for: {query}")
        print(f"Agent Result Type: {type(result)}")

        print(f"Agent Result:\n{result}")
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")