import os
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

def load_dataset(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load dataset from JSON file"""
    if not os.path.exists(file_path):
        print(f"Dataset file {file_path} does not exist.")
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def filter_by_company(data: Dict[str, List[Dict[str, Any]]], company: str) -> Dict[str, List[Dict[str, Any]]]:
    """Filter data by company"""
    if company.upper() == "ALL":
        return data
    
    # Case-insensitive company matching
    company_upper = company.upper()
    
    # Find the correct company key
    for key in data.keys():
        if key.upper() == company_upper:
            return {key: data[key]}
    
    # If not found, show available companies
    available_companies = list(data.keys())
    print(f"Company '{company}' not found in dataset.")
    print(f"Available companies: {', '.join(available_companies)}")
    return {}

def filter_by_date_range(data: Dict[str, List[Dict[str, Any]]], since_date: datetime, until_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
    """Filter data by date range"""
    filtered_data = {}
    
    for company, articles in data.items():
        filtered_articles = []
        for article in articles:
            # Try different date fields and formats
            article_date = None
            
            # Try scraped_date first (YYYY-MM-DD format)
            if "scraped_date" in article:
                try:
                    article_date = datetime.strptime(article["scraped_date"], "%Y-%m-%d")
                except ValueError:
                    pass
            
            # Try timestamp field (ISO format)
            if article_date is None and "timestamp" in article:
                try:
                    # Handle ISO format with Z suffix
                    timestamp_str = article["timestamp"].replace("Z", "+00:00")
                    article_date = datetime.fromisoformat(timestamp_str.replace("+00:00", ""))
                except ValueError:
                    pass
            
            # Try publishedAt field
            if article_date is None and "publishedAt" in article:
                try:
                    if "T" in article["publishedAt"]:
                        # ISO format
                        timestamp_str = article["publishedAt"].replace("Z", "")
                        article_date = datetime.fromisoformat(timestamp_str)
                    else:
                        # Simple date format
                        article_date = datetime.strptime(article["publishedAt"], "%Y-%m-%d")
                except ValueError:
                    pass
            
            # If we found a valid date, check if it's in range
            if article_date and since_date <= article_date <= until_date:
                filtered_articles.append(article)
        
        if filtered_articles:
            filtered_data[company] = filtered_articles
    
    return filtered_data

def parse_date_argument(date_str: str) -> datetime:
    """Parse date argument with special handling for 'oldest'"""
    if date_str.lower() == "oldest":
        return datetime(1970, 1, 1)
    
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format or 'oldest'.")

def save_results(results: Dict[str, List[Dict[str, Any]]], company: str, since: str, until: str):
    """Save results to a JSON file with proper nomenclature"""
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M%S")
    
    # Create filename based on parameters - format: date_time_[company-included]_[since]_[to]
    company_part = company.lower() if company.upper() != "ALL" else "all"
    since_part = since if since != "7days" else "7days"
    until_part = until if until != "today" else "today"

    filename = f"query_{current_date}_{current_time}_{company_part}_{since_part}_{until_part}.json"
    output_path = os.path.join("..", "data", "query", filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Query news dataset for NVIDIA, AMD, and Intel articles",
        epilog="""
Examples:
  python 4_query.py --company NVIDIA --since 2025-09-15 --until 2025-09-22
  python 4_query.py --company ALL --since oldest
  python 4_query.py --since 7days
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--company", default="ALL", 
                       help="Company to filter (NVIDIA, Intel, AMD, ALL). Default: ALL")
    parser.add_argument("--since", default="7days",
                       help="Start date (YYYY-MM-DD, 'oldest' for all data, or '7days' for last 7 days). Default: 7days")
    parser.add_argument("--until", default="today",
                       help="End date (YYYY-MM-DD or 'today'). Default: today")
    
    args = parser.parse_args()
    
    # Load dataset
    DATASET_FILE = os.path.join("..", "data", "dataset.json")
    data = load_dataset(DATASET_FILE)
    
    if not data:
        return
    
    # Parse date arguments
    today = datetime.now()
    
    if args.since == "7days":
        since_date = today - timedelta(days=7)
    else:
        since_date = parse_date_argument(args.since)
    
    if args.until == "today":
        until_date = today
    else:
        until_date = parse_date_argument(args.until)
    
    # Filter by company
    filtered_data = filter_by_company(data, args.company)
    
    # Filter by date range
    filtered_data = filter_by_date_range(filtered_data, since_date, until_date)
    
    # Display results summary
    total_articles = sum(len(articles) for articles in filtered_data.values())
    print(f"\nQuery Results:")
    print(f"Company: {args.company}")
    print(f"Date range: {since_date.strftime('%Y-%m-%d')} to {until_date.strftime('%Y-%m-%d')}")
    print(f"Total articles found: {total_articles}")
    
    for company, articles in filtered_data.items():
        print(f"  {company}: {len(articles)} articles")
    
    # Save results
    if filtered_data:
        save_results(filtered_data, args.company, args.since, args.until)
    else:
        print("No articles found matching the criteria.")

if __name__ == "__main__":
    main()