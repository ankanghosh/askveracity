import feedparser
import time
import logging
import re
import ssl
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
# Import the performance tracker
from utils.performance import PerformanceTracker
performance_tracker = PerformanceTracker()

logger = logging.getLogger("misinformation_detector")

# Disable SSL certificate verification for feeds with self-signed certs
ssl._create_default_https_context = ssl._create_unverified_context

# List of RSS feeds to check for news
# These are popular news sources with reliable and frequently updated RSS feeds
RSS_FEEDS = [
# --------------------
# 🌐 General World News
# --------------------
"http://rss.cnn.com/rss/cnn_world.rss",                         # CNN World News
"https://rss.nytimes.com/services/xml/rss/nyt/World.xml",      # NYT World News
"https://feeds.washingtonpost.com/rss/world",                  # The Washington Post World News
"https://feeds.bbci.co.uk/news/world/rss.xml",                 # BBC News - World

# --------------------
# 🧠 Tech & Startup News (Global)
# --------------------
"https://techcrunch.com/feed/",                                # TechCrunch - Startup and Technology News
"https://venturebeat.com/feed/",                               # VentureBeat - Tech News
"https://www.wired.com/feed/rss",                              # Wired - Technology News
"https://www.cnet.com/rss/news/",                              # CNET - Technology News
"https://news.google.com/rss?gl=IN&ceid=IN:en&topic=t&hl=en-IN",  # Google News India - Technology
"https://news.google.com/rss?gl=US&ceid=US:en&topic=t&hl=en-US",  # Google News US - Technology

# --------------------
# 💼 Startup & VC Focused
# --------------------
"https://news.crunchbase.com/feed/",                           # Crunchbase News - Startup Funding
"https://techstartups.com/feed/",                              # Tech Startups - Startup News

# --------------------
# 📰 Global Business & Corporate Feeds
# --------------------
"https://feeds.bloomberg.com/technology/news.rss",             # Bloomberg Technology News
"https://www.ft.com/technology?format=rss",                    # Financial Times Technology News
"https://news.google.com/rss?gl=IN&ceid=IN:en&topic=b&hl=en-IN",  # Google News India - Business

# --------------------
# 🇮🇳 India-specific News
# --------------------
"https://inc42.com/feed/",                                     # Inc42 - Indian Startups and Technology
"https://timesofindia.indiatimes.com/rssfeedstopstories.cms",           # TOI - Top Stories
"https://timesofindia.indiatimes.com/rssfeedmostrecent.cms",            # TOI - Most Recent Stories
"https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",         # TOI - India News
"https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",           # TOI - World News
"https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",             # TOI - Business News
"https://timesofindia.indiatimes.com/rssfeeds/54829575.cms",            # TOI - Cricket News
"https://timesofindia.indiatimes.com/rssfeeds/4719148.cms",             # TOI - Sports News
"https://timesofindia.indiatimes.com/rssfeeds/-2128672765.cms",         # TOI - Science News

# --------------------
# 🏏 Sports News (Global + Cricket)
# --------------------
"https://www.espn.com/espn/rss/news",                          # ESPN - Top Sports News
"https://feeds.skynews.com/feeds/rss/sports.xml",              # Sky News - Sports
"https://sports.ndtv.com/rss/all",                                 # NDTV Sports
"https://www.espncricinfo.com/rss/content/story/feeds/0.xml",  # ESPN Cricinfo - Cricket News

# --------------------
# ✅ Fact-Checking Sources
# --------------------
"https://www.snopes.com/feed/",                                # Snopes - Fact Checking
"https://www.politifact.com/rss/all/",                         # PolitiFact - Fact Checking
"https://www.factcheck.org/feed/",                             # FactCheck - Fact Checking
"https://leadstories.com/atom.xml",                            # Lead Stories - Fact Checking
"https://fullfact.org/feed/all/",                              # Full Fact - Fact Checking
"https://www.truthorfiction.com/feed/",                         # TruthOrFiction - Fact Checking

# --------------------
# 🗳️ Politics & Policy (General)
# --------------------
"https://feeds.bbci.co.uk/news/politics/rss.xml",              # BBC News - Politics
"https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC - Science & Environment

# --------------------
# 🗳️ Science
# --------------------
"https://www.nature.com/nature.rss",                              # Nature science
"https://feeds.science.org/rss/science-advances.xml"              # science.org
]

def clean_html(raw_html):
    """Remove HTML tags from text"""
    if not raw_html:
        return ""
    clean_regex = re.compile('<.*?>')
    clean_text = re.sub(clean_regex, '', raw_html)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def parse_feed(feed_url, timeout=5):
    """
    Parse a single RSS feed with proper timeout handling
    Uses requests with timeout first, then passes content to feedparser
    """
    try:
        # Use requests with timeout to fetch the RSS content
        response = requests.get(feed_url, timeout=timeout)
        response.raise_for_status()
        
        # Then parse the content with feedparser (which doesn't support timeout)
        feed = feedparser.parse(response.content)
        
        # Basic validation of the feed
        if hasattr(feed, 'entries') and feed.entries:
            return feed
        else:
            logger.warning(f"Feed {feed_url} parsed but contains no entries")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout while fetching feed {feed_url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching feed {feed_url}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching feed {feed_url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error parsing feed {feed_url}: {str(e)}")
        return None

def fetch_all_feeds(feeds_list=None, max_workers=5, timeout=5):
    """
    Fetch multiple RSS feeds with proper timeout handling
    Returns a list of (domain, feed) tuples for successfully fetched feeds
    """
    # Use default RSS_FEEDS list if none provided
    if feeds_list is None:
        feeds_list = RSS_FEEDS
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(parse_feed, url, timeout): url for url in feeds_list}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                feed = future.result()
                if feed and hasattr(feed, 'entries') and feed.entries:
                    # Extract domain for source attribution
                    domain = urlparse(url).netloc
                    results.append((domain, feed))
                    logger.info(f"Successfully fetched {domain} with {len(feed.entries)} entries")
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
    
    return results

def extract_date(entry):
    """Extract and normalize publication date from entry"""
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if hasattr(entry, date_field) and getattr(entry, date_field):
            try:
                # Convert time tuple to datetime
                time_tuple = getattr(entry, date_field)
                return datetime(time_tuple[0], time_tuple[1], time_tuple[2], 
                               time_tuple[3], time_tuple[4], time_tuple[5])
            except Exception as e:
                logger.debug(f"Error parsing {date_field}: {e}")
                continue
    
    # Try string dates
    for date_field in ['published', 'updated', 'pubDate']:
        if hasattr(entry, date_field) and getattr(entry, date_field):
            try:
                date_str = getattr(entry, date_field)
                # Try various formats
                for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z', 
                           '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            except Exception as e:
                logger.debug(f"Error parsing date string {date_field}: {e}")
                continue
    
    # Default to current time if parsing fails
    return datetime.now()

def is_recent(entry_date, claim=None, max_days=3):
    """
    Check if an entry is recent based on temporal indicators in the claim.
    
    Args:
        entry_date (datetime): The date of the entry to check
        claim (str, optional): The claim text to analyze for temporal indicators
        max_days (int, optional): Default maximum age in days
        
    Returns:
        bool: True if entry is considered recent, False otherwise
    """
    if not entry_date:
        return False
    
    # Default max days if no claim is provided
    default_days = max_days
    extended_days = 15  # For 'recently', 'this week', etc.
    
    if claim:
        # Specific day indicators get default days
        specific_day_terms = ["today", "yesterday", "day before yesterday"]
        
        # Extended time terms get extended days
        extended_time_terms = [
            "recently", "currently", "freshly", "this week", "few days", 
            "couple of days", "last week", "past week", "several days",
            "anymore"
        ]
        
        claim_lower = claim.lower()
        
        # Check for extended time terms first, then specific day terms
        if any(term in claim_lower for term in extended_time_terms):
            cutoff = datetime.now() - timedelta(days=extended_days)
            return entry_date > cutoff
        elif any(term in claim_lower for term in specific_day_terms):
            cutoff = datetime.now() - timedelta(days=default_days)
            return entry_date > cutoff
    
    # Default case - use standard window
    cutoff = datetime.now() - timedelta(days=default_days)
    return entry_date > cutoff

def get_entry_relevance(entry, query_terms, domain):
    """Calculate relevance score for an entry based on query match and recency"""
    if not hasattr(entry, 'title') or not entry.title:
        return 0
    
    # Extract text content
    title = entry.title or ""
    description = clean_html(entry.description) if hasattr(entry, 'description') else ""
    content = ""
    if hasattr(entry, 'content'):
        for content_item in entry.content:
            if 'value' in content_item:
                content += clean_html(content_item['value']) + " "
    
    # Extract published date
    pub_date = extract_date(entry)
    
    # Calculate recency score (0-1)
    recency_score = 0
    if pub_date:
        days_old = (datetime.now() - pub_date).days
        if days_old <= 1:  # Today or yesterday
            recency_score = 1.0
        elif days_old <= 2:
            recency_score = 0.8
        elif days_old <= 3:
            recency_score = 0.5
        else:
            recency_score = 0.2
    
    # Calculate relevance score based on keyword matches
    text = f"{title} {description} {content}".lower()
    
    # Count how many query terms appear in the content
    query_terms_lower = [term.lower() for term in query_terms]
    matches = sum(1 for term in query_terms_lower if term in text)
    
    # Calculate match score (0-1)
    match_score = min(1.0, matches / max(1, len(query_terms) * 0.7))
    
    # Boost score for exact phrase matches
    query_phrase = " ".join(query_terms_lower)
    if query_phrase in text:
        match_score += 0.5
    
    # Additional boost for title matches (they're more relevant)
    title_matches = sum(1 for term in query_terms_lower if term in title.lower())
    if title_matches > 0:
        match_score += 0.2 * (title_matches / len(query_terms_lower))
    
    # Source quality factor (can be adjusted based on source reliability)
    source_factor = 1.0
    high_quality_domains = ['bbc.co.uk', 'nytimes.com', 'reuters.com', 'washingtonpost.com', 
                           'espncricinfo.com', 'cricbuzz.com', 'snopes.com']
    if any(quality_domain in domain for quality_domain in high_quality_domains):
        source_factor = 1.2
    
    # Calculate final score
    final_score = (match_score * 0.6) + (recency_score * 0.4) * source_factor
    
    return min(1.0, final_score)  # Cap at 1.0

def retrieve_evidence_from_rss(claim, max_results=10, category_feeds=None):
    """
    Retrieve evidence from RSS feeds for a given claim
    
    Args:
        claim (str): The claim to verify
        max_results (int): Maximum number of results to return
        category_feeds (list, optional): List of category-specific RSS feeds to check
        
    Returns:
        list: List of relevant evidence items
    """
    start_time = time.time()
    logger.info(f"Retrieving evidence from RSS feeds for: {claim}")
    
    # Extract key terms from claim
    terms = [term.strip() for term in re.findall(r'\b\w+\b', claim) if len(term.strip()) > 2]
    
    try:
        # Use category-specific feeds if provided
        feeds_to_use = category_feeds if category_feeds else RSS_FEEDS
        
        # Log which feeds we're using
        if category_feeds:
            logger.info(f"Using {len(category_feeds)} category-specific RSS feeds")
        else:
            logger.info(f"Using {len(RSS_FEEDS)} default RSS feeds")
        
        # Limit the number of feeds to process for efficiency
        if len(feeds_to_use) > 10:
            # If we have too many feeds, select a subset
            # Prioritize fact-checking sources
            fact_check_feeds = [feed for feed in feeds_to_use if "fact" in feed.lower() or "snopes" in feed.lower() or "politifact" in feed.lower()]
            other_feeds = [feed for feed in feeds_to_use if feed not in fact_check_feeds]
            
            # Take all fact-checking feeds plus a random selection of others
            selected_feeds = fact_check_feeds + random.sample(other_feeds, min(max(0, 10 - len(fact_check_feeds)), len(other_feeds)))
            
        else:
            selected_feeds = feeds_to_use
            
        # Fetch all feeds in parallel with the selected feeds
        feeds = fetch_all_feeds(selected_feeds)
        
        if not feeds:
            logger.warning("No RSS feeds could be fetched")
            return []
        
        all_entries = []
        
        # Process all feed entries
        for domain, feed in feeds:
            for entry in feed.entries:
                # Calculate relevance score
                relevance = get_entry_relevance(entry, terms, domain)
                
                if relevance > 0.3:  # Only consider somewhat relevant entries
                    # Extract entry details
                    title = entry.title if hasattr(entry, 'title') else "No title"
                    link = entry.link if hasattr(entry, 'link') else ""
                    
                    # Extract and clean description/content
                    description = ""
                    if hasattr(entry, 'description'):
                        description = clean_html(entry.description)
                    elif hasattr(entry, 'summary'):
                        description = clean_html(entry.summary)
                    elif hasattr(entry, 'content'):
                        for content_item in entry.content:
                            if 'value' in content_item:
                                description += clean_html(content_item['value']) + " "
                    
                    # Truncate description if too long
                    if len(description) > 1000:
                        description = description[:1000] + "..."
                    
                    # Get publication date
                    pub_date = extract_date(entry)
                    date_str = pub_date.strftime('%Y-%m-%d') if pub_date else "Unknown date"
                    
                    # Format as evidence text
                    evidence_text = (
                        f"Title: {title}, "
                        f"Source: {domain} (RSS), "
                        f"Date: {date_str}, "
                        f"URL: {link}, "
                        f"Content: {description}"
                    )
                    
                    all_entries.append({
                        "text": evidence_text,
                        "relevance": relevance,
                        "date": pub_date or datetime.now()
                    })
        
        # Sort entries by relevance
        all_entries.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Take top results
        top_entries = all_entries[:max_results]
        
        logger.info(f"Retrieved {len(top_entries)} relevant RSS items from {len(feeds)} feeds in {time.time() - start_time:.2f}s")
        
        # Return just the text portion
        rss_results = [entry["text"] for entry in top_entries]
        
        # Log evidence retrieval performance
        success = bool(rss_results)
        source_count = {"rss": len(rss_results)}
        try:
            performance_tracker.log_evidence_retrieval(success, source_count)
        except Exception as e:
            logger.error(f"Error logging RSS evidence retrieval: {e}")
        
        return rss_results
    
    except Exception as e:
        logger.error(f"Error in RSS retrieval: {str(e)}")
        
        # Log failed evidence retrieval
        try:
            performance_tracker.log_evidence_retrieval(False, {"rss": 0})
        except Exception as log_error:
            logger.error(f"Error logging failed RSS evidence retrieval: {log_error}")
        
        return []