"""
Evidence retrieval module for the Fake News Detector application.

This module provides functions for retrieving evidence from various sources,
analyzing relevance using entity extraction and verb matching, and
combining evidence to support fact-checking operations.
"""

import logging
import time
import requests
import ssl
import urllib.request
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.api_utils import api_error_handler, safe_json_parse
from utils.models import get_nlp_model
from modules.claim_extraction import shorten_claim_for_evidence
from modules.rss_feed import retrieve_evidence_from_rss
from config import NEWS_API_KEY, FACTCHECK_API_KEY
from modules.category_detection import get_category_specific_rss_feeds, get_fallback_category, detect_claim_category
# Import the performance tracker
from utils.performance import PerformanceTracker
performance_tracker = PerformanceTracker()

logger = logging.getLogger("misinformation_detector")

def extract_claim_components(claim):
    """
    Extract key components from a claim using NER and dependency parsing.
    
    Args:
        claim (str): The claim text
        
    Returns:
        dict: Dictionary containing entities, verbs, and important keywords
    """
    if not claim:
        return {"entities": [], "verbs": [], "keywords": []}
    
    try:
        # Get NLP model
        nlp = get_nlp_model()
        doc = nlp(claim)
        
        # Extract named entities - keep original case for better matching
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        
        # Also extract any capitalized words as potential entities not caught by NER
        words = claim.split()
        for word in words:
            clean_word = word.strip('.,;:!?()[]{}""\'')
            # Check if word starts with capital letter and isn't already in entities
            if clean_word and clean_word[0].isupper() and clean_word not in entities:
                entities.append(clean_word)
        
        # Extract main verbs
        verbs = []
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                # Get the lemma to handle different verb forms
                verbs.append(token.lemma_.lower())
        
        # Extract important keywords (non-stopword nouns, adjectives)
        keywords = []
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop and len(token.text) > 2:
                keywords.append(token.text.lower())
        
        # Extract temporal indicators
        temporal_words = []
        temporal_indicators = ["today", "yesterday", "recently", "just", "now",
                               "current", "currently", "latest", "new", "week",
                               "month", "year", "announces", "announced", "introduces",
                               "introduced", "launches", "launched", "releases",
                               "released", "rolls out", "rolled out", "presents", "presented", "unveils", "unveiled", 
                               "starts", "started", "begins", "began", "initiates", "initiated", "anymore"
        ]

        for token in doc:
            if token.text.lower() in temporal_indicators:
                temporal_words.append(token.text.lower())
        
        return {
            "entities": entities,
            "verbs": verbs,
            "keywords": keywords,
            "temporal_words": temporal_words
        }
    except Exception as e:
        logger.error(f"Error extracting claim components: {e}")
        return {"entities": [], "verbs": [], "keywords": [], "temporal_words": []}

def analyze_evidence_relevance(evidence_items, claim_components):
    """
    Analyze evidence relevance based on entity match, verb match and keyword match.
    
    Args:
        evidence_items (list): List of evidence text strings
        claim_components (dict): Components extracted from the claim
        
    Returns:
        list: List of (evidence, score) tuples sorted by relevance score
    """
    if not evidence_items or not claim_components:
        return []
    
    scored_evidence = []
    
    # Extract components for easier access
    claim_entities = claim_components.get("entities", [])
    claim_verbs = claim_components.get("verbs", [])
    claim_keywords = claim_components.get("keywords", [])
    
    for evidence in evidence_items:
        if not isinstance(evidence, str):
            continue
        
        evidence_lower = evidence.lower()
        
        # 1. Count entity matches - try both case-sensitive and case-insensitive matching
        entity_matches = 0
        for entity in claim_entities:
            # Try exact match first (preserves case)
            if entity in evidence:
                entity_matches += 1
            # Then try lowercase match
            elif entity.lower() in evidence_lower:
                entity_matches += 1
        
        # 2. Count verb matches (always lowercase)
        verb_matches = sum(1 for verb in claim_verbs if verb in evidence_lower)
        
        # 3. Calculate entity and verb weighted score
        entity_verb_score = (entity_matches * 3.0) + (verb_matches * 2.0)
        
        # 4. Count keyword matches (always lowercase)
        keyword_matches = sum(1 for keyword in claim_keywords if keyword in evidence_lower)
        
        # 5. Determine final score based on entity and verb matches
        if entity_verb_score > 0:
            final_score = entity_verb_score
        else:
            final_score = keyword_matches * 1.0  # Use keyword matches if no entity/verb matches
        
        scored_evidence.append((evidence, final_score))
    
    # Sort by score (descending)
    scored_evidence.sort(key=lambda x: x[1], reverse=True)
    
    return scored_evidence

def get_recent_date_range(claim=None):
    """
    Return date range for news filtering based on temporal indicators in the claim.
    
    Args:
        claim (str, optional): The claim text to analyze for temporal indicators
        
    Returns:
        tuple: (from_date, to_date) as formatted strings 'YYYY-MM-DD'
    """
    today = datetime.now()
    
    # Default to 3 days for no claim or claims without temporal indicators
    default_days = 3
    extended_days = 15  # For 'recently', 'this week', etc.
    
    if claim:
        # Specific day indicators get 3 days
        specific_day_terms = ["today", "yesterday", "day before yesterday"]
        
        # Extended time terms get 15 days
        extended_time_terms = [
            "recently", "currently", "freshly", "this week", "few days", 
            "couple of days", "last week", "past week", "several days",
            "anymore"
        ]
        
        claim_lower = claim.lower()
        
        # Check for extended time terms first, then specific day terms
        if any(term in claim_lower for term in extended_time_terms):
            from_date = (today - timedelta(days=extended_days)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            logger.info(f"Using extended time range of {extended_days} days based on temporal indicators")
            return from_date, to_date
        elif any(term in claim_lower for term in specific_day_terms):
            from_date = (today - timedelta(days=default_days)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            logger.info(f"Using specific day range of {default_days} days based on temporal indicators")
            return from_date, to_date
    
    # Default case - use standard 3-day window
    from_date = (today - timedelta(days=default_days)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    return from_date, to_date

@api_error_handler("wikipedia")
def retrieve_evidence_from_wikipedia(claim):
    """Retrieve evidence from Wikipedia for a given claim"""
    logger.info(f"Retrieving evidence from Wikipedia for: {claim}")

    # Ensure shortened_claim is a string
    try:
        shortened_claim = shorten_claim_for_evidence(claim)
    except Exception as e:
        logger.error(f"Error in claim shortening: {e}")
        shortened_claim = claim  # Fallback to original claim

    # Ensure query_parts is a list of strings
    query_parts = str(shortened_claim).split()
    evidence = []
    source_count = {"wikipedia": 0}

    for i in range(len(query_parts), 0, -1):  # Start with full query, shorten iteratively
        try:
            # Safely join and encode query
            current_query = "+".join(query_parts[:i])
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={current_query}&format=json"
            logger.info(f"Wikipedia search URL: {search_url}")

            headers = {
                "User-Agent": "MisinformationDetectionResearchBot/1.0 (Research Project)"
            }

            # Make the search request with reduced timeout
            response = requests.get(search_url, headers=headers, timeout=7)
            response.raise_for_status()

            # Safely parse JSON
            search_data = safe_json_parse(response, "wikipedia")

            # Safely extract search results
            search_results = search_data.get("query", {}).get("search", [])

            # Ensure search_results is a list
            if not isinstance(search_results, list):
                logger.warning(f"Unexpected search results type: {type(search_results)}")
                search_results = []

            # Use ThreadPoolExecutor to fetch page content in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit up to 3 page requests in parallel
                futures = []
                for idx, result in enumerate(search_results[:3]):
                    # Ensure result is a dictionary
                    if not isinstance(result, dict):
                        logger.warning(f"Skipping non-dictionary result: {type(result)}")
                        continue

                    # Safely extract title
                    page_title = result.get("title", "")
                    if not page_title:
                        continue

                    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                    
                    # Submit the page request task to executor
                    futures.append(executor.submit(
                        fetch_wikipedia_page_content, 
                        page_url, 
                        page_title, 
                        headers
                    ))
                
                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        page_result = future.result()
                        if page_result:
                            evidence.append(page_result)
                            source_count["wikipedia"] += 1
                    except Exception as e:
                        logger.error(f"Error processing Wikipedia page: {e}")

            # Stop if we found any evidence
            if evidence:
                break

        except Exception as e:
            logger.error(f"Error retrieving from Wikipedia: {str(e)}")
            continue

    # Ensure success is a boolean
    success = bool(evidence)

    # Safely log evidence retrieval
    try:
        performance_tracker.log_evidence_retrieval(success, source_count)
    except Exception as e:
        logger.error(f"Error logging evidence retrieval: {e}")

    if not evidence:
        logger.warning("No evidence found from Wikipedia.")

    return evidence

def fetch_wikipedia_page_content(page_url, page_title, headers):
    """Helper function to fetch and parse Wikipedia page content"""
    try:
        # Get page content with reduced timeout
        page_response = requests.get(page_url, headers=headers, timeout=5)
        page_response.raise_for_status()

        # Extract relevant sections using BeautifulSoup
        soup = BeautifulSoup(page_response.text, 'html.parser')
        paragraphs = soup.find_all('p', limit=3)  # Limit to first 3 paragraphs
        content = " ".join([para.get_text(strip=True) for para in paragraphs])
        
        # Truncate content to reduce token usage earlier in the pipeline
        if len(content) > 1000:
            content = content[:1000] + "..."

        if content.strip():  # Ensure content is not empty
            return f"Title: {page_title}, URL: {page_url}, Content: {content}"
        return None
    except Exception as e:
        logger.error(f"Error fetching Wikipedia page {page_url}: {e}")
        return None

@api_error_handler("wikidata")
def retrieve_evidence_from_wikidata(claim):
    """Retrieve evidence from Wikidata for a given claim"""
    logger.info(f"Retrieving evidence from Wikidata for: {claim}")

    # Prepare entities for SPARQL query
    shortened_claim = shorten_claim_for_evidence(claim)
    query_terms = shortened_claim.split()

    # Initialize SPARQLWrapper for Wikidata
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Use a more conservative user agent to avoid blocks
    sparql.addCustomHttpHeader("User-Agent", "MisinformationDetectionResearchBot/1.0")
    
    # Fix SSL issues by disabling SSL verification for this specific request
    try:        
        # Create a context that doesn't verify certificates
        ssl_context = ssl._create_unverified_context()
        
        # Monkey patch the opener for SPARQLWrapper
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
    except Exception as e:
        logger.error(f"Error setting up SSL context: {str(e)}")

    # Construct basic SPARQL query for relevant entities
    query = """
    SELECT ?item ?itemLabel ?description ?article WHERE {
      SERVICE wikibase:mwapi {
        bd:serviceParam wikibase:api "EntitySearch" .
        bd:serviceParam wikibase:endpoint "www.wikidata.org" .
        bd:serviceParam mwapi:search "%s" .
        bd:serviceParam mwapi:language "en" .
        ?item wikibase:apiOutputItem mwapi:item .
      }
      ?item schema:description ?description .
      FILTER(LANG(?description) = "en")
      OPTIONAL {
        ?article schema:about ?item .
        ?article schema:isPartOf <https://en.wikipedia.org/> .
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 5
    """ % " ".join(query_terms)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

        wikidata_evidence = []

        for result in results["results"]["bindings"]:
            entity_label = result.get("itemLabel", {}).get("value", "Unknown")
            description = result.get("description", {}).get("value", "No description")
            article_url = result.get("article", {}).get("value", "")
            
            # Truncate description to reduce token usage
            if len(description) > 1000:
                description = description[:1000] + "..."

            evidence_text = f"Entity: {entity_label}, Description: {description}"
            if article_url:
                evidence_text += f", URL: {article_url}"

            wikidata_evidence.append(evidence_text)

        logger.info(f"Retrieved {len(wikidata_evidence)} Wikidata entities")
        
        # Log evidence retrieval performance
        success = bool(wikidata_evidence)
        source_count = {"wikidata": len(wikidata_evidence)}
        try:
            performance_tracker.log_evidence_retrieval(success, source_count)
        except Exception as e:
            logger.error(f"Error logging Wikidata evidence retrieval: {e}")
        
        return wikidata_evidence

    except Exception as e:
        logger.error(f"Error retrieving from Wikidata: {str(e)}")
        
        # Log failed evidence retrieval
        try:
            performance_tracker.log_evidence_retrieval(False, {"wikidata": 0})
        except Exception as log_error:
            logger.error(f"Error logging failed Wikidata evidence retrieval: {log_error}")
        
        return []

@api_error_handler("openalex")
def retrieve_evidence_from_openalex(claim):
    """Retrieve evidence from OpenAlex for a given claim (replacement for Semantic Scholar)"""
    logger.info(f"Retrieving evidence from OpenAlex for: {claim}")

    try:
        shortened_claim = shorten_claim_for_evidence(claim)
        query = shortened_claim.replace(" ", "+")
        
        # OpenAlex API endpoint
        api_url = f"https://api.openalex.org/works?search={query}&filter=is_paratext:false&per_page=3"
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "MisinformationDetectionResearchBot/1.0 (research.project@example.edu)",
        }
        
        scholarly_evidence = []
        
        try:
            # Request with reduced timeout
            response = requests.get(api_url, headers=headers, timeout=8)
            
            # Check response status
            if response.status_code == 200:
                # Successfully retrieved data
                data = safe_json_parse(response, "openalex")
                papers = data.get("results", [])

                for paper in papers:
                    title = paper.get("title", "Unknown Title")
                    abstract = paper.get("abstract_inverted_index", None)
                    
                    # OpenAlex stores abstracts in an inverted index format, so we need to reconstruct it
                    abstract_text = "No abstract available"
                    if abstract:
                        try:
                            # Simple approach to reconstruct from inverted index
                            # For a production app, implement a proper reconstruction algorithm
                            words = list(abstract.keys())
                            abstract_text = " ".join(words[:30]) + "..."
                        except Exception as e:
                            logger.error(f"Error reconstructing abstract: {e}")
                    
                    url = paper.get("doi", "")
                    if url and not url.startswith("http"):
                        url = f"https://doi.org/{url}"
                    
                    year = ""
                    publication_date = paper.get("publication_date", "")
                    if publication_date:
                        year = publication_date.split("-")[0]
                    
                    # Truncate abstract to reasonable length
                    if len(abstract_text) > 1000:
                        abstract_text = abstract_text[:1000] + "..."

                    evidence_text = f"Title: {title}, Year: {year}, Abstract: {abstract_text}, URL: {url}"
                    scholarly_evidence.append(evidence_text)

            else:
                logger.error(f"OpenAlex API error: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("OpenAlex request timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("OpenAlex connection error")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAlex request: {str(e)}")

        logger.info(f"Retrieved {len(scholarly_evidence)} scholarly papers from OpenAlex")
       
        # Log evidence retrieval performance
        success = bool(scholarly_evidence)
        source_count = {"openalex": len(scholarly_evidence)}
        try:
            performance_tracker.log_evidence_retrieval(success, source_count)
        except Exception as e:
            logger.error(f"Error logging OpenAlex evidence retrieval: {e}")

        return scholarly_evidence

    except Exception as e:
        logger.error(f"Fatal error in OpenAlex retrieval: {str(e)}")
        
        # Log failed evidence retrieval
        try:
            performance_tracker.log_evidence_retrieval(False, {"openalex": 0})
        except Exception as log_error:
            logger.error(f"Error logging failed OpenAlex evidence retrieval: {log_error}")
        
        return []

@api_error_handler("factcheck")
def retrieve_evidence_from_factcheck(claim):
    """Retrieve evidence from Google's Fact Check Tools API for a given claim"""
    logger.info(f"Retrieving evidence from Google's Fact Check Tools API for: {claim}")
    factcheck_api_key = FACTCHECK_API_KEY

    # Safely shorten claim
    try:
        shortened_claim = shorten_claim_for_evidence(claim)
    except Exception as e:
        logger.error(f"Error shortening claim: {e}")
        shortened_claim = claim

    query_parts = str(shortened_claim).split()
    factcheck_results = []
    source_count = {"factcheck": 0}

    for i in range(len(query_parts), 0, -1):  # Iteratively try shorter queries
        try:
            current_query = " ".join(query_parts[:i])
            encoded_query = urlencode({"query": current_query})
            factcheck_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{encoded_query}&key={factcheck_api_key}"
            logger.info(f"Factcheck URL: {factcheck_url}")

            # Make request with reduced timeout
            response = requests.get(factcheck_url, timeout=7)
            response.raise_for_status()
            data = safe_json_parse(response, "factcheck")

            # Safely extract claims
            claims = data.get("claims", [])
            if not isinstance(claims, list):
                logger.warning(f"Unexpected claims type: {type(claims)}")
                claims = []

            if claims:  # If results found
                logger.info(f"Results found for query '{current_query}'.")
                for item in claims:
                    try:
                        # Ensure item is a dictionary
                        if not isinstance(item, dict):
                            logger.warning(f"Skipping non-dictionary item: {type(item)}")
                            continue

                        claim_text = str(item.get("text", ""))
                        # Truncate claim text
                        if len(claim_text) > 1000:
                            claim_text = claim_text[:1000] + "..."
                            
                        reviews = item.get("claimReview", [])

                        # Ensure reviews is a list
                        if not isinstance(reviews, list):
                            logger.warning(f"Unexpected reviews type: {type(reviews)}")
                            reviews = []

                        for review in reviews:
                            # Ensure review is a dictionary
                            if not isinstance(review, dict):
                                logger.warning(f"Skipping non-dictionary review: {type(review)}")
                                continue

                            publisher = str(review.get("publisher", {}).get("name", "Unknown Source"))
                            rating = str(review.get("textualRating", "Unknown"))
                            review_url = str(review.get("url", ""))

                            if claim_text:
                                factcheck_results.append(
                                    f"Claim: {claim_text}, Rating: {rating}, " +
                                    f"Source: {publisher}, URL: {review_url}"
                                )
                                source_count["factcheck"] += 1

                    except Exception as e:
                        logger.error(f"Error processing FactCheck result: {e}")

                break  # Break once we have results
            else:
                logger.info(f"No results for query '{current_query}', trying shorter version.")

        except Exception as e:
            logger.error(f"Error in FactCheck retrieval: {e}")

    # Safely log evidence retrieval
    try:
        success = bool(factcheck_results)
        performance_tracker.log_evidence_retrieval(success, source_count)
    except Exception as e:
        logger.error(f"Error logging evidence retrieval: {e}")

    if not factcheck_results:
        logger.warning("No factcheck evidence found after trying all query variants.")

    return factcheck_results

@api_error_handler("newsapi")
def retrieve_news_articles(claim, requires_recent=False):
    """Retrieve evidence from News API for a given claim with improved single request approach"""
    logger.info(f"Retrieving evidence from News API for: {claim}")

    # Get API key
    news_api_key = NEWS_API_KEY
    if not news_api_key:
        logger.error("No News API key available")
        return []

    news_results = []
    source_count = {"news": 0}

    # Get date range for recent news
    from_date, to_date = get_recent_date_range()
    logger.info(f"Filtering for news from {from_date} to {to_date}")

    try:
        # Extract a simplified claim for better matching
        shortened_claim = shorten_claim_for_evidence(claim)
        
        # Use a single endpoint with proper parameters
        encoded_query = urlencode({"q": shortened_claim})
        
        # Use the 'everything' endpoint as it's more comprehensive
        news_api_url = f"https://newsapi.org/v2/everything?{encoded_query}&apiKey={news_api_key}&language=en&pageSize=5&sortBy=publishedAt"
        
        # Only apply date filtering if the claim requires recency
        if requires_recent:
            news_api_url += f"&from={from_date}&to={to_date}"
        
        log_url = news_api_url.replace(news_api_key, "API_KEY_REDACTED")
        logger.info(f"Requesting: {log_url}")

        # Make a single request with proper headers and reduced timeout
        headers = {
            "User-Agent": "MisinformationDetectionResearchBot/1.0",
            "X-Api-Key": news_api_key,
            "Accept": "application/json"
        }

        response = requests.get(
            news_api_url,
            headers=headers,
            timeout=8
        )

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = safe_json_parse(response, "newsapi")
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                logger.info(f"Found {len(articles)} articles")
                
                for article in articles:
                    try:
                        # Robust article parsing
                        title = str(article.get("title", ""))
                        description = str(article.get("description", ""))
                        content = str(article.get("content", ""))
                        source_name = str(article.get("source", {}).get("name", "Unknown"))
                        url = str(article.get("url", ""))
                        published_at = str(article.get("publishedAt", ""))
                        
                        # Parse date to prioritize recent content
                        article_date = None
                        try:
                            if published_at:
                                article_date = datetime.strptime(published_at.split('T')[0], '%Y-%m-%d')
                        except Exception as date_error:
                            logger.warning(f"Could not parse date: {published_at}")
                        
                        # Calculate recency score (higher = more recent)
                        recency_score = 1.0  # Default
                        if article_date:
                            days_old = (datetime.now() - article_date).days
                            if days_old == 0:  # Today
                                recency_score = 3.0
                            elif days_old == 1:  # Yesterday
                                recency_score = 2.0
                        
                        # Use description if content is empty or too short
                        if not content or len(content) < 50:
                            content = description
                        
                        # Truncate content to reduce token usage
                        if len(content) > 1000:
                            content = content[:1000] + "..."

                        # Ensure meaningful content
                        if title and (content or description):
                            news_item = {
                                "text": (
                                    f"Title: {title}, " +
                                    f"Source: {source_name}, " +
                                    f"Date: {published_at}, " +
                                    f"URL: {url}, " +
                                    f"Content: {content}"
                                ),
                                "recency_score": recency_score,
                                "date": article_date
                            }
                            news_results.append(news_item)
                            source_count["news"] += 1
                            logger.info(f"Added article: {title}")

                    except Exception as article_error:
                        logger.error(f"Error processing article: {article_error}")
                
                # Sort results by recency
                if news_results:
                    news_results.sort(key=lambda x: x.get('recency_score', 0), reverse=True)
    
    except Exception as query_error:
        logger.error(f"Error processing query: {query_error}")

    # Convert to plain text list for compatibility with existing code
    news_texts = [item["text"] for item in news_results]

    # Log evidence retrieval
    success = bool(news_texts)
    source_count = {"news": len(news_texts)}
    try:
        performance_tracker.log_evidence_retrieval(success, source_count)
    except Exception as log_error:
        logger.error(f"Error logging evidence retrieval: {log_error}")

    # Log results
    if news_texts:
        logger.info(f"Retrieved {len(news_texts)} news articles")
    else:
        logger.warning("No news articles found")

    return news_texts

def retrieve_combined_evidence(claim):
    """
    Retrieve evidence from multiple sources in parallel and analyze relevance.
    
    This function:
    1. Extracts claim components (entities, verbs, keywords)
    2. Determines if the claim is temporal
    3. Retrieves evidence from all sources in parallel
    4. Analyzes relevance based on entity and verb matching
    5. Returns the most relevant evidence items for claim verification
    
    Args:
        claim (str): The factual claim to gather evidence for
        
    Returns:
        list: List of the most relevant evidence items (max 5) for claim verification
    """
    logger.info(f"Starting evidence retrieval for: {claim}")
    start_time = time.time()

    # Extract key claim components for relevance matching
    claim_components = extract_claim_components(claim)
    logger.info(f"Extracted claim components: entities={claim_components.get('entities', [])}, verbs={claim_components.get('verbs', [])}")
    
    # Determine if claim has temporal attributes
    requires_recent_evidence = bool(claim_components.get("temporal_words", []))
    logger.info(f"Claim requires recent evidence: {requires_recent_evidence}")
    
    # Determine the claim category
    category, confidence = detect_claim_category(claim)
    logger.info(f"Detected claim category: {category} (confidence: {confidence:.2f})")
    
    # Initialize results container
    all_evidence = []
    source_counts = {}
    
    # Define all evidence sources to query in parallel
    evidence_sources = [
        ("wikipedia", retrieve_evidence_from_wikipedia, [claim]),
        ("wikidata", retrieve_evidence_from_wikidata, [claim]),
        ("scholarly", retrieve_evidence_from_openalex, [claim]),
        ("claimreview", retrieve_evidence_from_factcheck, [claim]),
        ("news", retrieve_news_articles, [claim, requires_recent_evidence])
    ]
    
    # Add RSS feeds based on category with appropriate fallback
    if category == "ai":
        # For AI category, add AI-specific RSS feeds
        category_feeds = get_category_specific_rss_feeds(category)
        evidence_sources.append(("rss_ai", retrieve_evidence_from_rss, [claim, 10, category_feeds]))
        
        # Add technology fallback feeds for AI
        fallback_category = get_fallback_category(category)  # Should be "technology"
        if fallback_category:
            fallback_feeds = get_category_specific_rss_feeds(fallback_category)
            evidence_sources.append(("rss_tech", retrieve_evidence_from_rss, [claim, 10, fallback_feeds]))
    else:
        # For other categories, add their specific RSS feeds
        category_feeds = get_category_specific_rss_feeds(category)
        if category_feeds:
            evidence_sources.append(("rss_category", retrieve_evidence_from_rss, [claim, 10, category_feeds]))
        
        # Add default RSS feeds as fallback for all non-AI categories
        evidence_sources.append(("rss_default", retrieve_evidence_from_rss, [claim, 10]))
    
    # Execute all evidence gathering in parallel
    with ThreadPoolExecutor(max_workers=len(evidence_sources)) as executor:
        # Create a mapping of futures to source names for easier tracking
        futures = {}
        for source_name, func, args in evidence_sources:
            future = executor.submit(func, *args)
            futures[future] = source_name
        
        # Process results as they complete
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                evidence_items = future.result()
                if evidence_items:
                    all_evidence.extend(evidence_items)
                    source_counts[source_name] = len(evidence_items)
                    logger.info(f"Retrieved {len(evidence_items)} items from {source_name}")
            except Exception as e:
                logger.error(f"Error retrieving from {source_name}: {str(e)}")
    
    # If no evidence was found at all, create a minimal placeholder
    if not all_evidence:
        logger.warning("No evidence found from any source")
        return [f"No specific evidence found for the claim: '{claim}'. This may be due to the claim being very recent, niche, or involving private information."]
    
    # Analyze evidence relevance
    scored_evidence = analyze_evidence_relevance(all_evidence, claim_components)
    
    # Return top 10 most relevant evidence items
    return [evidence for evidence, score in scored_evidence[:10]]