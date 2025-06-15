# Data Handling in AskVeracity

This document explains how data flows through the AskVeracity fact-checking and misinformation detection system, from user input to final verification results.

## Data Flow Overview

```
User Input → Claim Extraction → Category Detection → Evidence Retrieval → Evidence Analysis → Classification → Explanation → Result Display
```

## User Input Processing

### Input Sanitization and Extraction

1. **Input Acceptance:** The system accepts user input as free-form text through the Streamlit interface.

2. **Claim Extraction** (`modules/claim_extraction.py`):
   - For concise inputs (<30 words), the system preserves the input as-is
   - For longer texts, an LLM extracts the main factual claim
   - Validation ensures the extraction doesn't add information not present in the original
   - Entity preservation is verified using spaCy's NER

3. **Claim Shortening:**
   - For evidence retrieval, claims are shortened to preserve key entities and context
   - Preserves entity mentions, key nouns, titles, country references, and negation contexts

## Evidence Retrieval and Processing

### Multi-source Evidence Gathering

Evidence is collected from multiple sources in parallel (`modules/evidence_retrieval.py`):

1. **Category Detection** (`modules/category_detection.py`):
   - Detects the claim category (ai, science, technology, politics, business, world, sports, entertainment)
   - Prioritizes sources based on category
   - No category receives preferential weighting; assignment is based purely on keyword matching

2. **Wikipedia** evidence:
   - Search Wikipedia API for relevant articles
   - Extract introductory paragraphs
   - Process in parallel for up to 3 top search results

3. **Wikidata** evidence:
   - SPARQL queries for structured data
   - Entity extraction with descriptions

4. **News API** evidence:
   - Retrieval from NewsAPI.org with date filtering
   - Prioritizes recent articles
   - Extracts titles, descriptions, and content snippets

5. **RSS Feed** evidence (`modules/rss_feed.py`):
   - Parallel retrieval from multiple RSS feeds
   - Category-specific feeds selection
   - Relevance and recency scoring

6. **ClaimReview** evidence:
   - Google's Fact Check Tools API integration
   - Retrieves fact-checks from fact-checking organizations
   - Includes ratings and publisher information

7. **Scholarly** evidence:
   - OpenAlex API for academic sources
   - Extracts titles, abstracts, and publication dates

8. **Category Fallback** mechanism:
   - For AI claims, uses both AI-specific and technology RSS feeds simultaneously
   - For other categories, falls back to default RSS feeds
   - Ensures robust evidence retrieval across related domains

### Evidence Preprocessing

Each evidence item is standardized to a consistent format:
```
Title: [title], Source: [source], Date: [date], URL: [url], Content: [content snippet]
```

Length limits are applied to reduce token usage:
- Content snippets are limited to ~1000 characters
- Evidence items are truncated while maintaining context

## Evidence Analysis and Relevance Ranking

### Relevance Assessment

Evidence is analyzed and scored for relevance:

1. **Component Extraction:**
   - Extract entities, verbs, and keywords from the claim
   - Use NLP processing to identify key claim components

2. **Entity and Verb Matching:**
   - Match entities from claim to evidence (case-sensitive and case-insensitive)
   - Match verbs from claim to evidence
   - Score based on matches (entity matches weighted higher than verb matches)

3. **Temporal Relevance:**
   - Detection of temporal indicators in claims
   - Date-based filtering for time-sensitive claims
   - Adjusts evidence retrieval window based on claim temporal context

4. **Scoring Formula:**
   ```
   final_score = (entity_matches * 3.0) + (verb_matches * 2.0)
   ```
   If no entity or verb matches, fall back to keyword matching:
   ```
   final_score = keyword_matches * 1.0
   ```

### Evidence Selection

The system selects the most relevant evidence:

1. **Relevance Sorting:**
   - Evidence items sorted by relevance score (descending)
   - Top 10 most relevant items selected

2. **Handling No Evidence:**
   - If no evidence is found, a placeholder is returned
   - Ensures graceful handling of edge cases

## Truth Classification

### Evidence Classification (`modules/classification.py`)

Each evidence item is classified individually:

1. **LLM Classification:**
   - Each evidence item is analyzed by an LLM
   - Classification categories: support, contradict, insufficient
   - Confidence score (0-100) assigned to each classification
   - Structured output parsing with fallback mechanisms

2. **Tense Normalization:**
   - Normalizes verb tenses in claims to ensure consistent classification
   - Converts present simple and perfect forms to past tense equivalents
   - Preserves semantic equivalence across tense variations

### Verdict Aggregation

Evidence classifications are aggregated to determine the final verdict:

1. **Weighted Aggregation:**
   - 55% weight for count of support/contradict items
   - 45% weight for quality (confidence) of support/contradict items

2. **Confidence Calculation:**
   - Formula: `1.0 - (min_score / max_score)`
   - Higher confidence for consistent evidence
   - Lower confidence for mixed or insufficient evidence

3. **Final Verdict Categories:**
   - "True (Based on Evidence)"
   - "False (Based on Evidence)"
   - "Uncertain"

## Explanation Generation

### Explanation Creation (`modules/explanation.py`)

Human-readable explanations are generated based on the verdict:

1. **Template Selection:**
   - Different prompts for true, false, and uncertain verdicts
   - Special handling for claims containing negation

2. **Confidence Communication:**
   - Translation of confidence scores to descriptive language
   - Clear communication of certainty/uncertainty

3. **Very Low Confidence Handling:**
   - Special explanations for verdicts with very low confidence (<10%)
   - Strong recommendations to verify with authoritative sources

## Result Presentation

Results are presented in the Streamlit UI with multiple components:

1. **Verdict Display:**
   - Color-coded verdict (green for true, red for false, gray for uncertain)
   - Confidence percentage
   - Explanation text

2. **Evidence Presentation:**
   - Tabbed interface for different evidence views with URLs if available
   - Supporting and contradicting evidence tabs
   - Source distribution summary

3. **Input Guidance:**
   - Tips for claim formatting
   - Guidance for time-sensitive claims
   - Suggestions for verb tense based on claim age

4. **Processing Insights:**
   - Processing time
   - AI reasoning steps
   - Source distribution statistics

## Data Persistence and Privacy

AskVeracity prioritizes user privacy:

1. **No Data Storage:**
   - User claims are not stored persistently
   - Results are maintained only in session state
   - No user data is collected or retained

2. **Session Management:**
   - Session state in Streamlit manages current user interaction
   - Session is cleared when starting a new verification

3. **API Interaction:**
   - External API calls use their respective privacy policies
   - OpenAI API usage follows their data handling practices

4. **Caching:**
   - Model caching for performance
   - Resource cleanup on application termination

## Performance Tracking

The system includes a performance tracking utility (`utils/performance.py`):

1. **Metrics Tracked:**
   - Claims processed count
   - Evidence retrieval success rates
   - Processing times
   - Confidence scores
   - Source types used
   - Temporal relevance

2. **Usage:**
   - Performance metrics are logged during processing
   - Summary of select metrics available in the final result
   - Used for system optimization

## Performance Evaluation

The system includes a performance evaluation script (`evaluate_performance.py`):

1. **Test Claims:**
   - Predefined set of test claims with known ground truth labels
   - Claims categorized as "True", "False", or "Uncertain"

2. **Metrics:**
   - Overall accuracy: Percentage of claims correctly classified according to ground truth
   - Safety rate: Percentage of claims either correctly classified or safely categorized as "Uncertain" rather than making an incorrect assertion
   - Per-class accuracy and safety rates
   - Average processing time
   - Average confidence score
   - Classification distributions

3. **Visualization:**
   - Charts for accuracy by classification type
   - Charts for safety rate by classification type
   - Processing time by classification type
   - Confidence scores by classification type

4. **Results Storage:**
   - Detailed results saved to JSON file
   - Visualization charts saved as PNG files
   - All results stored in the `results/` directory

## Error Handling and Resilience

The system implements robust error handling:

1. **API Error Handling** (`utils/api_utils.py`):
   - Decorator-based error handling
   - Exponential backoff for retries
   - Rate limiting respecting API constraints

2. **Safe JSON Parsing:**
   - Defensive parsing of API responses
   - Fallback mechanisms for invalid responses

3. **Graceful Degradation:**
   - Multiple fallback strategies
   - Core functionality preservation even when some sources fail

4. **Fallback Mechanisms:**
   - Fallback for truth classification when classifier is not called
   - Fallback for explanation generation when explanation generator is not called
   - Ensures complete results even with partial component failures