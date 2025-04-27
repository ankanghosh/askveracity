# AskVeracity Architecture

## System Overview

AskVeracity is a fact-checking and misinformation detection application that verifies factual claims by gathering and analyzing evidence from multiple sources. The system follows an agentic approach using LangGraph's ReAct agent framework for orchestrating the verification process.

## Core Components

### 1. Agent System

The system implements a LangGraph-based agent that orchestrates the entire fact-checking process:

- **Core Agent:** Defined in `agent.py`, the ReAct agent coordinates the execution of individual tools in a logical sequence to verify claims.
- **Agent Tools:** Implemented as callable functions that the agent can invoke:
  - `claim_extractor`: Extracts the main factual claim from user input
  - `evidence_retriever`: Gathers evidence from multiple sources
  - `truth_classifier`: Evaluates the claim against evidence
  - `explanation_generator`: Creates human-readable explanations

### 2. Web Interface

The user interface is implemented using Streamlit:

- **Main App:** Defined in `app.py`, provides the interface for users to submit claims and view results
- **Caching:** Uses Streamlit's caching mechanisms to optimize performance
- **Results Display:** Shows verdict, confidence, explanation, and evidence details

## Module Architecture

```
askveracity/
│
├── agent.py                   # LangGraph agent implementation
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration and API keys
├── evaluate_performance.py    # Performance evaluation script
│
├── modules/                   # Core functionality modules
│   ├── claim_extraction.py    # Claim extraction functionality  
│   ├── evidence_retrieval.py  # Evidence gathering from various sources
│   ├── classification.py      # Truth classification logic
│   ├── explanation.py         # Explanation generation
│   ├── rss_feed.py            # RSS feed evidence retrieval
│   └── category_detection.py  # Claim category detection
│
├── utils/                     # Utility functions
│   ├── api_utils.py           # API rate limiting and error handling
│   ├── performance.py         # Performance tracking utilities
│   └── models.py              # Model initialization functions
│
├── results/                   # Performance evaluation results
│   ├── performance_results.json # Evaluation metrics
│   └── *.png                  # Performance visualization charts
│
└── docs/ # Documentation
   ├── assets/ # Images and other media
   │   └── app_screenshot.png # Application screenshot
   ├── architecture.md # System design and component interactions
   ├── configuration.md # Setup and environment configuration
   ├── data-handling.md # Data processing and flow
   └── changelog.md # Version history
```

## Component Interactions

### Claim Verification Flow

1. **User Input:** User submits a claim via the Streamlit interface
2. **Agent Initialization:** The ReAct agent is initialized with fact-checking tools
3. **Claim Extraction:** The agent extracts the main factual claim
4. **Category Detection:** The system detects the category of the claim (ai, science, technology, politics, business, world, sports, entertainment)
5. **Evidence Retrieval:** Multi-source evidence gathering with priority based on claim category
6. **Evidence Analysis:** Entity and verb matching assesses evidence relevance
7. **Truthfulness Classification:** The agent evaluates the claim against the evidence
8. **Explanation Generation:** Human-readable explanation is generated
9. **Results Display:** Results are presented to the user with evidence details

### Evidence Retrieval Architecture

Evidence retrieval is a core component of the misinformation detection system:

1. **Multi-source Retrieval:** The system collects evidence from:
   - Wikipedia
   - Wikidata
   - News API
   - RSS feeds
   - Fact-checking sites (via Google Fact Check Tools API)
   - Academic sources (via OpenAlex)

2. **Category-aware Prioritization:** Sources are prioritized based on the detected category of the claim:
   - Each category (ai, science, technology, politics, business, world, sports, entertainment) has dedicated RSS feeds
   - AI category falls back to technology sources when needed
   - Other categories fall back to default RSS feeds

3. **Parallel Processing:** Evidence retrieval uses ThreadPoolExecutor for parallel API requests with optimized timeouts

4. **Rate Limiting:** API calls are managed by a token bucket rate limiter to respect API usage limits

5. **Error Handling:** Robust error handling with exponential backoff for retries

6. **Source Verification:** The system provides direct URLs to original sources for all evidence items, enabling users to verify information at its original source


### Classification System

The truth classification process involves:

1. **Evidence Analysis:** Each evidence item is classified as supporting, contradicting, or insufficient
2. **Confidence Scoring:** Confidence scores are assigned to each classification
3. **Aggregation:** Individual evidence classifications are aggregated to determine the final verdict

## Technical Details

### Language Models

- Uses OpenAI's Large Language Model GPT-3.5 Turbo via LangChain
- Configurable model selection in `utils/models.py`

### NLP Processing

- spaCy for natural language processing tasks
- Named entity recognition for claim and evidence analysis
- Entity and verb matching for evidence relevance scoring

### Performance Optimization

- Caching of models and results
- Prioritized and parallel evidence retrieval
- Early relevance analysis during retrieval process

### Error Resilience

- Multiple fallback mechanisms
- Graceful degradation when sources are unavailable
- Comprehensive error logging

## Performance Evaluation Results

The system has been evaluated using a test set of 40 claims across three categories (True, False, and Uncertain). A typical performance profile shows:

1. **Overall Accuracy:** ~52.5% across all claim types
   * Accuracy: Percentage of claims correctly classified according to their ground truth label

2. **Safety Rate:** ~70.0% across all claim types
   * Safety Rate: Percentage of claims that were either correctly classified or safely categorized as "Uncertain" rather than making an incorrect assertion

3. **Class-specific Metrics:**
   * True claims: ~40-60% accuracy, ~55-85% safety rate
   * False claims: ~15-35% accuracy, ~50-70% safety rate
   * Uncertain claims: ~50.0% accuracy, ~50.0% safety rate (for Uncertain claims, accuracy equals safety rate)

4. **Confidence Scores:**
   * True claims: ~0.62-0.74 average confidence
   * False claims: ~0.42-0.50 average confidence
   * Uncertain claims: ~0.38-0.50 average confidence

5. **Processing Times:**
   * True claims: ~21-32 seconds average
   * False claims: ~24-37 seconds average
   * Uncertain claims: ~23-31 seconds average

**Note:** The class-specific metrics, confidence scores, and processing times vary by test run.

These metrics vary between evaluation runs due to the dynamic nature of evidence sources and the real-time information landscape. The system is designed to adapt to this variability, making it well-suited for real-world fact-checking scenarios where information evolves over time.

## Misinformation Detection Capabilities

The system's approach to detecting misinformation includes:

1. **Temporal Relevance:** Checks if evidence is temporally appropriate for the claim
2. **Contradiction Detection:** Identifies evidence that directly contradicts claims
3. **Evidence Diversity:** Ensures diverse evidence sources for more robust verification
4. **Domain Prioritization:** Applies a small relevance boost to content from established news and fact-checking domains in the RSS feed handling
5. **Safety-First Classification:** Prioritizes preventing the spread of misinformation by avoiding incorrect assertions when evidence is insufficient

This architecture enables AskVeracity to efficiently gather, analyze, and present evidence relevant to user claims, supporting the broader effort to detect and counteract misinformation.