# AskVeracity Configuration Guide

This document describes how to set up and configure the AskVeracity fact-checking and misinformation detection system.

## Prerequisites

Before setting up AskVeracity, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- API keys for external services

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/askveracity.git
   cd askveracity
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## API Key Configuration

AskVeracity requires several API keys to access external services. You have two options for configuring these keys:

### Option 1: Using Streamlit Secrets (Recommended for Local Development)

1. Create a `.streamlit` directory if it doesn't exist:
   ```bash
   mkdir -p .streamlit
   ```

2. Create a `secrets.toml` file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

3. Edit the `.streamlit/secrets.toml` file with your API keys:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   NEWS_API_KEY = "your_news_api_key"
   FACTCHECK_API_KEY = "your_factcheck_api_key"
   ```

### Option 2: Using Environment Variables

1. Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```

2. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEWS_API_KEY=your_news_api_key
   FACTCHECK_API_KEY=your_factcheck_api_key
   ```

3. Load the environment variables:
   ```python
   # In Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

   Or in your terminal:
   ```bash
   # Unix/Linux/MacOS
   source .env
   
   # Windows
   # Install python-dotenv[cli] and run
   dotenv run streamlit run app.py
   ```

## Required API Keys

AskVeracity uses the following external APIs:

1. **OpenAI API** (Required)
   - Used for claim extraction, classification, and explanation generation
   - Get an API key from [OpenAI's website](https://platform.openai.com/)

2. **News API** (Optional but recommended)
   - Used for retrieving news article evidence
   - Get an API key from [NewsAPI.org](https://newsapi.org/)

3. **Google Fact Check Tools API** (Optional but recommended)
   - Used for retrieving fact-checking evidence
   - Get an API key from [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api)

## Configuration Files

### config.py

The main configuration file is `config.py`, which contains:

- API key handling
- Rate limiting configuration
- Error backoff settings
- RSS feed settings

Important configuration sections in `config.py`:

```python
# Rate limiting configuration
RATE_LIMITS = {
    # api_name: {"requests": max_requests, "period": period_in_seconds}
    "newsapi": {"requests": 100, "period": 3600},  # 100 requests per hour
    "factcheck": {"requests": 1000, "period": 86400},  # 1000 requests per day
    "semantic_scholar": {"requests": 10, "period": 300},  # 10 requests per 5 minutes
    "wikidata": {"requests": 60, "period": 60},  # 60 requests per minute
    "wikipedia": {"requests": 200, "period": 60},  # 200 requests per minute
    "rss": {"requests": 300, "period": 3600}  # 300 RSS requests per hour
}

# Error backoff settings
ERROR_BACKOFF = {
    "max_retries": 5,
    "initial_backoff": 1,  # seconds
    "backoff_factor": 2,  # exponential backoff
}

# RSS feed settings
RSS_SETTINGS = {
    "max_feeds_per_request": 10,  # Maximum number of feeds to try per request
    "max_age_days": 3,            # Maximum age of RSS items to consider
    "timeout_seconds": 5,         # Timeout for RSS feed requests
    "max_workers": 5              # Number of parallel workers for fetching feeds
}
```

### Category-Specific RSS Feeds

Category-specific RSS feeds are defined in `modules/category_detection.py`. These feeds are used to prioritize sources based on the detected claim category:

```python
CATEGORY_SPECIFIC_FEEDS = {
    "ai": [
        "https://www.artificialintelligence-news.com/feed/",
        "https://openai.com/news/rss.xml",
        # Additional AI-specific feeds
    ],
    "science": [
        "https://www.science.org/rss/news_current.xml",
        "https://www.nature.com/nature.rss",
        # Additional science feeds
    ],
    # Additional categories
}
```

## Hugging Face Spaces Deployment

### Setting Up a Space

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Streamlit" as the SDK
   - Choose the hardware tier (use the default 16GB RAM)

2. Upload the project files:
   - You can upload files directly through the Hugging Face web interface
   - Alternatively, use Git to push to the Hugging Face repository
   - Make sure to include all necessary files including requirements.txt

### Setting Up Secrets

1. Add API keys as secrets:
   - Go to the "Settings" tab of your Space
   - Navigate to the "Repository secrets" section
   - Add your API keys:
     - `OPENAI_API_KEY`
     - `NEWS_API_KEY`
     - `FACTCHECK_API_KEY`

### Configuring the Space

Edit the metadata in the `README.md` file:

```yaml
---
title: Askveracity
emoji: 📉
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: mit
short_description: Fact-checking and misinformation detection tool.
---
```

## Custom Configuration

### Adjusting Rate Limits

You can adjust the rate limits in `config.py` based on your API subscription levels:

```python
# Update for higher tier News API subscription
RATE_LIMITS["newsapi"] = {"requests": 500, "period": 3600}  # 500 requests per hour
```

### Modifying RSS Feeds

The list of RSS feeds can be found in `modules/rss_feed.py` and category-specific feeds in `modules/category_detection.py`. You can add or remove feeds as needed.

### Performance Evaluation

The system includes a performance evaluation script `evaluate_performance.py` that:

1. Runs the fact-checking system on a predefined set of test claims
2. Calculates accuracy, safety rate, processing time, and confidence metrics
3. Generates visualization charts in the `results/` directory
4. Saves detailed results to `results/performance_results.json`

To run the performance evaluation:

```bash
python evaluate_performance.py [--limit N] [--output FILE]
```

- `--limit N`: Limit evaluation to first N claims (default: all)
- `--output FILE`: Save results to FILE (default: performance_results.json)

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.

## Troubleshooting

### API Key Issues

If you encounter API key errors:

1. Verify that your API keys are set correctly
2. Check the logs for specific error messages
3. Make sure API keys are not expired or rate-limited

### Model Loading Errors

If spaCy model fails to load:

```bash
# Reinstall the model
python -m spacy download en_core_web_sm --force
```

### Rate Limiting

If you encounter rate limiting issues:

1. Reduce the number of requests by adjusting `RATE_LIMITS` in `config.py`
2. Increase the backoff parameters in `ERROR_BACKOFF`
3. Subscribe to higher API tiers if available

### Memory Issues

If the application crashes due to memory issues:

1. Reduce the number of parallel workers in `RSS_SETTINGS`
2. Limit the maximum number of evidence items processed

## Performance Optimization

For better performance:

1. Upgrade to a higher-tier OpenAI model for improved accuracy
2. Increase the number of parallel workers for evidence retrieval
3. Add more relevant RSS feeds to improve evidence gathering