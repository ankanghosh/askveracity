"""
Configuration module for the Fake News Detector application.

This module handles loading configuration parameters, API keys,
and source credibility data needed for the fact checking system.
It manages environment variables and file-based configurations.
"""

import os
import logging
from pathlib import Path
import streamlit as st

# Configure logger
logger = logging.getLogger("misinformation_detector")

# Base paths
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# First try to get API keys from Streamlit secrets, then fall back to environment variables
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", os.environ.get("NEWS_API_KEY", ""))
    FACTCHECK_API_KEY = st.secrets.get("FACTCHECK_API_KEY", os.environ.get("FACTCHECK_API_KEY", ""))
except AttributeError:
    # Fall back to environment variables if Streamlit secrets aren't available
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    FACTCHECK_API_KEY = os.environ.get("FACTCHECK_API_KEY", "")

# Log secrets status (but not the values)
if OPENAI_API_KEY:
    logger.info("OPENAI_API_KEY is set")
else:
    logger.warning("OPENAI_API_KEY not set. The application will not function properly.")
    
if NEWS_API_KEY:
    logger.info("NEWS_API_KEY is set")
else:
    logger.warning("NEWS_API_KEY not set. News evidence retrieval will be limited.")
    
if FACTCHECK_API_KEY:
    logger.info("FACTCHECK_API_KEY is set")
else:
    logger.warning("FACTCHECK_API_KEY not set. Fact-checking evidence will be limited.")

# Set API key in environment to ensure it's available to all components
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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