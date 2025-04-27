"""
Model management utility for the Fake News Detector application.

This module provides functions for initializing, caching, and
retrieving language models used throughout the application.
It ensures models are loaded efficiently and reused appropriately.
"""

import os
import logging
import functools
from langchain_openai import ChatOpenAI
import spacy

logger = logging.getLogger("misinformation_detector")

# Global variables for models
nlp = None
model = None
models_initialized = False

# Add caching decorator
def cached_model(func):
    """
    Decorator to cache model loading for improved performance.
    
    This decorator ensures that models are only loaded once and
    then reused for subsequent calls, improving performance by
    avoiding redundant model loading.
    
    Args:
        func (callable): Function that loads a model
        
    Returns:
        callable: Wrapped function that returns a cached model
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Use function name as cache key
        key = func.__name__
        if key not in cache:
            logger.info(f"Model not in cache, calling {key}...")
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def initialize_models():
    """
    Initialize all required models.
    
    This function loads and initializes all the language models
    needed by the application, including spaCy for NLP tasks and
    OpenAI for LLM-based processing.
    
    Returns:
        str: Initialization status message
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    global nlp, model, models_initialized
    
    # Skip initialization if already done
    if models_initialized:
        logger.info("Models already initialized, skipping initialization")
        return "Models already initialized"
    
    # Check OpenAI API key
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        logger.error("OPENAI_API_KEY environment variable not set or empty")
        raise ValueError("OpenAI API key is required. Please set it in the Hugging Face Space secrets.")

    try:
        # Load NLP model
        try:
            logger.info("Loading spaCy NLP model...")
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy NLP model")
        except OSError as e:
            # This handles the case if the model wasn't installed correctly
            logger.warning(f"Could not load spaCy model: {str(e)}")
            logger.info("Attempting to download spaCy model...")
            try:
                import subprocess
                import sys
                # This downloads the model if it's missing
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                # Try loading again
                nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully downloaded and loaded spaCy model")
            except Exception as download_err:
                logger.error(f"Failed to download spaCy model: {str(download_err)}")
                # Continue with other initialization, we'll handle missing NLP model elsewhere

        # Set up OpenAI model
        logger.info("Initializing ChatOpenAI model...")
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        logger.info("Initialized ChatOpenAI model")
        
        # Mark initialization as complete
        models_initialized = True
        return "Models initialized successfully"
    
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise e

@cached_model
def get_nlp_model():
    """
    Get the spaCy NLP model, initializing if needed.
    
    This function returns a cached spaCy model for NLP tasks.
    If the model hasn't been loaded yet, it will be loaded.
    
    Returns:
        spacy.Language: Loaded spaCy model
    """
    global nlp
    if nlp is None:
        try:
            # Try to load just the spaCy model if not loaded yet
            logger.info("Loading spaCy NLP model...")
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy NLP model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            # Fall back to full initialization
            initialize_models()
    return nlp

@cached_model
def get_llm_model():
    """
    Get the ChatOpenAI model, initializing if needed.
    
    This function returns a cached OpenAI LLM model.
    If the model hasn't been loaded yet, it will be loaded.
    
    Returns:
        ChatOpenAI: Loaded LLM model
    """
    global model
    if model is None:
        try:
            # Try to load just the LLM model if not loaded yet
            logger.info("Initializing ChatOpenAI model...")
            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            logger.info("Initialized ChatOpenAI model")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI model: {str(e)}")
            # Fall back to full initialization
            initialize_models()
    return model