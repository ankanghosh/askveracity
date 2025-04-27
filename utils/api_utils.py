"""
API utilities for the Fake News Detector application.

This module provides utilities for handling API calls, rate limiting,
error handling, and exponential backoff for retrying failed requests.
"""

import time
import functools
import random
import logging
import requests
from datetime import datetime, timedelta
from collections import deque

from config import RATE_LIMITS, ERROR_BACKOFF

logger = logging.getLogger("misinformation_detector")

class RateLimiter:
    """
    Rate limiter for API calls with support for different APIs.
    
    This class implements a token bucket algorithm for rate limiting,
    with support for different rate limits for different APIs.
    It also provides exponential backoff for error handling.
    """
    
    def __init__(self):
        """Initialize the rate limiter with configuration from settings."""
        # Store rate limits for different APIs
        self.limits = {}
        
        # Initialize limits from config
        for api_name, limit_info in RATE_LIMITS.items():
            self.limits[api_name] = {
                "requests": limit_info["requests"], 
                "period": limit_info["period"], 
                "timestamps": deque()
            }

        # Error backoff settings
        self.max_retries = ERROR_BACKOFF["max_retries"]
        self.initial_backoff = ERROR_BACKOFF["initial_backoff"]
        self.backoff_factor = ERROR_BACKOFF["backoff_factor"]

    def check_and_update(self, api_name):
        """
        Check if request is allowed and update timestamps.
        
        Args:
            api_name (str): Name of the API to check
            
        Returns:
            tuple: (allowed, wait_time)
                - allowed (bool): Whether the request is allowed
                - wait_time (float): Time to wait if not allowed
        """
        if api_name not in self.limits:
            return True, 0  # Unknown API, allow by default

        now = datetime.now()
        limit_info = self.limits[api_name]

        # Remove timestamps older than the period
        cutoff = now - timedelta(seconds=limit_info["period"])
        while limit_info["timestamps"] and limit_info["timestamps"][0] < cutoff:
            limit_info["timestamps"].popleft()

        # Check if we're at the rate limit
        if len(limit_info["timestamps"]) >= limit_info["requests"]:
            # Calculate wait time until oldest timestamp expires
            wait_time = (limit_info["timestamps"][0] + timedelta(seconds=limit_info["period"]) - now).total_seconds()
            return False, max(0, wait_time)

        # Add current timestamp and allow request
        limit_info["timestamps"].append(now)
        return True, 0

    def wait_if_needed(self, api_name):
        """
        Wait if rate limit is reached.
        
        Args:
            api_name (str): Name of the API to check
            
        Returns:
            bool: True if waited, False otherwise
        """
        allowed, wait_time = self.check_and_update(api_name)
        if not allowed:
            logger.info(f"Rate limit reached for {api_name}. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time + 0.1)  # Add a small buffer
            return True
        return False

    def get_backoff_time(self, attempt):
        """
        Calculate exponential backoff time with jitter.
        
        Args:
            attempt (int): Current attempt number (0-based)
            
        Returns:
            float: Backoff time in seconds
        """
        backoff = self.initial_backoff * (self.backoff_factor ** attempt)
        # Add jitter to prevent thundering herd problem
        jitter = random.uniform(0, 0.1 * backoff)
        return backoff + jitter


# Create rate limiter instance
rate_limiter = RateLimiter()

# API Error Handler decorator
def api_error_handler(api_name):
    """
    Decorator for API calls with error handling and rate limiting.
    
    This decorator handles rate limiting, retries with exponential
    backoff, and error handling for API calls.
    
    Args:
        api_name (str): Name of the API being called
        
    Returns:
        callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Apply rate limiting - make sure rate_limiter exists and has the method
                if hasattr(rate_limiter, 'wait_if_needed'):
                    rate_limiter.wait_if_needed(api_name)

                # Track retries
                for attempt in range(rate_limiter.max_retries):
                    try:
                        return func(*args, **kwargs)
                    except requests.exceptions.HTTPError as e:
                        status_code = e.response.status_code if hasattr(e, 'response') else 0

                        # Handle specific HTTP errors
                        if status_code == 429:  # Too Many Requests
                            logger.warning(f"{api_name} rate limit exceeded (429). Attempt {attempt+1}/{rate_limiter.max_retries}")
                            # Get retry-after header or use exponential backoff
                            retry_after = e.response.headers.get('Retry-After')
                            if retry_after and retry_after.isdigit():
                                wait_time = int(retry_after)
                            else:
                                wait_time = rate_limiter.get_backoff_time(attempt)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        elif status_code >= 500:  # Server errors
                            logger.warning(f"{api_name} server error ({status_code}). Attempt {attempt+1}/{rate_limiter.max_retries}")
                            time.sleep(rate_limiter.get_backoff_time(attempt))
                        elif status_code == 403:  # Forbidden - likely API key issue
                            logger.error(f"{api_name} access forbidden (403). Check API key.")
                            return None  # Don't retry on auth errors
                        elif status_code == 404:  # Not Found
                            logger.warning(f"{api_name} resource not found (404).")
                            return None  # Don't retry on resource not found
                        else:
                            logger.error(f"{api_name} HTTP error: {e}")
                            if attempt < rate_limiter.max_retries - 1:
                                wait_time = rate_limiter.get_backoff_time(attempt)
                                logger.info(f"Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                return None

                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"{api_name} connection error: {e}")
                        if attempt < rate_limiter.max_retries - 1:
                            wait_time = rate_limiter.get_backoff_time(attempt)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            return None

                    except requests.exceptions.Timeout as e:
                        logger.error(f"{api_name} timeout error: {e}")
                        if attempt < rate_limiter.max_retries - 1:
                            wait_time = rate_limiter.get_backoff_time(attempt)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            return None

                    except Exception as e:
                        logger.error(f"{api_name} unexpected error: {str(e)}")
                        if attempt < rate_limiter.max_retries - 1:
                            wait_time = rate_limiter.get_backoff_time(attempt)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            return None

                # If we've exhausted all retries
                logger.error(f"{api_name} call failed after {rate_limiter.max_retries} attempts")
                return None

            except Exception as e:
                # Catch any unexpected errors in the decorator itself
                logger.error(f"{api_name} decorator error: {str(e)}")
                return None

        return wrapper
    return decorator

def safe_json_parse(response, api_name):
    """
    Safely parse JSON response with error handling.
    
    Args:
        response (requests.Response): Response object to parse
        api_name (str): Name of the API for logging
        
    Returns:
        dict: Parsed JSON or empty dict on error
    """
    try:
        return response.json()
    except ValueError as e:
        logger.error(f"Error parsing {api_name} JSON response: {e}")
        logger.debug(f"Response content: {response.text[:500]}...")
        return {}