"""
Performance tracking utility for the Fake News Detector application.

This module provides functionality to track and analyze the
performance of the application, including processing times,
success rates, and resource utilization.
"""

import time
import logging

logger = logging.getLogger("misinformation_detector")

class PerformanceTracker:
    """
    Tracks and logs performance metrics for the fact-checking system.
    
    This class maintains counters and statistics for various performance
    metrics, such as processing times, evidence retrieval success rates,
    and confidence scores.
    """
    
    def __init__(self):
        """Initialize the performance tracker with empty metrics."""
        self.metrics = {
            "claims_processed": 0,
            "evidence_retrieval_success_rate": [],
            "processing_times": [],
            "confidence_scores": [],
            "source_types_used": {},
            "temporal_relevance": []
        }

    def log_claim_processed(self):
        """
        Increment the counter for processed claims.
        This should be called whenever a claim is processed successfully.
        """
        self.metrics["claims_processed"] += 1

    def log_evidence_retrieval(self, success, sources_count):
        """
        Log the success or failure of evidence retrieval.
        
        Args:
            success (bool): Whether evidence retrieval was successful
            sources_count (dict): Count of evidence items by source type
        """
        # Ensure success is a boolean
        success_value = 1 if success else 0
        self.metrics["evidence_retrieval_success_rate"].append(success_value)

        # Safely process source types
        if isinstance(sources_count, dict):
            for source_type, count in sources_count.items():
                # Ensure source_type is a string and count is an integer
                source_type = str(source_type)
                try:
                    count = int(count)
                except (ValueError, TypeError):
                    count = 1

                # Update source types used
                self.metrics["source_types_used"][source_type] = \
                    self.metrics["source_types_used"].get(source_type, 0) + count

    def log_processing_time(self, start_time):
        """
        Log the processing time for an operation.
        
        Args:
            start_time (float): Start time obtained from time.time()
        """
        end_time = time.time()
        processing_time = end_time - start_time
        self.metrics["processing_times"].append(processing_time)

    def log_confidence_score(self, score):
        """
        Log a confidence score.
        
        Args:
            score (float): Confidence score between 0 and 1
        """
        # Ensure score is a float between 0 and 1
        try:
            score = float(score)
            if 0 <= score <= 1:
                self.metrics["confidence_scores"].append(score)
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence score: {score}")

    def log_temporal_relevance(self, relevance_score):
        """
        Log a temporal relevance score.
        
        Args:
            relevance_score (float): Temporal relevance score between 0 and 1
        """
        # Ensure relevance score is a float between 0 and 1
        try:
            relevance_score = float(relevance_score)
            if 0 <= relevance_score <= 1:
                self.metrics["temporal_relevance"].append(relevance_score)
        except (ValueError, TypeError):
            logger.warning(f"Invalid temporal relevance score: {relevance_score}")

    def get_summary(self):
        """
        Get a summary of all performance metrics.
        
        Returns:
            dict: Summary of performance metrics
        """
        # Safely calculate averages with error handling
        def safe_avg(metric_list):
            try:
                return sum(metric_list) / max(len(metric_list), 1)
            except (TypeError, ValueError):
                return 0.0

        return {
            "claims_processed": self.metrics["claims_processed"],
            "avg_evidence_retrieval_success_rate": safe_avg(self.metrics["evidence_retrieval_success_rate"]),
            "avg_processing_time": safe_avg(self.metrics["processing_times"]),
            "avg_confidence_score": safe_avg(self.metrics["confidence_scores"]),
            "source_types_used": dict(self.metrics["source_types_used"]),
            "avg_temporal_relevance": safe_avg(self.metrics["temporal_relevance"])
        }

    def reset(self):
        """Reset all performance metrics."""
        self.__init__()
        logger.info("Performance metrics have been reset")
        return "Performance metrics reset successfully"