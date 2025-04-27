"""
Modules package initialization.

This package contains the core modules for the AskVeracity fact-checking system.
"""

from .claim_extraction import extract_claims, shorten_claim_for_evidence
from .evidence_retrieval import retrieve_combined_evidence
from .classification import classify_with_llm, aggregate_evidence
from .explanation import generate_explanation

__all__ = [
    'extract_claims',
    'shorten_claim_for_evidence',
    'retrieve_combined_evidence',
    'classify_with_llm',
    'aggregate_evidence',
    'generate_explanation'
]