import logging
import time
import re
from langdetect import detect
import spacy

from utils.performance import PerformanceTracker
from utils.models import get_nlp_model, get_llm_model
from modules.classification import normalize_tense

logger = logging.getLogger("misinformation_detector")

performance_tracker = PerformanceTracker()

def extract_claims(text):
    """
    Extract the main factual claim from the provided text.
    For concise claims (<30 words), preserves them exactly.
    For longer text, uses OpenAI to extract the claim.
    """
    logger.info(f"Extracting claims from: {text}")
    start_time = time.time()

    # First, check if the input already appears to be a concise claim
    if len(text.split()) < 30:
        logger.info("Input appears to be a concise claim already, preserving as-is")
        performance_tracker.log_processing_time(start_time)
        performance_tracker.log_claim_processed()
        return text

    try:
        # For longer text, use OpenAI for extraction
        extracted_claim = extract_with_openai(text)
        
        # Log processing time
        performance_tracker.log_processing_time(start_time)
        performance_tracker.log_claim_processed()
        
        logger.info(f"Extracted claim: {extracted_claim}")
        return extracted_claim
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        # Fallback to original text on error
        return text

def extract_with_openai(text):
    """
    Use OpenAI model for claim extraction
    """
    try:
        # Get LLM model
        llm_model = get_llm_model()
        
        # Create a very explicit prompt to avoid hallucination
        prompt = f"""
        Extract the main factual claim from the following text. 
        DO NOT add any information not present in the original text.
        DO NOT add locations, dates, or other details.
        ONLY extract what is explicitly stated.
        
        Text: {text}
        
        Main factual claim:
        """
        
        # Call OpenAI with temperature=0 for deterministic output
        response = llm_model.invoke(prompt, temperature=0)
        extracted_claim = response.content.strip()
        
        # Further clean up any explanations or extra text
        if ":" in extracted_claim:
            parts = extracted_claim.split(":")
            if len(parts) > 1:
                extracted_claim = parts[-1].strip()
        
        logger.info(f"OpenAI extraction: {extracted_claim}")
        
        # Validate that we're not adding info not in the original
        nlp = get_nlp_model()
        extracted_claim = validate_extraction(text, extracted_claim, nlp)
        
        return extracted_claim
    except Exception as e:
        logger.error(f"Error in OpenAI claim extraction: {str(e)}")
        return text  # Fallback to original

def validate_extraction(original_text, extracted_claim, nlp):
    """
    Validate that the extracted claim doesn't add information not present in the original text
    """
    # If extraction fails or is empty, return original
    if not extracted_claim or extracted_claim.strip() == "":
        logger.warning("Empty extraction result, using original text")
        return original_text
    
    # Check for added location information
    location_terms = ["united states", "america", "u.s.", "usa", "china", "india", "europe", 
                      "russia", "japan", "uk", "germany", "france", "australia"]
    for term in location_terms:
        if term in extracted_claim.lower() and term not in original_text.lower():
            logger.warning(f"Extraction added location '{term}' not in original, using original text")
            return original_text
    
    # Check for entity preservation/addition using spaCy
    try:
        # Get entities from extracted text
        extracted_doc = nlp(extracted_claim)
        extracted_entities = [ent.text.lower() for ent in extracted_doc.ents]
        
        # Get entities from original text
        original_doc = nlp(original_text)
        original_entities = [ent.text.lower() for ent in original_doc.ents]
        
        # Check for new entities that don't exist in original
        for entity in extracted_entities:
            if not any(entity in orig_entity or orig_entity in entity for orig_entity in original_entities):
                logger.warning(f"Extraction added new entity '{entity}', using original text")
                return original_text
        
        return extracted_claim
    except Exception as e:
        logger.error(f"Error in extraction validation: {str(e)}")
        return original_text  # On error, safer to return original

def shorten_claim_for_evidence(claim):
    """
    Shorten a claim to use for evidence retrieval by preserving important entities,
    verbs, and keywords while maintaining claim context
    
    Args:
        claim (str): The original claim
        
    Returns:
        str: A shortened version of the claim optimized for evidence retrieval
    """
    try:
        normalized_claim = normalize_tense(claim)
        # Get NLP model
        nlp = get_nlp_model()
        
        # Process claim with NLP
        doc = nlp(claim)
        
        # Components to extract
        important_components = []
        
        # 1. Extract all named entities as highest priority
        entities = [ent.text for ent in doc.ents]
        important_components.extend(entities)
        
        # 2. Extract key proper nouns if not already captured in entities
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in important_components:
                important_components.append(token.text)
        
        # 3. Extract main verbs (actions)
        verbs = []
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                verbs.append(token.text)
        
        # 4. Check for important title terms like "president", "prime minister"
        title_terms = ["president", "prime minister", "minister", "chancellor", "premier", 
                      "governor", "mayor", "senator", "CEO", "founder", "director"]
        
        for term in title_terms:
            if term in claim.lower():
                # Find the full phrase (e.g., "Canadian Prime Minister")
                matches = re.finditer(r'(?i)(?:\w+\s+)*\b' + re.escape(term) + r'\b(?:\s+\w+)*', claim)
                for match in matches:
                    phrase = match.group(0)
                    if phrase not in important_components:
                        important_components.append(phrase)
        
        # 5. Add important temporal indicators
        temporal_terms = ["today", "yesterday", "recently", "just", "now",
                               "current", "currently", "latest", "new", "week",
                               "month", "year", "announces", "announced", "introduces",
                               "introduced", "launches", "launched", "releases",
                               "released", "rolls out", "rolled out", "presents", "presented", "unveils", "unveiled", 
                               "starts", "started", "begins", "began", "initiates", "initiated", "anymore"
        ]
        
        # Add significant temporal context
        temporal_context = []
        for term in temporal_terms:
            if term in claim.lower():
                temporal_matches = re.finditer(r'(?i)(?:\w+\s+){0,2}\b' + re.escape(term) + r'\b(?:\s+\w+){0,2}', claim)
                for match in temporal_matches:
                    temporal_context.append(match.group(0))
        
        # 6. Always include negation words as they're critical for meaning
        negation_terms = ["not", "no longer", "former", "ex-", "isn't", "aren't", "doesn't", "don't"]
        
        negation_context = []
        for term in negation_terms:
            if term in claim.lower():
                # Find the context around the negation (3 words before and after)
                neg_matches = re.finditer(r'(?i)(?:\w+\s+){0,3}\b' + re.escape(term) + r'\b(?:\s+\w+){0,3}', claim)
                for match in neg_matches:
                    negation_context.append(match.group(0))
        
        # Combine all components
        all_components = important_components + verbs + temporal_context + negation_context
        
        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for component in all_components:
            if component.lower() not in seen:
                seen.add(component.lower())
                unique_components.append(component)
        
        # If we have too few components (< 2), use the original claim
        if len(unique_components) < 2:
            # If the claim is already short (< 10 words), use as is
            if len(claim.split()) < 10:
                return claim
            
            # Otherwise, use the first 8 words
            words = claim.split()
            return " ".join(words[:min(8, len(words))])
        
        # Join components to create shortened claim
        # Sort components to maintain approximate original word order
        def get_position(comp):
            return claim.lower().find(comp.lower())
        
        unique_components.sort(key=get_position)
        shortened_claim = " ".join(unique_components)
        
        # If the shortened claim is still too long, limit to first 10 words
        if len(shortened_claim.split()) > 10:
            return " ".join(shortened_claim.split()[:10])
            
        return shortened_claim
        
    except Exception as e:
        logger.error(f"Error in shortening claim: {str(e)}")
        # Return original claim on error
        return claim