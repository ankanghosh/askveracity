import logging
import re
from utils.models import get_llm_model, get_nlp_model
from utils.performance import PerformanceTracker

logger = logging.getLogger("misinformation_detector")

performance_tracker = PerformanceTracker()

def classify_with_llm(query, evidence):
    """
    Classification function that evaluates evidence against a claim
    to determine support, contradiction, or insufficient evidence.
    
    This function analyzes the provided evidence to evaluate if it supports, 
    contradicts, or is insufficient to verify the claim. It implements:
    - Strict output formatting requirements
    - Evidence source validation
    - Confidence scoring based on confirmation strength
    - Flexible regex pattern matching
    - Detailed debug logging
    - Fallback parsing for non-standard responses
    
    Args:
        query (str): The factual claim being verified
        evidence (list): Evidence items to evaluate against the claim
        
    Returns:
        list: Classification results with labels and confidence scores
    """
    logger.info(f"Classifying evidence for claim: {query}")
    
    # Get the LLM model
    llm_model = get_llm_model()
    
    # Skip if no evidence
    if not evidence:
        logger.warning("No evidence provided for classification")
        return []

    # Normalize evidence to a list
    if not isinstance(evidence, list):
        if evidence:
            try:
                evidence = [evidence]
            except Exception as e:
                logger.error(f"Could not convert evidence to list: {e}")
                return []
        else:
            return []

    # Extract essential claim components for improved keyword detection
    claim_components = extract_claim_keywords(query)
    essential_keywords = claim_components.get("keywords", [])
    essential_entities = claim_components.get("entities", [])
    
    # Ensure processing is limited to top 10 evidence items to reduce token usage
    evidence = evidence[:10]

    # Validate evidence for verifiable sources
    validated_evidence = []
    for idx, chunk in enumerate(evidence):
        # Basic evidence validation
        if not isinstance(chunk, str) or not chunk.strip():
            continue
            
        # Check if evidence contains source information
        has_valid_source = False
        if "URL:" in chunk and ("http://" in chunk or "https://" in chunk):
            has_valid_source = True
        elif "Source:" in chunk and len(chunk.split("Source:")[1].strip()) > 3:
            has_valid_source = True
            
        # Add validation flag to evidence
        validated_evidence.append({
            "text": chunk,
            "index": idx + 1,
            "has_valid_source": has_valid_source
        })
    
    # If no valid evidence remains, return early
    if not validated_evidence:
        logger.warning("No valid evidence items to classify")
        return []

    try:
        # Format evidence items with validation information
        evidence_text = ""
        for item in validated_evidence:
            # Truncate long evidence
            chunk_text = item["text"]
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."
            
            # Include validation status in the prompt
            source_status = "WITH VALID SOURCE" if item["has_valid_source"] else "WARNING: NO CLEAR SOURCE"
            evidence_text += f"EVIDENCE {item['index']}:\n{chunk_text}\n[{source_status}]\n\n"

        # Create a structured prompt with explicit format instructions and validation requirements
        prompt = f"""
        CLAIM: {query}

        EVIDENCE:
        {evidence_text}

        TASK: Evaluate if each evidence supports, contradicts, or is insufficient/irrelevant to the claim.
        
        INSTRUCTIONS:
        1. For each evidence, provide your analysis in EXACTLY this format:
        
        EVIDENCE [number] ANALYSIS:
        Classification: [Choose exactly one: support/contradict/insufficient]
        Confidence: [number between 0-100]
        Reason: [brief explanation]
        
        2. Support = Evidence EXPLICITLY confirms ALL parts of the claim are true
        3. Contradict = Evidence EXPLICITLY confirms the claim is false
        4. Insufficient = Evidence is irrelevant, ambiguous, or doesn't provide enough information
        
        CRITICAL VALIDATION RULES:
        - Mark as "support" ONLY when evidence EXPLICITLY mentions ALL key entities AND actions from the claim
        - Do not label evidence as "support" if it only discusses the same topic without confirming the specific claim
        - Do not make inferential leaps - if the evidence doesn't explicitly state the claim, mark as "insufficient"
        - Assign LOW confidence (0-50) when evidence doesn't explicitly mention all claim elements
        - Assign ZERO confidence (0) to evidence without valid sources
        - If evidence describes similar but different events, mark as "insufficient", not "support" or "contradict"
        - If evidence describes the same topic as the claim but does not confirm or contradict the claim, mark as "insufficient", not "support" or "contradict"
        - If evidence is in a different language or unrelated topic, mark as "insufficient" with 0 confidence
        - Check that all entities (names, places, dates, numbers) in the claim are explicitly confirmed
        
        FOCUS ON THE EXACT CLAIM ONLY.
        ESSENTIAL KEYWORDS TO LOOK FOR: {', '.join(essential_keywords)}
        ESSENTIAL ENTITIES TO VERIFY: {', '.join(essential_entities)}

        IMPORTANT NOTE ABOUT VERB TENSES: When analyzing this claim, treat present tense verbs (like "unveils") 
        and perfect form verbs (like "has unveiled") as equivalent to their simple past tense forms 
        (like "unveiled"). The tense variation should not affect your classification decision.
        """

        # Get response with temperature=0 for consistency
        result = llm_model.invoke(prompt, temperature=0)
        result_text = result.content.strip()
        
        # Log the raw LLM response for debugging
        logger.debug(f"Raw LLM classification response:\n{result_text}")
        
        # Define a more flexible regex pattern matching the requested format
        # This pattern accommodates variations in whitespace and formatting
        analysis_pattern = r'EVIDENCE\s+(\d+)\s+ANALYSIS:[\s\n]*Classification:[\s\n]*(support|contradict|insufficient)[\s\n]*Confidence:[\s\n]*(\d+)[\s\n]*Reason:[\s\n]*(.*?)(?=[\s\n]*EVIDENCE\s+\d+\s+ANALYSIS:|[\s\n]*$)'
        
        # Parse each evidence analysis
        classification_results = []
        
        # Try matching with our pattern
        matches = list(re.finditer(analysis_pattern, result_text, re.IGNORECASE | re.DOTALL))
        
        # Log match information for debugging
        logger.debug(f"Found {len(matches)} structured evidence analyses in response")
        
        # Process matches
        for match in matches:
            try:
                evidence_idx = int(match.group(1)) - 1
                classification = match.group(2).lower()
                confidence = int(match.group(3)) / 100.0  # Convert to 0-1 scale
                reason = match.group(4).strip()
                
                # Check if this evidence item exists in our original list
                if 0 <= evidence_idx < len(evidence):
                    # Get the original evidence text
                    evidence_text = evidence[evidence_idx]
                    
                    # Check for valid source
                    source_valid = False
                    if "URL:" in evidence_text and ("http://" in evidence_text or "https://" in evidence_text):
                        source_valid = True
                    elif "Source:" in evidence_text:
                        source_valid = True
                        
                    # Reduce confidence for evidence without valid sources
                    if not source_valid and confidence > 0.3:
                        confidence = 0.3
                        reason += " (Confidence reduced due to lack of verifiable source)"
                        
                    # Create result entry
                    classification_results.append({
                        "label": classification,
                        "confidence": confidence,
                        "evidence": evidence_text,
                        "reason": reason
                    })
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing evidence analysis: {e}")
        
        # If no structured matches were found, try using a simpler approach
        if not classification_results:
            logger.warning("No structured evidence analysis found, using fallback method")
            
            # Log detailed information about the failure
            logger.warning(f"Expected format not found in response. Response excerpt: {result_text[:200]}...")
            
            # Simple fallback parsing based on keywords
            for idx, ev in enumerate(evidence):
                # Check for keywords in the LLM response
                ev_mention = f"EVIDENCE {idx+1}"
                if ev_mention in result_text:
                    # Find the section for this evidence
                    parts = result_text.split(ev_mention)
                    if len(parts) > 1:
                        analysis_text = parts[1].split("EVIDENCE")[0] if "EVIDENCE" in parts[1] else parts[1]
                        
                        # Determine classification
                        label = "insufficient"  # Default
                        confidence = 0.0  # Default - zero confidence for fallback parsing

                        # Check for support indicators
                        if "support" in analysis_text.lower() or "confirms" in analysis_text.lower():
                            label = "support"
                            confidence = 0.4  # Lower confidence for fallback support
                            
                        # Check for contradict indicators    
                        elif "contradict" in analysis_text.lower() or "false" in analysis_text.lower():
                            label = "contradict"
                            confidence = 0.4  # Lower confidence for fallback contradict
                        
                        # Check for valid source to adjust confidence
                        source_valid = False
                        if "URL:" in ev and ("http://" in ev or "https://" in ev):
                            source_valid = True
                        elif "Source:" in ev:
                            source_valid = True
                            
                        if not source_valid:
                            confidence = min(confidence, 0.3)
                            
                        # Create basic result
                        classification_results.append({
                            "label": label,
                            "confidence": confidence,
                            "evidence": ev,
                            "reason": f"Determined via fallback parsing. {'Valid source found.' if source_valid else 'Warning: No clear source identified.'}"
                        })
                        
                        logger.debug(f"Fallback parsing for evidence {idx+1}: {label} with confidence {confidence}")
        
        logger.info(f"Classified {len(classification_results)} evidence items")
        return classification_results

    except Exception as e:
        logger.error(f"Error in evidence classification: {str(e)}")
        # Provide a basic fallback
        fallback_results = []
        for ev in evidence:
            fallback_results.append({
                "label": "insufficient",
                "confidence": 0.5,
                "evidence": ev,
                "reason": "Classification failed with error, using fallback"
            })
        return fallback_results

def normalize_tense(claim):
    """
    Normalize verb tenses in claims to ensure consistent classification.
    
    This function standardizes verb forms by converting present simple tense 
    verbs (e.g., "unveils") and perfect forms (e.g., "has unveiled") to their 
    past tense equivalents (e.g., "unveiled"). This ensures that semantically 
    equivalent claims are processed consistently regardless of verb tense 
    variations.
    
    Args:
        claim (str): The original claim text to normalize
        
    Returns:
        str: The normalized claim with consistent tense handling
        
    Note:
        This function specifically targets present simple and perfect forms,
        preserving the semantic differences of continuous forms (is unveiling)
        and future tense (will unveil).
    """
    # Define patterns to normalize common verb forms.
    # Each tuple contains (regex_pattern, replacement_text)
    tense_patterns = [
        # Present simple to past tense conversions
        (r'\bunveils\b', r'unveiled'),
        (r'\blaunches\b', r'launched'),
        (r'\breleases\b', r'released'),
        (r'\bannounces\b', r'announced'),
        (r'\binvites\b', r'invited'),
        (r'\bretaliates\b', r'retaliated'),
        (r'\bends\b', r'ended'),
        (r'\bbegins\b', r'began'),
        (r'\bstarts\b', r'started'),
        (r'\bcompletes\b', r'completed'),
        (r'\bfinishes\b', r'finished'),
        (r'\bintroduces\b', r'introduced'),
        (r'\bcreates\b', r'created'),
        (r'\bdevelops\b', r'developed'),
        (r'\bpublishes\b', r'published'),
        (r'\bacquires\b', r'acquired'),
        (r'\bbuys\b', r'bought'),
        (r'\bsells\b', r'sold'),
        
        # Perfect forms (has/have/had + past participle) to simple past
        (r'\b(has|have|had)\s+unveiled\b', r'unveiled'),
        (r'\b(has|have|had)\s+launched\b', r'launched'),
        (r'\b(has|have|had)\s+released\b', r'released'),
        (r'\b(has|have|had)\s+announced\b', r'announced'),
        (r'\b(has|have|had)\s+invited\b', r'invited'),
        (r'\b(has|have|had)\s+retaliated\b', r'retaliated'),
        (r'\b(has|have|had)\s+ended\b', r'ended'),
        (r'\b(has|have|had)\s+begun\b', r'began'),
        (r'\b(has|have|had)\s+started\b', r'started'),
        (r'\b(has|have|had)\s+introduced\b', r'introduced'),
        (r'\b(has|have|had)\s+created\b', r'created'),
        (r'\b(has|have|had)\s+developed\b', r'developed'),
        (r'\b(has|have|had)\s+published\b', r'published'),
        (r'\b(has|have|had)\s+acquired\b', r'acquired'),
        (r'\b(has|have|had)\s+bought\b', r'bought'),
        (r'\b(has|have|had)\s+sold\b', r'sold')
    ]
    
    # Apply normalization patterns
    normalized = claim
    for pattern, replacement in tense_patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Log if normalization occurred for debugging purposes
    if normalized != claim:
        logger.info(f"Normalized claim from: '{claim}' to: '{normalized}'")
        
    return normalized 

def aggregate_evidence(classification_results):
    """
    Aggregate evidence classifications to determine overall verdict
    using a weighted scoring system of evidence count and quality.
    
    Args:
        classification_results (list): List of evidence classification results
        
    Returns:
        tuple: (verdict, confidence) - The final verdict and confidence score
    """
    logger.info(f"Aggregating evidence from {len(classification_results) if classification_results else 0} results")
    
    if not classification_results:
        logger.warning("No classification results to aggregate")
        return "Uncertain", 0.0  # Default with zero confidence
    
    # Only consider support and contradict evidence items
    support_items = [item for item in classification_results if item.get("label") == "support"]
    contradict_items = [item for item in classification_results if item.get("label") == "contradict"]
    
    # Count number of support and contradict items
    support_count = len(support_items)
    contradict_count = len(contradict_items)
    
    # Calculate confidence scores for support and contradict items
    support_confidence_sum = sum(item.get("confidence", 0) for item in support_items)
    contradict_confidence_sum = sum(item.get("confidence", 0) for item in contradict_items)
    
    # Apply weights: 55% for count, 45% for quality (confidence)
    # Normalize counts to avoid division by zero
    max_count = max(1, max(support_count, contradict_count))
    
    # Calculate weighted scores
    count_support_score = (support_count / max_count) * 0.55
    count_contradict_score = (contradict_count / max_count) * 0.55
    
    # Normalize confidence scores to avoid division by zero
    max_confidence_sum = max(1, max(support_confidence_sum, contradict_confidence_sum))
    
    quality_support_score = (support_confidence_sum / max_confidence_sum) * 0.45
    quality_contradict_score = (contradict_confidence_sum / max_confidence_sum) * 0.45
    
    # Total scores
    total_support = count_support_score + quality_support_score
    total_contradict = count_contradict_score + quality_contradict_score
    
    # Check if all evidence is irrelevant/insufficient
    if support_count == 0 and contradict_count == 0:
        logger.info("All evidence items are irrelevant/insufficient")
        return "Uncertain", 0.0
    
    # Determine verdict based on higher total score
    if total_support > total_contradict:
        verdict = "True (Based on Evidence)"
        min_score = total_contradict
        max_score = total_support
    else:
        verdict = "False (Based on Evidence)"
        min_score = total_support
        max_score = total_contradict
    
    # Calculate final confidence using the formula:
    # (1 - min_score/max_score) * 100%
    if max_score > 0:
        final_confidence = 1.0 - (min_score / max_score)
    else:
        final_confidence = 0.0
    
    # Handle cases where confidence is very low
    if final_confidence == 0.0:
        return "Uncertain", 0.0
    elif final_confidence < 0.1:  # Less than 10%
        # Keep the verdict but with very low confidence
        logger.info(f"Very low confidence verdict: {verdict} with {final_confidence:.2f} confidence")
    
    logger.info(f"Final verdict: {verdict}, confidence: {final_confidence:.2f}")
    
    return verdict, final_confidence

def extract_claim_keywords(claim):
    """
    Extract important keywords from claim using NLP processing
    
    Args:
        claim (str): The claim text
        
    Returns:
        dict: Dictionary containing keywords and other claim components
    """
    try:
        # Get NLP model
        nlp = get_nlp_model()
        
        # Process claim with NLP
        doc = nlp(claim)
        
        # Extract entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract important keywords (non-stopword nouns, adjectives, and verbs longer than 3 chars)
        keywords = []
        for token in doc:
            # Keep all important parts of speech, longer than 3 characters
            if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"] and not token.is_stop and len(token.text) > 3:
                keywords.append(token.text.lower())
            # Also include some important modifiers and quantifiers
            elif token.pos_ in ["NUM", "ADV"] and not token.is_stop and len(token.text) > 1:
                keywords.append(token.text.lower())
        
        # Extract verbs separately
        verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB" and not token.is_stop]
        
        # Also extract multi-word phrases that might be important
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3 and not all(token.is_stop for token in chunk):
                noun_phrases.append(chunk.text.lower())
        
        # Add phrases to keywords if not already included
        for phrase in noun_phrases:
            if phrase not in keywords and phrase.lower() not in [k.lower() for k in keywords]:
                keywords.append(phrase.lower())
        
        # Return all components
        return {
            "entities": entities,
            "keywords": keywords,
            "verbs": verbs,
            "noun_phrases": noun_phrases
        }
        
    except Exception as e:
        logger.error(f"Error extracting claim keywords: {e}")
        # Return basic fallback using simple word extraction
        words = [word.lower() for word in claim.split() if len(word) > 3]
        return {"keywords": words, "entities": [], "verbs": [], "noun_phrases": []}