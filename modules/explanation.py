import logging
import re
import ast
from utils.models import get_llm_model

logger = logging.getLogger("misinformation_detector")

def extract_most_relevant_evidence(evidence_results):
    """
    Intelligently extract the most relevant piece of evidence
    
    Args:
        evidence_results (list): List of evidence items
    
    Returns:
        str: Most relevant evidence piece
    """
    if not evidence_results:
        return None

    # If evidence is a dictionary with 'evidence' key
    if isinstance(evidence_results[0], dict):
        # Sort by confidence if available
        sorted_evidence = sorted(
            evidence_results, 
            key=lambda x: x.get('confidence', 0), 
            reverse=True
        )
        
        # Return the evidence from the highest confidence item
        for item in sorted_evidence:
            evidence = item.get('evidence')
            if evidence:
                return evidence

    # If plain list of evidence
    return next((ev for ev in evidence_results if ev and isinstance(ev, str)), None)

def generate_explanation(claim, evidence_results, truth_label, confidence=None):
    """
    Generate an explanation for the claim's classification based on evidence.
    
    This function creates a human-readable explanation of why a claim was classified
    as true, false, or uncertain. It handles different truth label formats through
    normalization and provides robust fallback mechanisms for error cases.
    
    Args:
        claim (str): The original factual claim being verified
        evidence_results (list/str): Evidence supporting the classification, can be
                                    a list of evidence items or structured results
        truth_label (str): Classification of the claim (True/False/Uncertain),
                           which may come in various formats
        confidence (float, optional): Confidence level between 0 and 1
                                     
    Returns:
        str: Natural language explanation of the verdict with appropriate
             confidence framing and evidence citations
    """
    logger.info(f"Generating explanation for claim with verdict: {truth_label}")
    
    try:
        # Normalize truth_label to handle different formats consistently
        normalized_label = normalize_truth_label(truth_label)
        
        # Normalize evidence_results to a list
        if not isinstance(evidence_results, list):
            try:
                evidence_results = ast.literal_eval(str(evidence_results)) if evidence_results else []
            except:
                evidence_results = [evidence_results] if evidence_results else []

        # Get the LLM model
        explanation_model = get_llm_model()

        # Extract most relevant evidence
        most_relevant_evidence = extract_most_relevant_evidence(evidence_results)

        # Prepare evidence text for prompt
        evidence_text = "\n".join([
            f"Evidence {i+1}: {str(ev)[:200] + '...' if len(str(ev)) > 200 else str(ev)}"
            for i, ev in enumerate(evidence_results[:5])
        ])

        # Filter only supporting and contradicting evidence for clarity
        support_items = [item for item in evidence_results if isinstance(item, dict) and item.get("label") == "support"]
        contradict_items = [item for item in evidence_results if isinstance(item, dict) and item.get("label") == "contradict"]
        
        # Convert confidence to percentage and description
        confidence_desc = ""
        very_low_confidence = False
        
        # For Uncertain verdicts, always use 0% confidence regardless of evidence confidence values
        if "uncertain" in normalized_label.lower():
            confidence = 0.0
            confidence_desc = "no confidence (0%)"
        elif confidence is not None:
            confidence_pct = int(confidence * 100)
            
            if confidence == 0.0:
                confidence_desc = "no confidence (0%)"
            elif confidence < 0.1:
                confidence_desc = f"very low confidence ({confidence_pct}%)"
                very_low_confidence = True
            elif confidence < 0.3:
                confidence_desc = f"low confidence ({confidence_pct}%)"
            elif confidence < 0.7:
                confidence_desc = f"moderate confidence ({confidence_pct}%)"
            elif confidence < 0.9:
                confidence_desc = f"high confidence ({confidence_pct}%)"
            else:
                confidence_desc = f"very high confidence ({confidence_pct}%)"
        else:
            # Default if no confidence provided
            confidence_desc = "uncertain confidence"

        # Create prompt with specific instructions based on the type of claim
        has_negation = any(neg in claim.lower() for neg in ["not", "no longer", "isn't", "doesn't", "won't", "cannot"])
        
        # For claims with "True" verdict
        if "true" in normalized_label.lower():
            # Special case for very low confidence (but not zero)
            if very_low_confidence:
                prompt = f"""
                Claim: "{claim}"
                
                Verdict: {normalized_label} (with {confidence_desc})

                Available Evidence:
                {evidence_text}

                Task: Generate a clear explanation that:
                1. States that the claim appears to be true based on the available evidence
                2. EMPHASIZES that the confidence level is VERY LOW ({confidence_pct}%)
                3. Explains that this means the evidence slightly favors the claim but is not strong enough to be certain
                4. STRONGLY recommends that the user verify this with other authoritative sources
                5. Is factual and precise
                """
            else:
                prompt = f"""
                Claim: "{claim}"
                
                Verdict: {normalized_label} (with {confidence_desc})

                Available Evidence:
                {evidence_text}

                Task: Generate a clear explanation that:
                1. Clearly states that the claim IS TRUE based on the evidence
                2. {"Pay special attention to the logical relationship since the claim contains negation" if has_negation else "Explains why the evidence supports the claim"}
                3. Uses confidence level of {confidence_desc}
                4. Highlights the most relevant supporting evidence
                5. Is factual and precise
                """

        # For claims with "False" verdict
        elif "false" in normalized_label.lower():
            # Special case for very low confidence (but not zero)
            if very_low_confidence:
                prompt = f"""
                Claim: "{claim}"
                
                Verdict: {normalized_label} (with {confidence_desc})

                Available Evidence:
                {evidence_text}

                Task: Generate a clear explanation that:
                1. States that the claim appears to be false based on the available evidence
                2. EMPHASIZES that the confidence level is VERY LOW ({confidence_pct}%)
                3. Explains that this means the evidence slightly contradicts the claim but is not strong enough to be certain
                4. STRONGLY recommends that the user verify this with other authoritative sources
                5. Is factual and precise
                """
            else:
                prompt = f"""
                Claim: "{claim}"
                
                Verdict: {normalized_label} (with {confidence_desc})

                Available Evidence:
                {evidence_text}

                Task: Generate a clear explanation that:
                1. Clearly states that the claim IS FALSE based on the evidence
                2. {"Pay special attention to the logical relationship since the claim contains negation" if has_negation else "Explains why the evidence contradicts the claim"}
                3. Uses confidence level of {confidence_desc}
                4. Highlights the contradicting evidence
                5. Is factual and precise
                """

        # For uncertain claims
        else:
            prompt = f"""
            Claim: "{claim}"
            
            Verdict: {normalized_label} (with {confidence_desc})

            Available Evidence:
            {evidence_text}

            Task: Generate a clear explanation that:
            1. Clearly states that there is insufficient evidence to determine if the claim is true or false
            2. Explains what information is missing or why the available evidence is insufficient
            3. Uses confidence level of {confidence_desc}
            4. Makes NO speculation about whether the claim might be true or false
            5. Explicitly mentions that the user should seek information from other reliable sources
            """

        # Generate explanation with multiple attempts for reliability
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Invoke the model
                response = explanation_model.invoke(prompt)
                explanation = response.content.strip()

                # Validate explanation length
                if explanation and len(explanation.split()) >= 5:
                    return explanation

            except Exception as attempt_error:
                logger.error(f"Explanation generation attempt {attempt+1} failed: {str(attempt_error)}")

        # Ultimate fallback explanations if all attempts fail
        if "uncertain" in normalized_label.lower():
            return f"The claim '{claim}' cannot be verified due to insufficient evidence. The available information does not provide clear support for or against this claim. Consider consulting reliable sources for verification."
        elif very_low_confidence:
            return f"The claim '{claim}' appears to be {'supported' if 'true' in normalized_label.lower() else 'contradicted'} by the evidence, but with very low confidence ({confidence_pct}%). The evidence is not strong enough to make a definitive determination. It is strongly recommended to verify this information with other authoritative sources."
        elif "true" in normalized_label.lower():
            return f"The claim '{claim}' is supported by the evidence with {confidence_desc}. {most_relevant_evidence or 'The evidence indicates this claim is accurate.'}"
        else:
            return f"The claim '{claim}' is contradicted by the evidence with {confidence_desc}. {most_relevant_evidence or 'The evidence indicates this claim is not accurate.'}"

    except Exception as e:
        logger.error(f"Comprehensive error in explanation generation: {str(e)}")
        # Final fallback with minimal but useful information
        normalized_label = normalize_truth_label(truth_label)
        return f"The claim is classified as {normalized_label} based on the available evidence."

def normalize_truth_label(truth_label):
    """
    Normalize truth label to handle different formats consistently.
    
    This function extracts the core truth classification (True/False/Uncertain) from
    potentially complex or inconsistently formatted truth labels. It preserves
    contextual information like "(Based on Evidence)" when present.
    
    Args:
        truth_label (str): The truth label to normalize, which may contain 
                           additional descriptive text or formatting
        
    Returns:
        str: Normalized truth label that preserves the core classification and
             important context while eliminating inconsistencies
             
    Examples:
        >>> normalize_truth_label("True (Based on Evidence)")
        "True (Based on Evidence)"
        >>> normalize_truth_label("false (Based on Evidence)")
        "False (Based on Evidence)"
        >>> normalize_truth_label("The evidence shows this claim is False")
        "False"
    """
    if not truth_label:
        return "Uncertain"
    
    # Convert to string if not already
    label_str = str(truth_label)
    
    # Extract the core label if it contains additional text like "(Based on Evidence)"
    base_label_match = re.search(r'(True|False|Uncertain|Error)', label_str, re.IGNORECASE)
    if base_label_match:
        # Get the core label and capitalize it for consistency
        base_label = base_label_match.group(1).capitalize()
        
        # Add back the context if it was present
        if "(Based on Evidence)" in label_str:
            return f"{base_label} (Based on Evidence)"
        return base_label
    
    # Return the original if we couldn't normalize it
    return label_str