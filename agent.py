"""
Agent module for the Fake News Detector application.

This module implements a LangGraph-based agent that orchestrates
the fact-checking process. It defines the agent setup, tools,
and processing pipeline for claim verification.
"""

import os
import time
import logging
import traceback
import json
import ast
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent

from utils.models import get_llm_model
from utils.performance import PerformanceTracker
from modules.claim_extraction import extract_claims
from modules.evidence_retrieval import retrieve_combined_evidence
from modules.classification import classify_with_llm, aggregate_evidence
from modules.explanation import generate_explanation

# Configure logger
logger = logging.getLogger("misinformation_detector")

# Reference to global performance tracker
performance_tracker = PerformanceTracker()

# Define LangGraph Tools
@tool
def claim_extractor(query):
    """
    Tool that extracts factual claims from a given text.
    
    Args:
        query (str): Text containing potential factual claims
        
    Returns:
        str: Extracted factual claim
    """
    performance_tracker.log_claim_processed()
    return extract_claims(query)

@tool
def evidence_retriever(query):
    """
    Tool that retrieves evidence from multiple sources for a claim.
    
    Args:
        query (str): The factual claim to gather evidence for
        
    Returns:
        list: List of evidence items from various sources
    """
    return retrieve_combined_evidence(query)

@tool
def truth_classifier(query, evidence):
    """
    Tool that classifies the truthfulness of a claim based on evidence.
    
    This function analyzes the provided evidence to determine if a claim is true,
    false, or uncertain. It implements a weighted scoring approach considering
    both the number of supporting/contradicting evidence items and their quality.
    
    Args:
        query (str): The factual claim to classify
        evidence (list): Evidence items to evaluate against the claim
        
    Returns:
        str: JSON string containing verdict, confidence, and classification results
             with a guaranteed structure for consistent downstream processing
    """
    # Perform classification on the evidence
    classification_results = classify_with_llm(query, evidence)
    
    # Aggregate results to determine overall verdict and confidence
    truth_label, confidence = aggregate_evidence(classification_results)
    
    # Debug logging
    logger.info(f"Classification results: {len(classification_results)} items")
    logger.info(f"Aggregate result: {truth_label}, confidence: {confidence}")
    
    # Ensure truth_label is never None
    if not truth_label:
        truth_label = "Uncertain"
        confidence = 0.0
    
    # Return a structured dictionary with all needed information
    result = {
        "verdict": truth_label,
        "confidence": confidence,
        "results": classification_results
    }

    # Log confidence score
    performance_tracker.log_confidence_score(confidence)
    
    # Convert to JSON string for consistent handling
    return json.dumps(result)

@tool
def explanation_generator(claim, evidence_results, truth_label):
    """
    Tool that generates a human-readable explanation for the verdict.
    
    This function creates a clear, natural language explanation of why a claim
    was classified as true, false, or uncertain based on the evidence. It handles
    various truth label formats and extracts appropriate confidence values.
    
    Args:
        claim (str): The factual claim being verified
        evidence_results (list): Evidence items and classification results
        truth_label (str): The verdict (True/False/Uncertain), which may come
                          in different formats
        
    Returns:
        str: Natural language explanation of the verdict with confidence
             framing and evidence citations
             
    Note:
        The function extracts confidence values from evidence when available
        or uses appropriate defaults based on the verdict type. It includes
        robust error handling to ensure explanations are always generated,
        even in edge cases.
    """
    try:
        # Extract confidence if available in evidence_results
        confidence = None
        if isinstance(evidence_results, list) and evidence_results and isinstance(evidence_results[0], dict):
            # Try to get confidence from results
            confidence_values = [result.get('confidence', 0) for result in evidence_results if 'confidence' in result]
            if confidence_values:
                confidence = max(confidence_values)
        
        # If confidence couldn't be extracted, use a default value based on the verdict
        if confidence is None:
            if truth_label and ("True" in truth_label or "False" in truth_label):
                confidence = 0.7  # Default for definitive verdicts
            else:
                confidence = 0.5  # Default for uncertain verdicts
        
        # Generate the explanation
        explanation = generate_explanation(claim, evidence_results, truth_label, confidence)
        logger.info(f"Generated explanation: {explanation[:100]}...")
        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        # Provide a fallback explanation with basic information
        return f"The claim '{claim}' has been evaluated as {truth_label}. The available evidence provides {confidence or 'moderate'} confidence in this assessment. For more detailed information, please review the evidence provided."
    
def setup_agent():
    """
    Create and configure a ReAct agent with the fact-checking tools.
    
    This function configures a LangGraph ReAct agent with all the
    necessary tools for fact checking, including claim extraction,
    evidence retrieval, classification, and explanation generation.
    
    Returns:
        object: Configured LangGraph agent ready for claim processing
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    # Make sure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        logger.error("OPENAI_API_KEY environment variable not set or empty.")
        raise ValueError("OpenAI API key is required")

    # Define tools with any customizations
    tools = [
        claim_extractor,
        evidence_retriever,
        truth_classifier,
        explanation_generator
    ]

    # Define the prompt template with clearer, more efficient instructions
    FORMAT_INSTRUCTIONS_TEMPLATE = """
    Use the following format:
    Question: the input question you must answer
    Action: the action to take, should be one of: {tool_names}
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Action/Action Input/Observation can repeat N times)
    Final Answer: the final answer to the original input question
    """
    
    prompt = PromptTemplate(
        input_variables=["input", "tool_names"],
        template=f"""
        You are a fact-checking assistant that verifies claims by gathering evidence and 
        determining their truthfulness. Follow these exact steps in sequence:

        1. Call claim_extractor to extract the main factual claim
        2. Call evidence_retriever to gather evidence about the claim
        3. Call truth_classifier to evaluate the claim using the evidence
        4. Call explanation_generator to explain the result
        5. Provide your Final Answer that summarizes everything

        Execute these steps in order without unnecessary thinking steps between tool calls.
        Be direct and efficient in your verification process.
        
        {FORMAT_INSTRUCTIONS_TEMPLATE}
        """
    )
    
    try:
        # Get the LLM model
        model = get_llm_model()
        
        # Create the agent with a shorter timeout
        graph = create_react_agent(model, tools=tools)
        logger.info("Agent created successfully")
        return graph
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise e

def process_claim(claim, agent=None, recursion_limit=20):
    """
    Process a claim to determine its truthfulness using the agent.
    
    This function invokes the LangGraph agent to process a factual claim,
    extract supporting evidence, evaluate the claim's truthfulness, and
    generate a human-readable explanation.
    
    Args:
        claim (str): The factual claim to be verified
        agent (object, optional): Initialized LangGraph agent. If None, an error is logged.
        recursion_limit (int, optional): Maximum recursion depth for agent. Default: 20.
            Higher values allow more complex reasoning but increase processing time.
            
    Returns:
        dict: Result dictionary containing:
            - claim: Extracted factual claim
            - evidence: List of evidence pieces
            - evidence_count: Number of evidence pieces
            - classification: Verdict (True/False/Uncertain)
            - confidence: Confidence score (0-1)
            - explanation: Human-readable explanation of the verdict
            - final_answer: Final answer from the agent
            - Or error information if processing failed
    """
    if agent is None:
        logger.error("Agent not initialized. Call setup_agent() first.")
        return None
        
    start_time = time.time()
    logger.info(f"Processing claim with agent: {claim}")
    
    try:
        # IMPORTANT: Create fresh inputs for each claim
        # This ensures we don't carry over state from previous claims
        inputs = {"messages": [("user", claim)]}
        
        # Set configuration - reduced recursion limit for faster processing
        config = {"recursion_limit": recursion_limit}
        
        # Invoke the agent
        response = agent.invoke(inputs, config)
        
        # Format the response
        result = format_response(response)
        
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"Claim processed in {elapsed:.2f} seconds")

        # Track processing time and overall success
        performance_tracker.log_processing_time(start_time)
        performance_tracker.log_claim_processed()
        
        # Track confidence if available
        if result and "confidence" in result:
            performance_tracker.log_confidence_score(result["confidence"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing claim with agent: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def format_response(response):
    """
    Format the agent's response into a structured result.
    
    This function extracts key information from the agent's response,
    including the claim, evidence, classification, and explanation.
    It also performs error handling and provides fallback values.
    
    Args:
        response (dict): Raw response from the LangGraph agent
        
    Returns:
        dict: Structured result containing claim verification data
    """
    try:
        if not response or "messages" not in response:
            return {"error": "Invalid response format"}
            
        messages = response.get("messages", [])
        
        # Initialize result container with default values
        result = {
            "claim": None,
            "evidence": [],
            "evidence_count": 0,
            "classification": "Uncertain",
            "confidence": 0.0,  # Default zero confidence
            "explanation": "Insufficient evidence to evaluate this claim.",
            "final_answer": None,
            "thoughts": []
        }
        
        # Track if we found results from each tool
        found_tools = {
            "claim_extractor": False,
            "evidence_retriever": False,
            "truth_classifier": False,
            "explanation_generator": False
        }
        
        # Extract information from messages
        tool_outputs = {}
        
        for idx, message in enumerate(messages):
            # Extract agent thoughts
            if hasattr(message, "content") and getattr(message, "type", "") == "assistant":
                content = message.content
                if "Thought:" in content:
                    thought_parts = content.split("Thought:", 1)
                    if len(thought_parts) > 1:
                        thought = thought_parts[1].split("\n")[0].strip()
                        result["thoughts"].append(thought)
            
            # Extract tool outputs
            if hasattr(message, "type") and message.type == "tool":
                tool_name = getattr(message, "name", "unknown")
                
                # Store tool outputs
                tool_outputs[tool_name] = message.content
                
                # Extract specific information
                if tool_name == "claim_extractor":
                    found_tools["claim_extractor"] = True
                    if message.content:
                        result["claim"] = message.content
                    
                elif tool_name == "evidence_retriever":
                    found_tools["evidence_retriever"] = True
                    # Handle string representation of a list
                    if message.content:
                        if isinstance(message.content, list):
                            result["evidence"] = message.content
                            result["evidence_count"] = len(message.content)
                        elif isinstance(message.content, str) and message.content.startswith("[") and message.content.endswith("]"):
                            try:
                                parsed_content = ast.literal_eval(message.content)
                                if isinstance(parsed_content, list):
                                    result["evidence"] = parsed_content
                                    result["evidence_count"] = len(parsed_content)
                                else:
                                    result["evidence"] = [message.content]
                                    result["evidence_count"] = 1
                            except:
                                result["evidence"] = [message.content]
                                result["evidence_count"] = 1
                        else:
                            result["evidence"] = [message.content]
                            result["evidence_count"] = 1
                            logger.warning(f"Evidence retrieved is not a list: {type(message.content)}")
                        
                elif tool_name == "truth_classifier":
                    found_tools["truth_classifier"] = True
                    
                    # Log the incoming content for debugging
                    logger.info(f"Truth classifier content type: {type(message.content)}")
                    logger.info(f"Truth classifier content: {message.content}")
                    
                    # Handle JSON formatted result from truth_classifier()
                    if isinstance(message.content, str):
                        try:
                            # Parse the JSON string
                            parsed_content = json.loads(message.content)
                            
                            # Extract the values from the parsed content
                            result["classification"] = parsed_content.get("verdict", "Uncertain")
                            result["confidence"] = float(parsed_content.get("confidence", 0.0))
                            result["classification_results"] = parsed_content.get("results", [])
                            
                            # Add low confidence warning for results < 10%
                            if 0 < result["confidence"] < 0.1:
                                result["low_confidence_warning"] = True
                            
                            logger.info(f"Extracted from JSON: verdict={result['classification']}, confidence={result['confidence']}")
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse truth classifier JSON: {message.content}")
                        except Exception as e:
                            logger.warning(f"Error extracting from truth classifier output: {e}")
                    else:
                        logger.warning(f"Unexpected truth_classifier content format: {message.content}")
                        
                elif tool_name == "explanation_generator":
                    found_tools["explanation_generator"] = True
                    if message.content:
                        result["explanation"] = message.content
                        logger.info(f"Found explanation from tool: {message.content[:100]}...")
                    
            # Get final answer from last message
            elif idx == len(messages) - 1 and hasattr(message, "content"):
                result["final_answer"] = message.content
        
        # Log which tools weren't found
        missing_tools = [tool for tool, found in found_tools.items() if not found]
        if missing_tools:
            logger.warning(f"Missing tool outputs in response: {', '.join(missing_tools)}")
        
        # IMPORTANT: ENHANCED FALLBACK MECHANISM
        # Always run truth classification if evidence was collected but classifier wasn't called
        if found_tools["evidence_retriever"] and not found_tools["truth_classifier"]:
            logger.info("Truth classifier was not called by the agent, executing fallback classification")
            
            try:                
                # Get the evidence from the results
                evidence = result["evidence"]
                claim = result["claim"] or "Unknown claim"
                
                # Force classification even with minimal evidence
                if evidence:
                    # Classify with available evidence
                    classification_results = classify_with_llm(claim, evidence)
                    truth_label, confidence = aggregate_evidence(classification_results)
                    
                    # Update result with classification results
                    result["classification"] = truth_label
                    result["confidence"] = confidence
                    result["classification_results"] = classification_results
                    
                    # Add low confidence warning if needed
                    if 0 < confidence < 0.1:
                        result["low_confidence_warning"] = True
                    
                    logger.info(f"Fallback classification: {truth_label}, confidence: {confidence}")
                else:
                    # If no evidence at all, maintain uncertain with zero confidence
                    result["classification"] = "Uncertain"
                    result["confidence"] = 0.0
                    logger.info("No evidence available for fallback classification")
            except Exception as e:
                logger.error(f"Error in fallback truth classification: {e}")
        
        # ENHANCED: Always generate explanation if classification exists but explanation wasn't called
        if (found_tools["truth_classifier"] or result["classification"] != "Uncertain") and not found_tools["explanation_generator"]:
            logger.info("Explanation generator was not called by the agent, using fallback explanation generation")
            
            try:                
                # Get the necessary inputs for explanation generation
                claim = result["claim"] or "Unknown claim"
                evidence = result["evidence"]
                truth_label = result["classification"] 
                confidence_value = result["confidence"]
                classification_results = result.get("classification_results", [])
                
                # Choose the best available evidence for explanation
                explanation_evidence = classification_results if classification_results else evidence
                
                # Force explanation generation even with minimal evidence
                explanation = generate_explanation(claim, explanation_evidence, truth_label, confidence_value)
                
                # Use the generated explanation
                if explanation:
                    logger.info(f"Generated fallback explanation: {explanation[:100]}...")
                    result["explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating fallback explanation: {e}")
        
        # Make sure evidence exists
        if result["evidence_count"] > 0 and (not result["evidence"] or len(result["evidence"]) == 0):
            logger.warning("Evidence count is non-zero but evidence list is empty. This is a data inconsistency.")
            result["evidence_count"] = 0
        
        # Add debug info about the final result
        logger.info(f"Final classification: {result['classification']}, confidence: {result['confidence']}")
        logger.info(f"Final explanation: {result['explanation'][:100]}...")
        
        # Add performance metrics
        result["performance"] = performance_tracker.get_summary()
        
        # Memory management - limit the size of evidence and thoughts
        # To keep memory usage reasonable for web deployment
        if "evidence" in result and isinstance(result["evidence"], list):
            limited_evidence = []
            for ev in result["evidence"]:
                if isinstance(ev, str) and len(ev) > 500:
                    limited_evidence.append(ev[:497] + "...")
                else:
                    limited_evidence.append(ev)
            result["evidence"] = limited_evidence
            
        # Limit thoughts to conserve memory
        if "thoughts" in result and len(result["thoughts"]) > 10:
            result["thoughts"] = result["thoughts"][:10]
        
        return result
        
    except Exception as e:
        logger.error(f"Error formatting agent response: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e), 
            "traceback": traceback.format_exc(),
            "classification": "Error",
            "confidence": 0.0,
            "explanation": "An error occurred while processing this claim."
        }