"""
Main Streamlit application for the Fake News Detector.

This module implements the user interface for claim verification,
rendering the results and handling user interactions. It also
manages the application lifecycle including initialization and cleanup.
"""

import streamlit as st
import time
import json
import os
import logging
import atexit
import sys
from pathlib import Path

# Configure logging first, before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("misinformation_detector")

# Check for critical environment variables
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not set. Please configure this in your Hugging Face Spaces secrets.")
    
# Import our modules
from utils.models import initialize_models
from utils.performance import PerformanceTracker

# Import agent functionality
import agent

# Initialize performance tracker
performance_tracker = PerformanceTracker()

# Ensure data directory exists
data_dir = Path("data")
if not data_dir.exists():
    logger.info("Creating data directory")
    data_dir.mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="AskVeracity",
    page_icon="🔍",
    layout="wide",
)

# Hide the "Press ⌘+Enter to apply" text with CSS
st.markdown("""
<style>
/* Hide the shortcut text that appears at the bottom of text areas */
.stTextArea div:has(textarea) + div {
    visibility: hidden !important;
    height: 0px !important;
    position: absolute !important;
}
</style>
""", unsafe_allow_html=True)

def reset_claim_specific_state():
    """
    Reset claim-specific state while preserving model caching.
    
    This function resets only the state variables related to the processing
    of a specific claim, without clearing cached models to maintain efficiency.
    """
    logger.info("Resetting claim-specific state")
    
    # Reset performance tracker metrics but not the instance itself
    global performance_tracker
    performance_tracker.reset()
    
    # Clear session state variables related to the current claim
    if 'result' in st.session_state:
        st.session_state.result = None
    
    if 'has_result' in st.session_state:
        st.session_state.has_result = False
        
    if 'claim_to_process' in st.session_state:
        st.session_state.claim_to_process = ""
        
    # If we already have an agent, keep the instance but ensure it starts fresh
    if hasattr(st.session_state, 'agent') and st.session_state.agent:
        # Recreate the agent to ensure fresh state
        try:
            logger.info("Refreshing agent state for new claim processing")
            # We're keeping the cached models but reinitializing the agent
            st.session_state.agent = agent.setup_agent()
        except Exception as e:
            logger.error(f"Error refreshing agent: {e}")

@st.cache_resource
def get_agent():
    """
    Initialize and cache the agent for reuse across requests.
    
    This function creates and caches the fact-checking agent to avoid
    recreating it for every request. It's decorated with st.cache_resource
    to ensure the agent is only initialized once per session.
    
    Returns:
        object: Initialized LangGraph agent for fact checking
    """
    logger.info("Initializing models and agent (cached)")
    initialize_models()
    return agent.setup_agent()

def cleanup_resources():
    """
    Clean up resources when app is closed.
    
    This function is registered with atexit to ensure resources
    are properly released when the application terminates.
    """
    try:
        # Clear any cached data
        st.cache_data.clear()
        
        # Reset performance tracker
        performance_tracker.reset()
        
        # Log cleanup
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup handler
atexit.register(cleanup_resources)

# App title and description
st.title("🔍 AskVeracity")
st.markdown("""
            This is a simple AI-powered agentic tool - a fact-checking system that analyzes claims to determine 
            their truthfulness by gathering and analyzing evidence from various sources, such as Wikipedia, 
            news outlets, and academic repositories. The application aims to support broader efforts in misinformation detection.
""")

# Sidebar with app information
with st.sidebar:
    st.header("About")
    st.info(
        "This system uses a combination of NLP techniques and LLMs to "
        "extract claims, gather evidence, and classify the truthfulness of statements.\n\n"
        "**Technical:** Built with Python, Streamlit, LangGraph, and OpenAI, leveraging spaCy for NLP and various APIs for retrieving evidence from diverse sources."
    )
    
    # Application information
    st.markdown("### How It Works")
    st.info(
        "1. Enter any recent news or a factual claim\n"
        "2. Our AI gathers evidence from Wikipedia, news sources, and academic repositories\n"
        "3. The system analyzes the evidence to determine truthfulness\n"
        "4. Results show the verdict with supporting evidence"
    )
    
    # Our Mission
    st.markdown("### Our Mission")
    st.info(
        "AskVeracity aims to combat misinformation in real-time through an open-source application built with accessible tools. "
        "We believe in empowering people with factual information to make informed decisions."
    )
    
    # Limitations and Usage
    st.markdown("### Limitations")
    st.warning(
        "Due to resource constraints, AskVeracity may not always provide real-time results with perfect accuracy. "
        "Performance is typically best with widely-reported news and information published within the last 48 hours. "
        "Additionally, the system evaluates claims based on current evidence - a claim that was true in the past "
        "may be judged false if circumstances have changed, and vice versa."
        "Currently, AskVeracity is only available in English."
    )
    
    # Best Practices
    st.markdown("### Best Practices")
    st.success(
        "For optimal results:\n\n"
        "• Keep claims short and precise\n\n"
        "• Each part of the claim is important\n\n"        
        "• Include key details in your claim\n\n"
        "• Phrase claims as direct statements rather than questions\n\n"
        "• Be specific about who said what\n\n"
        "• For very recent announcements or technical features, try checking company blogs, official documentation, or specialized tech news sites directly\n\n"
        "• If receiving an \"Uncertain\" verdict, try alternative phrasings or more general versions of the claim\n\n"
        "• Consider that some technical features might be in limited preview programs with minimal public documentation"
    )
    
    # Example comparison
    with st.expander("📝 Examples of Effective Claims"):
        st.markdown("""
        **Less precise:** "Country A-Country B Relations Are Moving in Positive Direction as per Country B Minister John Doe."
        
        **More precise:** "Country B's External Affairs Minister John Doe has claimed that Country A-Country B Relations Are Moving in Positive Direction."
        """)
    
    # Important Notes
    st.markdown("### Important Notes")
    st.info(
        "• AskVeracity covers general topics and is not specialized in any single domain or location\n\n"
        "• Results can vary based on available evidence and LLM behavior\n\n"
        "• The system is designed to indicate uncertainty when evidence is insufficient\n\n"
        "• AskVeracity is not a chatbot and does not maintain conversation history\n\n"
        "• We recommend cross-verifying critical information with additional sources"
    )
    
    # Privacy Information
    st.markdown("### Data Privacy")
    st.info(
        "We do not collect or store any data about the claims you submit. "
        "Your interactions are processed by OpenAI's API. Please refer to "
        "[OpenAI's privacy policy](https://openai.com/policies/privacy-policy) for details on their data handling practices."
    )
    
    # Feedback Section
    st.markdown("### Feedback")
    st.success(
        "AskVeracity is evolving and we welcome your feedback to help us improve. "
        "Please reach out to us with questions, suggestions, or concerns."
    )

# Initialize session state variables
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'claim_to_process' not in st.session_state:
    st.session_state.claim_to_process = ""
if 'has_result' not in st.session_state:
    st.session_state.has_result = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0
if 'fresh_state' not in st.session_state:
    st.session_state.fresh_state = True
# Initialize verify button disabled state
if 'verify_btn_disabled' not in st.session_state:
    st.session_state.verify_btn_disabled = False
# Add a new state to track input content
if 'input_content' not in st.session_state:
    st.session_state.input_content = ""

# Main interface
st.markdown("### Enter a claim to verify")

# Define a callback for input change
def on_input_change():
    st.session_state.input_content = st.session_state.claim_input_area

# Input area with callback - key fix here!
claim_input = st.text_area("", 
                         value=st.session_state.input_content,
                         height=100,
                         placeholder=(
                             "Examples: The Eiffel Tower is located in Rome, Italy. "
                             "Meta recently released its Llama 4 large language model. "
                             "Justin Trudeau is not the Canadian Prime Minister anymore. "
                             "A recent piece of news."
                         ),
                         key="claim_input_area",
                         on_change=on_input_change,
                         label_visibility="collapsed", 
                         max_chars=None,
                         disabled=st.session_state.processing)

# Add information about claim formatting
st.info("""
**Tip for more accurate results:**
- As older news tends to get deprioritized by sources, trying recent news may yield better results       
- Try news claims as they appear in the sources
- For claims older than 36 hours, consider rephrasing the claim by removing time-sensitive words like "recently," "today," "now," etc.
- Rephrase verbs from present tense to past tense for older events. Examples below:
  - Instead of "launches/unveils/releases" → use "has launched/unveiled/released"
  - Instead of "announces/invites/retaliates/ends" → use "has announced/invited/retaliated/ended"
""")

# Information about result variability
st.caption("""
💡 **Note:** Results may vary slightly each time, even for the same claim. This is by design, allowing our system to:
- Incorporate the most recent evidence available
- Benefit from the AI's ability to consider multiple perspectives
- Adapt to evolving information landscapes
""")

st.warning("⏱️ **Note:** Processing times may vary from 10 seconds to 3 minutes depending on query complexity, available evidence, and current API response times.")

# Create a clean interface based on state
if st.session_state.fresh_state:
    # Only show the verify button in fresh state
    verify_button = st.button(
        "Verify Claim", 
        type="primary", 
        key="verify_btn",
        disabled=st.session_state.verify_btn_disabled
    )
    
    # When button is clicked and not already processing
    if verify_button and not st.session_state.processing:
        # Only show error if claim input is completely empty
        if not claim_input or claim_input.strip() == "":
            st.error("Please enter a claim to verify.")
        else:
            # Reset claim-specific state before processing a new claim
            reset_claim_specific_state()
            
            # Store the claim and set processing state
            st.session_state.claim_to_process = claim_input
            st.session_state.processing = True
            st.session_state.fresh_state = False
            st.session_state.verify_btn_disabled = True
            # Force a rerun to refresh UI
            st.rerun()
            
else:
    # This is either during processing or showing results
    
    # Create a container for processing and results
    analysis_container = st.container()
    
    with analysis_container:
        # If we're processing, show the processing UI
        if st.session_state.processing:
            st.subheader("🔄 Processing...")
            status = st.empty()
            status.text("Verifying claim... (this may take a while)")
            progress_bar = st.progress(0)
            
            # Initialize models and agent if needed
            if not hasattr(st.session_state, 'agent_initialized'):
                with st.spinner("Initializing system..."):
                    st.session_state.agent = get_agent()
                    st.session_state.agent_initialized = True
            
            try:
                # Use the stored claim for processing
                claim_to_process = st.session_state.claim_to_process
                
                # Process the claim with the agent
                start_time = time.time()
                result = agent.process_claim(claim_to_process, st.session_state.agent)
                total_time = time.time() - start_time
                
                # Update progress as claim processing completes
                progress_bar.progress(100)
                
                # Check for None result
                if result is None:
                    st.error("Failed to process the claim. Please try again.")
                    st.session_state.processing = False
                    st.session_state.fresh_state = True
                    st.session_state.verify_btn_disabled = False
                else:
                    # If result exists but key values are missing, provide default values
                    if "classification" not in result or result["classification"] is None:
                        result["classification"] = "Uncertain"
                        
                    if "confidence" not in result or result["confidence"] is None:
                        result["confidence"] = 0.0  # Default to 0.0
                        
                    if "explanation" not in result or result["explanation"] is None:
                        result["explanation"] = "Insufficient evidence was found to determine the truthfulness of this claim."
                    
                    # Update result with timing information
                    if "processing_times" not in result:
                        result["processing_times"] = {"total": total_time}
                    
                    # Store the result and timing information
                    st.session_state.result = result
                    st.session_state.total_time = total_time
                    st.session_state.has_result = True
                    st.session_state.processing = False
                    
                    # Clear processing indicators before showing results
                    status.empty()
                    progress_bar.empty()
                    
                    # Force rerun to display results
                    st.rerun()
                    
            except Exception as e:
                # Handle any exceptions and reset processing state
                logger.error(f"Error during claim processing: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
                st.session_state.processing = False
                st.session_state.fresh_state = True
                st.session_state.verify_btn_disabled = False
                # Force rerun to re-enable button
                st.rerun()
                
        # Display results if available
        elif st.session_state.has_result and st.session_state.result:
            result = st.session_state.result
            total_time = st.session_state.total_time
            claim_to_process = st.session_state.claim_to_process
            
            st.subheader("📊 Verification Results")
            
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                # Display only the original claim
                st.markdown(f"**Claim:** {claim_to_process}")
                
                # Make verdict colorful based on classification
                truth_label = result.get('classification', 'Uncertain')
                if truth_label and "True" in truth_label:
                    verdict_color = "green"
                elif truth_label and "False" in truth_label:
                    verdict_color = "red"
                else:
                    verdict_color = "gray"
                    
                st.markdown(f"**Verdict:** <span style='color:{verdict_color};font-size:1.2em'>{truth_label}</span>", unsafe_allow_html=True)
                
                # Ensure confidence value is used
                if "confidence" in result and result["confidence"] is not None:
                    confidence_value = result["confidence"]
                    # Make sure confidence is a numeric value between 0 and 1
                    try:
                        confidence_value = float(confidence_value)
                        if confidence_value < 0:
                            confidence_value = 0.0
                        elif confidence_value > 1:
                            confidence_value = 1.0
                    except (ValueError, TypeError):
                        confidence_value = 0.0  # Fallback to zero confidence
                else:
                    confidence_value = 0.0  # Default confidence

                # Display the confidence
                st.markdown(f"**Confidence:** {confidence_value:.2%}")
                
                # Display low confidence warning if applicable
                if 0 < confidence_value < 0.1:
                    st.warning("⚠️ **Very Low Confidence:** This result has very low confidence. Please verify with other authoritative sources.")
                
                # Display explanation
                st.markdown(f"**Explanation:** {result.get('explanation', 'No explanation available.')}")
                
                # Add disclaimer about cross-verification
                st.info("⚠️ **Note:** Please cross-verify important information with additional reliable sources.")
                
                if truth_label == "Uncertain":
                    st.info("💡 **Tip for Uncertain Results:** This claim might be too recent for our sources, too specialized, or might not be widely reported or supported with evidence. It is also possible that the claim does not fall into any of these categories and our system may have failed to fetch the correct evidence. Try checking official sites and blogs, news sites, or other related documentation for more information.")
            
            with result_col2:
                st.markdown("**Processing Time**")
                times = result.get("processing_times", {"total": total_time})
                st.markdown(f"- **Total:** {times.get('total', total_time):.2f}s")
                
                # Show agent thoughts
                if "thoughts" in result and result["thoughts"]:
                    st.markdown("**AI Reasoning Process**")
                    thoughts = result.get("thoughts", [])
                    for i, thought in enumerate(thoughts[:5]):  # Show top 5 thoughts
                        st.markdown(f"{i+1}. {thought}")
                    if len(thoughts) > 5:
                        with st.expander("Show all reasoning steps"):
                            for i, thought in enumerate(thoughts):
                                st.markdown(f"{i+1}. {thought}")
            
            # Display evidence
            st.subheader("📝 Evidence")
            evidence_count = result.get("evidence_count", 0)
            evidence = result.get("evidence", [])
            
            # Ensure evidence is a list
            if not isinstance(evidence, list):
                if isinstance(evidence, str):
                    # Try to parse string as a list
                    try:
                        import ast
                        parsed_evidence = ast.literal_eval(evidence)
                        if isinstance(parsed_evidence, list):
                            evidence = parsed_evidence
                        else:
                            evidence = [evidence]
                    except:
                        evidence = [evidence]
                else:
                    evidence = [str(evidence)] if evidence else []
            
            # Update evidence count based on actual evidence list
            evidence_count = len(evidence)
            
            # Get classification results
            classification_results = result.get("classification_results", [])
            
            # Check for empty evidence
            if evidence_count == 0 or not any(ev for ev in evidence if ev):
                st.warning("No relevant evidence was found for this claim. The verdict may not be reliable.")
            else:
                # Add message about processing large number of evidence items
                st.info("The system processes a large number of evidence items across multiple sources and provides a response based on the top relevant evidence items.")
                
                # Filter to only show support and contradict evidence
                if classification_results:
                    support_evidence = []
                    contradict_evidence = []
                    
                    # Extract supporting and contradicting evidence
                    for res in classification_results:
                        if isinstance(res, dict) and "label" in res and "evidence" in res:
                            if res.get("label") == "support":
                                support_evidence.append(res)
                            elif res.get("label") == "contradict":
                                contradict_evidence.append(res)
                    
                    # Sort by confidence
                    support_evidence.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    contradict_evidence.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    
                    # Show counts of relevant evidence
                    st.markdown(f"Found {len(support_evidence)} supporting and {len(contradict_evidence)} contradicting evidence items")
                    
                    # Only show evidence tabs if we have evidence
                    if evidence and any(ev for ev in evidence if ev):
                        # Create tabs for supporting and contradicting evidence only (removed All Evidence tab)
                        evidence_tabs = st.tabs(["Supporting Evidence", "Contradicting Evidence", "Source Details"])
                        
                        # Supporting Evidence tab
                        with evidence_tabs[0]:
                            if support_evidence:
                                for i, res in enumerate(support_evidence):
                                    evidence_text = res.get("evidence", "")
                                    confidence = res.get("confidence", 0)
                                    reason = res.get("reason", "No reason provided")
                                    
                                    if evidence_text and isinstance(evidence_text, str) and evidence_text.strip():
                                        with st.expander(f"Supporting Evidence {i+1} (Confidence: {confidence:.2%})", expanded=i==0):
                                            st.text(evidence_text)
                                            st.markdown(f"**Reason:** {reason}")
                            else:
                                st.info("No supporting evidence was found for this claim.")
                        
                        # Contradicting Evidence tab
                        with evidence_tabs[1]:
                            if contradict_evidence:
                                for i, res in enumerate(contradict_evidence):
                                    evidence_text = res.get("evidence", "")
                                    confidence = res.get("confidence", 0)
                                    reason = res.get("reason", "No reason provided")
                                    
                                    if evidence_text and isinstance(evidence_text, str) and evidence_text.strip():
                                        with st.expander(f"Contradicting Evidence {i+1} (Confidence: {confidence:.2%})", expanded=i==0):
                                            st.text(evidence_text)
                                            st.markdown(f"**Reason:** {reason}")
                            else:
                                st.info("No contradicting evidence was found for this claim.")
                        
                        # Source Details tab (keeping original functionality)
                        with evidence_tabs[2]:
                            st.markdown("The system evaluates evidence from various sources to determine the verdict.")
                            
                            evidence_sources = {}
                            for ev in evidence:
                                if not ev or not isinstance(ev, str):
                                    continue
                                    
                                source = "Unknown"
                                # Extract source info from evidence text
                                if "URL:" in ev:
                                    import re
                                    url_match = re.search(r'URL: https?://(?:www\.)?([^/]+)', ev)
                                    if url_match:
                                        source = url_match.group(1)
                                elif "Source:" in ev:
                                    import re
                                    source_match = re.search(r'Source: ([^,]+)', ev)
                                    if source_match:
                                        source = source_match.group(1)
                                
                                if source in evidence_sources:
                                    evidence_sources[source] += 1
                                else:
                                    evidence_sources[source] = 1
                            
                            # Display evidence source distribution
                            if evidence_sources:
                                st.markdown("**Evidence Source Distribution**")
                                for source, count in evidence_sources.items():
                                    st.markdown(f"- {source}: {count} item(s)")
                            else:
                                st.info("No source information available in the evidence.")
                    else:
                        st.warning("No evidence was retrieved for this claim.")
                else:
                    # Fallback if no classification results
                    st.markdown(f"Retrieved {evidence_count} pieces of evidence, but none were classified as supporting or contradicting.")
                    st.warning("No supporting or contradicting evidence was found for this claim.")
                
            # Button to start a new verification
            if st.button("Verify Another Claim", type="primary", key="new_verify_btn"):
                # Reset all necessary state variables
                st.session_state.fresh_state = True
                st.session_state.has_result = False
                st.session_state.result = None
                st.session_state.processing = False
                st.session_state.claim_to_process = ""
                st.session_state.verify_btn_disabled = False
                # Clear the input field by resetting the input_content
                st.session_state.input_content = ""
                st.rerun()

# Footer with additional information
st.markdown("---")
st.caption("""
**AskVeracity** is an open-source tool designed to help combat misinformation through transparent evidence gathering and analysis. 
While we strive for accuracy, the system has inherent limitations based on available data sources, API constraints, and the evolving nature of information.
""")