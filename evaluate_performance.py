#!/usr/bin/env python3
"""
Performance Evaluation Script for AskVeracity.

This script evaluates the performance of the AskVeracity fact-checking system
using a predefined set of test claims with known ground truth labels.
It collects metrics on accuracy, safety rate, processing time, and confidence scores
without modifying the core codebase.

Usage:
    python evaluate_performance.py [--limit N] [--output FILE]

Options:
    --limit N        Limit evaluation to first N claims (default: all)
    --output FILE    Save results to FILE (default: performance_results.json)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the parent directory to sys.path if this script is run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent
import agent
from utils.models import initialize_models

# IMPORTANT NOTE FOR DEVELOPERS:
# The test claims below include many recent events that will become outdated.
# When using this script for testing or evaluation, please update these claims 
# with relevant and up-to-date examples to ensure meaningful results.
# Performance metrics are heavily influenced by the recency and verifiability 
# of these claims, so using outdated claims will likely lead to poor results.

# Define the test claims with ground truth labels
TEST_CLAIMS = [
    # True claims
    {"claim": "Dozens killed as gunmen massacre tourists in Kashmir beauty spot.", "expected": "True"},
    {"claim": "Pope Francis dies at 88.", "expected": "True"},
    {"claim": "OpenAI released new reasoning models called o3 and o4-mini.", "expected": "True"},
    {"claim": "Trump And Zelensky Clash Again As US Says Crimea Now Russian Territory.", "expected": "True"},
    {"claim": "Twelve states sue Donald Trump administration in trade court over chaotic and illegal tariff policy.", "expected": "True"},
    {"claim": "Zomato has been renamed to Eternal Limited.", "expected": "True"},
    {"claim": "The Taj Mahal is located in Agra.", "expected": "True"},
    {"claim": "ISRO achieves second docking with SpaDeX satellites.", "expected": "True"},
    {"claim": "The TV series Adolescence is streaming on Netflix.", "expected": "True"},
    {"claim": "Vladimir Putin offers to halt Ukraine invasion.", "expected": "True"},
    {"claim": "Meta released its Llama 4 language model.", "expected": "True"},
    {"claim": "Google launched Gemini 2.5 Pro Experimental, the first model in the Gemini 2.5 family.", "expected": "True"},
    {"claim": "Microsoft is rolling out improved Recall feature for Windows Insiders.", "expected": "True"},
    {"claim": "Microsoft announced a 1-bit language model that can run on CPU.", "expected": "True"},
    {"claim": "Royal Challengers Bengaluru beat Rajasthan Royals by 11 runs in yesterday's IPL match.", "expected": "True"},
    {"claim": "Anthropic introduced Claude Research.", "expected": "True"},
    {"claim": "The IMF has lowered India's growth projection for the fiscal year 2025-26 to 6.2 per cent.", "expected": "True"},
    {"claim": "In Bundesliga, Bayern Munich beat Heidenheim 4-0 last week.", "expected": "True"},
    {"claim": "Manchester United in Europa League semi-finals.", "expected": "True"},
    
    # False claims
    {"claim": "The Eiffel Tower is in Rome.", "expected": "False"},
    {"claim": "The earth is flat.", "expected": "False"},
    {"claim": "Rishi Sunak is the current Prime Minister of the UK.", "expected": "False"},
    {"claim": "New Zealand won the ICC Champions Trophy in 2025.", "expected": "False"},
    {"claim": "US President Donald trump to visit India next week.", "expected": "False"},
    {"claim": "Quantum computers have definitively solved the protein folding problem.", "expected": "False"},
    {"claim": "CRISPR gene editing has successfully cured type 1 diabetes in human clinical trials.", "expected": "False"},
    {"claim": "Google's new quantum computer, Willow, has demonstrated remarkable capabilities by solving mathematical problems far beyond the reach of the fastest supercomputers.", "expected": "False"},
    {"claim": "NASA confirmed that the James Webb Space Telescope has found definitive evidence of alien life on an exoplanet.", "expected": "False"},
    {"claim": "Google launched Gemini 3.", "expected": "False"},
    {"claim": "A solar eclipse was be seen in India on October 17, 2024.", "expected": "False"},
    {"claim": "Tom Cruise and Shah Rukh Khan have starred in a Bollywood movie in the past.", "expected": "False"},
    {"claim": "Germany has the highest GDP in the world.", "expected": "False"},
    
    # Uncertain claims
    {"claim": "Aliens have visited the Earth.", "expected": "Uncertain"},
    {"claim": "Information that falls into a black hole is permanently lost or destroyed.", "expected": "Uncertain"},
    {"claim": "Time travel into the past is possible.", "expected": "Uncertain"},
    {"claim": "Bigfoot (or Yeti) exists in remote wilderness areas.", "expected": "Uncertain"},
    {"claim": "Intelligent life exists elsewhere in the universe.", "expected": "Uncertain"},
    {"claim": "Yogi Adityanath will be the next Prime Minister of India.", "expected": "Uncertain"},
    {"claim": "Consciousness continues to exist after biological death.", "expected": "Uncertain"},
    {"claim": "There are multiple parallel universes.", "expected": "Uncertain"}
]

def setup_argument_parser():
    """
    Set up command line argument parsing.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate AskVeracity performance")
    parser.add_argument("--limit", type=int, help="Limit evaluation to first N claims")
    parser.add_argument("--output", type=str, default="performance_results.json", 
                        help="Output file for results (default: performance_results.json)")
    return parser.parse_args()

def initialize_system():
    """
    Initialize the system for evaluation.
    
    Returns:
        object: Initialized LangGraph agent
    """
    print("Initializing models and agent...")
    initialize_models()
    eval_agent = agent.setup_agent()
    return eval_agent

def normalize_classification(classification):
    """
    Normalize classification labels for consistent comparison.
    
    Args:
        classification (str): Classification label from the system
        
    Returns:
        str: Normalized classification label ("True", "False", or "Uncertain")
    """
    if not classification:
        return "Uncertain"
    
    if "true" in classification.lower():
        return "True"
    elif "false" in classification.lower():
        return "False"
    else:
        return "Uncertain"

def is_correct(actual, expected):
    """
    Determine if the actual classification matches the expected classification.
    
    Args:
        actual (str): Actual classification from the system
        expected (str): Expected (ground truth) classification
        
    Returns:
        bool: True if classifications match, False otherwise
    """
    # Normalize both for comparison
    normalized_actual = normalize_classification(actual)
    normalized_expected = expected
    
    return normalized_actual == normalized_expected

def is_safe(actual, expected):
    """
    Determine if the classification is "safe" - either correct or abstained (Uncertain)
    instead of making an incorrect assertion.
    
    Args:
        actual (str): Actual classification from the system
        expected (str): Expected (ground truth) classification
        
    Returns:
        bool: True if the classification is safe, False otherwise
    """
    # Normalize both for comparison
    normalized_actual = normalize_classification(actual)
    normalized_expected = expected
    
    # If the classification is correct, it's definitely safe
    if normalized_actual == normalized_expected:
        return True
    
    # If the system classified as "Uncertain", that's safe (abstaining rather than wrong assertion)
    if normalized_actual == "Uncertain":
        return True
    
    # Otherwise, the system made an incorrect assertion (False as True or True as False)
    return False

def evaluate_claims(test_claims, eval_agent, limit=None):
    """
    Evaluate a list of claims using the fact-checking system.
    
    Args:
        test_claims (list): List of test claims with expected classifications
        eval_agent (object): Initialized LangGraph agent
        limit (int, optional): Maximum number of claims to evaluate
        
    Returns:
        tuple: (results, metrics)
            - results (list): Detailed results for each claim
            - metrics (dict): Aggregated performance metrics
    """
    
    # Limit the number of claims if requested
    if limit and limit > 0:
        claims_to_evaluate = test_claims[:limit]
    else:
        claims_to_evaluate = test_claims
    
    results = []
    total_count = len(claims_to_evaluate)
    correct_count = 0
    safe_count = 0
    
    # Classification counts
    classification_counts = {"True": 0, "False": 0, "Uncertain": 0}
    
    # Track processing times by expected classification
    processing_times = {"True": [], "False": [], "Uncertain": []}
    
    # Confidence scores by expected classification
    confidence_scores = {"True": [], "False": [], "Uncertain": []}
    
    # Track correct classifications by expected classification
    correct_by_class = {"True": 0, "False": 0, "Uncertain": 0}
    safe_by_class = {"True": 0, "False": 0, "Uncertain": 0}
    total_by_class = {"True": 0, "False": 0, "Uncertain": 0}
    
    print(f"Evaluating {len(claims_to_evaluate)} claims...")
    
    # Process each claim
    for idx, test_case in enumerate(claims_to_evaluate):
        claim = test_case["claim"]
        expected = test_case["expected"]
        
        print(f"\nProcessing claim {idx+1}/{len(claims_to_evaluate)}: {claim}")
        
        try:
            # Process the claim and measure time
            start_time = time.time()
            result = agent.process_claim(claim, eval_agent)
            total_time = time.time() - start_time
            
            # Extract classification and confidence
            classification = result.get("classification", "Uncertain")
            confidence = result.get("confidence", 0.0)
            
            # Normalize classification for comparison
            normalized_classification = normalize_classification(classification)
            
            # Check if classification is correct
            correct = is_correct(normalized_classification, expected)
            if correct:
                correct_count += 1
                correct_by_class[expected] += 1
            
            # Check if classification is safe
            safe = is_safe(normalized_classification, expected)
            if safe:
                safe_count += 1
                safe_by_class[expected] += 1
            
            # Update classification count
            classification_counts[normalized_classification] = classification_counts.get(normalized_classification, 0) + 1
            
            # Update counts by expected class
            total_by_class[expected] += 1
            
            # Update processing times
            processing_times[expected].append(total_time)
            
            # Update confidence scores
            confidence_scores[expected].append(confidence)
            
            # Save detailed result
            detail_result = {
                "claim": claim,
                "expected": expected,
                "actual": normalized_classification,
                "correct": correct,
                "safe": safe,
                "confidence": confidence,
                "processing_time": total_time
            }
            
            results.append(detail_result)
            
            # Print progress indicator
            outcome = "✓" if correct else "✗"
            safety = "(safe)" if safe and not correct else ""
            print(f"  Result: {normalized_classification} (Expected: {expected}) {outcome} {safety}")
            print(f"  Time: {total_time:.2f}s, Confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"Error processing claim: {str(e)}")
            results.append({
                "claim": claim,
                "expected": expected,
                "error": str(e)
            })
    
    # Calculate performance metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    safety_rate = safe_count / total_count if total_count > 0 else 0
    
    # Calculate per-class metrics
    class_metrics = {}
    for cls in ["True", "False", "Uncertain"]:
        class_accuracy = correct_by_class[cls] / total_by_class[cls] if total_by_class[cls] > 0 else 0
        class_safety_rate = safe_by_class[cls] / total_by_class[cls] if total_by_class[cls] > 0 else 0
        avg_time = sum(processing_times[cls]) / len(processing_times[cls]) if processing_times[cls] else 0
        avg_confidence = sum(confidence_scores[cls]) / len(confidence_scores[cls]) if confidence_scores[cls] else 0
        
        class_metrics[cls] = {
            "accuracy": class_accuracy,
            "safety_rate": class_safety_rate,
            "count": total_by_class[cls],
            "correct": correct_by_class[cls],
            "safe": safe_by_class[cls],
            "avg_processing_time": avg_time,
            "avg_confidence": avg_confidence
        }
    
    # Calculate overall metrics
    all_times = [r.get("processing_time", 0) for r in results if "processing_time" in r]
    all_confidence = [r.get("confidence", 0) for r in results if "confidence" in r]
    
    metrics = {
        "total_claims": total_count,
        "correct_claims": correct_count,
        "safe_claims": safe_count,
        "accuracy": accuracy,
        "safety_rate": safety_rate,
        "avg_processing_time": sum(all_times) / len(all_times) if all_times else 0,
        "avg_confidence": sum(all_confidence) / len(all_confidence) if all_confidence else 0,
        "classification_counts": classification_counts,
        "per_class_metrics": class_metrics
    }
    
    return results, metrics

def save_results(results, metrics, output_file):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results (list): Detailed results for each claim
        metrics (dict): Aggregated performance metrics
        output_file (str): Path to output file
    """
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "detailed_results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def print_summary(metrics):
    """
    Print a summary of performance metrics.
    
    Args:
        metrics (dict): Aggregated performance metrics
    """
    print("\n" + "="*70)
    print(f"PERFORMANCE SUMMARY")
    print("="*70)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"Total Claims: {metrics['total_claims']}")
    print(f"Correctly Classified: {metrics['correct_claims']}")
    print(f"Safely Classified: {metrics['safe_claims']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Safety Rate: {metrics['safety_rate']:.2%}")
    print(f"Average Processing Time: {metrics['avg_processing_time']:.2f} seconds")
    print(f"Average Confidence Score: {metrics['avg_confidence']:.2f}")
    
    # Per-class metrics as table
    print("\nPer-Class Performance:")
    table_data = []
    headers = ["Class", "Count", "Correct", "Safe", "Accuracy", "Safety Rate", "Avg Time", "Avg Confidence"]
    
    for cls, cls_metrics in metrics['per_class_metrics'].items():
        table_data.append([
            cls,
            cls_metrics['count'],
            cls_metrics['correct'],
            cls_metrics['safe'],
            f"{cls_metrics['accuracy']:.2%}",
            f"{cls_metrics['safety_rate']:.2%}",
            f"{cls_metrics['avg_processing_time']:.2f}s",
            f"{cls_metrics['avg_confidence']:.2f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def create_charts(metrics, output_dir="."):
    """
    Create visualizations of performance metrics.
    
    Args:
        metrics (dict): Aggregated performance metrics
        output_dir (str): Directory to save charts
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Accuracy by class
        plt.figure(figsize=(10, 6))
        classes = list(metrics['per_class_metrics'].keys())
        accuracies = [metrics['per_class_metrics'][cls]['accuracy'] for cls in classes]
        
        plt.bar(classes, accuracies, color=['green', 'red', 'gray'])
        plt.title('Accuracy by Classification Type')
        plt.xlabel('Classification')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_class.png'))
        plt.close()  # Close the figure to free memory
        
        # Plot 2: Safety rate by class
        plt.figure(figsize=(10, 6))
        safety_rates = [metrics['per_class_metrics'][cls]['safety_rate'] for cls in classes]
        
        plt.bar(classes, safety_rates, color=['green', 'red', 'gray'])
        plt.title('Safety Rate by Classification Type')
        plt.xlabel('Classification')
        plt.ylabel('Safety Rate')
        plt.ylim(0, 1)
        
        for i, v in enumerate(safety_rates):
            plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'safety_rate_by_class.png'))
        plt.close()  # Close the figure to free memory
        
        # Plot 3: Processing time by class
        plt.figure(figsize=(10, 6))
        times = [metrics['per_class_metrics'][cls]['avg_processing_time'] for cls in classes]
        
        plt.bar(classes, times, color=['green', 'red', 'gray'])
        plt.title('Average Processing Time by Classification Type')
        plt.xlabel('Classification')
        plt.ylabel('Time (seconds)')
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.5, f"{v:.2f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_time_by_class.png'))
        plt.close()  # Close the figure to free memory
        
        # Plot 4: Confidence scores by class
        plt.figure(figsize=(10, 6))
        confidence = [metrics['per_class_metrics'][cls]['avg_confidence'] for cls in classes]
        
        plt.bar(classes, confidence, color=['green', 'red', 'gray'])
        plt.title('Average Confidence Score by Classification Type')
        plt.xlabel('Classification')
        plt.ylabel('Confidence Score')
        plt.ylim(0, 1)
        
        for i, v in enumerate(confidence):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_by_class.png'))
        plt.close()  # Close the figure to free memory
        
        print(f"\nCharts created in {output_dir}")
        
    except Exception as e:
        print(f"Error creating charts: {str(e)}")
        print("Continuing without charts.")

def main():
    """Main evaluation function that runs the entire evaluation process."""
    # Parse arguments
    args = setup_argument_parser()
    
    # Initialize the agent
    eval_agent = initialize_system()
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set output file path
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(results_dir, output_file)
    
    # Evaluate claims
    results, metrics = evaluate_claims(TEST_CLAIMS, eval_agent, args.limit)
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    save_results(results, metrics, output_file)
    
    # Create charts
    create_charts(metrics, results_dir)

if __name__ == "__main__":
    main()