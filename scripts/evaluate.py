#!/usr/bin/env python3
"""
Evaluation script for trained models
"""

import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import ModelEvaluator
from src.inference import SentimentAnalyzer


def main(args):
    """Main evaluation function"""
    
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Initialize evaluator
    print(f"\nLoading model from {args.model_path}")
    evaluator = ModelEvaluator(args.model_path)
    
    # Run evaluation
    print("\nEvaluating on test set...")
    results = evaluator.evaluate_on_test_set()
    
    print(f"\nTest Set Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (weighted): {sum(results['f1'][i] * results['support'][i] for i in range(2)) / sum(results['support']):.4f}")
    
    # Baseline comparison
    if args.compare_baseline:
        print("\nComparing with baseline...")
        baseline_results = evaluator.compare_with_baseline()
        print(f"Baseline Accuracy: {baseline_results['baseline_accuracy']:.4f}")
        print(f"Improvement: {(results['accuracy'] - baseline_results['baseline_accuracy']) * 100:.1f}%")
    
    # Error analysis
    if args.error_analysis:
        print("\nPerforming error analysis...")
        error_analysis = evaluator.error_analysis(results)
        print(f"Total errors: {error_analysis['error_patterns']['total_errors']}")
        print(f"Error rate: {error_analysis['error_patterns']['error_rate']:.2%}")
        print(f"High confidence errors: {len(error_analysis['error_types']['high_conf_errors'])}")
    
    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        evaluator.visualize_results(results, error_analysis, args.output_dir)
        print(f"Visualizations saved to {args.output_dir}")
    
    # Test inference
    if args.test_inference:
        print("\n" + "=" * 50)
        print("Testing Inference Pipeline")
        print("=" * 50)
        
        analyzer = SentimentAnalyzer(args.model_path)
        
        test_texts = [
            "This movie was absolutely fantastic! Best film I've seen all year.",
            "Terrible movie. Complete waste of time and money.",
            "It was okay, nothing special but not terrible either.",
            "The acting was great but the plot was confusing.",
            "10/10 would definitely watch again!!!"
        ]
        
        print("\nSample Predictions:")
        for text in test_texts:
            result = analyzer.predict(text, return_all_scores=True)
            print(f"\nText: {text}")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
            if 'scores' in result:
                print(f"Scores: negative={result['scores']['negative']:.3f}, positive={result['scores']['positive']:.3f}")
    
    # Save results
    if args.save_results:
        results_summary = {
            "model_path": args.model_path,
            "test_accuracy": results['accuracy'],
            "classification_report": results['classification_report']
        }
        
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment analysis model")
    
    parser.add_argument("--model_path", type=str, default="./outputs/best_model",
                        help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for results")
    parser.add_argument("--compare_baseline", action="store_true", default=True,
                        help="Compare with baseline model")
    parser.add_argument("--error_analysis", action="store_true", default=True,
                        help="Perform detailed error analysis")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Create visualization plots")
    parser.add_argument("--test_inference", action="store_true", default=True,
                        help="Test inference pipeline with examples")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save evaluation results to JSON")
    
    args = parser.parse_args()
    main(args)