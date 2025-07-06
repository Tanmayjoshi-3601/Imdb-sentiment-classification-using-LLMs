#!/usr/bin/env python3
"""
Main training script for IMDB sentiment analysis
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation import IMDBDataProcessor
from src.model_config import ModelConfigurator
from src.training import ModelTrainer, TRAINING_CONFIGS
from src.evaluation import evaluate_model


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    """Main training pipeline"""
    
    # Set random seeds
    set_seed(args.seed)
    
    print("=" * 50)
    print("IMDB Sentiment Analysis - Fine-Tuning Pipeline")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Step 1: Data Preparation
    if args.prepare_data or not os.path.exists("./data/tokenized_imdb"):
        print("\n1. Preparing Dataset...")
        processor = IMDBDataProcessor(
            model_name=args.model_name,
            max_length=args.max_length
        )
        dataset_dict = processor.load_and_prepare_dataset(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size
        )
        tokenized_datasets = processor.prepare_datasets(dataset_dict)
        dataset_info = processor.save_datasets(tokenized_datasets)
        print("Dataset preparation complete!")
    else:
        print("\n1. Using existing prepared dataset...")
        from datasets import load_from_disk
        tokenized_datasets = load_from_disk("./data/tokenized_imdb")
    
    # Step 2: Model Configuration
    print(f"\n2. Configuring Model: {args.model_name}")
    configurator = ModelConfigurator(args.model_name, num_labels=2)
    model, param_info = configurator.setup_model(
        freeze_embeddings=args.freeze_embeddings,
        freeze_layers=args.freeze_layers
    )
    tokenizer = configurator.get_tokenizer()
    
    # Step 3: Training
    print(f"\n3. Training with {args.config} configuration...")
    trainer_module = ModelTrainer(model, tokenizer)
    
    # Hyperparameter search (optional)
    if args.hp_search:
        print("Running hyperparameter search...")
        study, best_params = trainer_module.hyperparameter_search(n_trials=args.hp_trials)
        print(f"Best parameters found: {best_params}")
    
    # Train with selected configuration
    config = TRAINING_CONFIGS[args.config]
    output_dir = f"./outputs/{args.config}"
    trainer, train_result, eval_result, logs, train_time = trainer_module.train_with_config(
        config, output_dir
    )
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Final eval F1: {eval_result['eval_f1']:.4f}")
    
    # Step 4: Evaluation
    if args.evaluate:
        print("\n4. Evaluating Model...")
        
        # Save the best model
        best_model_path = "./outputs/best_model"
        os.makedirs(best_model_path, exist_ok=True)
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        
        # Run evaluation
        results, baseline_results, error_analysis = evaluate_model(
            best_model_path, trainer
        )
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Baseline Accuracy: {baseline_results['baseline_accuracy']:.4f}")
        print(f"Improvement: {(results['accuracy'] - baseline_results['baseline_accuracy']) * 100:.1f}%")
        
        # Save results
        final_results = {
            "model_name": args.model_name,
            "config": args.config,
            "test_accuracy": results['accuracy'],
            "baseline_accuracy": baseline_results['baseline_accuracy'],
            "error_rate": error_analysis['error_patterns']['error_rate']
        }
        
        with open("./outputs/final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for IMDB sentiment analysis")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--freeze_embeddings", action="store_true",
                        help="Freeze embedding layer")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of encoder layers to freeze")
    
    # Data arguments
    parser.add_argument("--train_size", type=int, default=4000,
                        help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=500,
                        help="Number of validation samples")
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of test samples")
    parser.add_argument("--prepare_data", action="store_true",
                        help="Force data preparation even if exists")
    
    # Training arguments
    parser.add_argument("--config", type=str, default="balanced",
                        choices=["conservative", "balanced", "aggressive"],
                        help="Training configuration")
    parser.add_argument("--hp_search", action="store_true",
                        help="Run hyperparameter search")
    parser.add_argument("--hp_trials", type=int, default=3,
                        help="Number of hyperparameter search trials")
    
    # Other arguments
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Run evaluation after training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)