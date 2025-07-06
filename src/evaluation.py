"""
Model evaluation and error analysis module
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os


class ModelEvaluator:
    """Comprehensive model evaluation and error analysis"""
    
    def __init__(self, model_path, dataset_path="./data/tokenized_imdb"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.datasets = load_from_disk(dataset_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load dataset info
        with open("./data/dataset_info.json", "r") as f:
            self.dataset_info = json.load(f)
    
    def evaluate_on_test_set(self, trainer=None):
        """Comprehensive evaluation on test set"""
        if trainer:
            # Use trainer for predictions
            test_predictions = trainer.predict(self.datasets["test"])
            y_pred = np.argmax(test_predictions.predictions, axis=1)
            y_true = test_predictions.label_ids
            y_probs = torch.softmax(torch.from_numpy(test_predictions.predictions), dim=1).numpy()
        else:
            # Manual evaluation
            test_dataset = self.datasets["test"]
            dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            all_predictions = []
            all_labels = []
            all_probs = []
            
            print("Evaluating on test set...")
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            y_pred = np.array(all_predictions)
            y_true = np.array(all_labels)
            y_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Create detailed report
        report = classification_report(
            y_true, y_pred,
            target_names=self.dataset_info["label_names"],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            "accuracy": accuracy,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred,
            "labels": y_true,
            "probabilities": y_probs
        }
        
        return results
    
    def compare_with_baseline(self, baseline_model_name="bert-base-uncased"):
        """Compare fine-tuned model with pre-trained baseline"""
        print("Loading baseline model...")
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            baseline_model_name,
            num_labels=2
        )
        baseline_model.to(self.device)
        baseline_model.eval()
        
        # Evaluate baseline
        test_dataset = self.datasets["test"]
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        baseline_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating baseline"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                
                outputs = baseline_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                baseline_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        baseline_accuracy = accuracy_score(all_labels, baseline_predictions)
        baseline_f1 = precision_recall_fscore_support(
            all_labels, baseline_predictions, average='weighted'
        )[2]
        
        return {
            "baseline_accuracy": baseline_accuracy,
            "baseline_f1": baseline_f1,
            "baseline_predictions": baseline_predictions
        }
    
    def error_analysis(self, results):
        """Detailed error analysis"""
        predictions = np.array(results["predictions"])
        labels = np.array(results["labels"])
        probs = np.array(results["probabilities"])
        
        # Find misclassified examples
        misclassified_mask = predictions != labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Analyze confidence of errors
        confidences = probs.max(axis=1)
        error_confidences = confidences[misclassified_mask]
        correct_confidences = confidences[~misclassified_mask]
        
        # Categorize errors
        error_types = {
            "false_positives": [],
            "false_negatives": [],
            "high_conf_errors": [],
            "low_conf_errors": []
        }
        
        test_dataset = self.datasets["test"]
        
        for idx in misclassified_indices[:100]:  # Analyze first 100 errors
            example = test_dataset[int(idx)]
            confidence = confidences[idx]
            
            # Decode text
            tokens = example['input_ids']
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            error_info = {
                "index": int(idx),
                "true_label": self.dataset_info["label_names"][labels[idx]],
                "predicted_label": self.dataset_info["label_names"][predictions[idx]],
                "confidence": float(confidence),
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            }
            
            if labels[idx] == 0 and predictions[idx] == 1:
                error_types["false_positives"].append(error_info)
            elif labels[idx] == 1 and predictions[idx] == 0:
                error_types["false_negatives"].append(error_info)
            
            if confidence > 0.9:
                error_types["high_conf_errors"].append(error_info)
            elif confidence < 0.6:
                error_types["low_conf_errors"].append(error_info)
        
        # Calculate error patterns
        error_patterns = {
            "total_errors": len(misclassified_indices),
            "error_rate": len(misclassified_indices) / len(predictions),
            "avg_error_confidence": float(error_confidences.mean()),
            "avg_correct_confidence": float(correct_confidences.mean()),
            "high_confidence_error_rate": sum(1 for c in error_confidences if c > 0.9) / len(error_confidences) if len(error_confidences) > 0 else 0
        }
        
        return {
            "error_types": error_types,
            "error_patterns": error_patterns,
            "error_confidences": error_confidences,
            "correct_confidences": correct_confidences,
            "misclassified_indices": misclassified_indices
        }
    
    def visualize_results(self, results, error_analysis, save_dir="./outputs"):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(results["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.dataset_info["label_names"],
                    yticklabels=self.dataset_info["label_names"])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 2. Confidence Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(error_analysis["error_confidences"], bins=20, alpha=0.7, color='red', label='Errors', edgecolor='black')
        ax1.hist(error_analysis["correct_confidences"], bins=20, alpha=0.7, color='green', label='Correct', edgecolor='black')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        
        # 3. Per-class metrics
        classes = self.dataset_info["label_names"]
        metrics = ['precision', 'recall', 'f1']
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = results[metric]
            ax2.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Performance')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(classes)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_metrics.png'))
        plt.close()


def evaluate_model(model_path, trainer=None):
    """Main evaluation function"""
    evaluator = ModelEvaluator(model_path)
    
    # Test set evaluation
    results = evaluator.evaluate_on_test_set(trainer)
    
    # Baseline comparison
    baseline_results = evaluator.compare_with_baseline()
    
    # Error analysis
    error_analysis = evaluator.error_analysis(results)
    
    # Visualizations
    evaluator.visualize_results(results, error_analysis)
    
    return results, baseline_results, error_analysis