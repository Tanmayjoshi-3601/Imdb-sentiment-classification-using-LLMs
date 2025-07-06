"""
Optimized inference pipeline for production use
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm


class SentimentAnalyzer:
    """Optimized inference pipeline for sentiment analysis"""
    
    def __init__(self, model_path, device=None, max_length=256):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Configuration
        self.max_length = max_length
        self.id2label = {0: "negative", 1: "positive"}
        self.label2id = {"negative": 0, "positive": 1}
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with a dummy input"""
        dummy_input = "This is a warmup sentence."
        _ = self.predict(dummy_input)
    
    def predict(self, text: Union[str, List[str]], return_all_scores=False):
        """
        Predict sentiment for single text or batch of texts
        
        Args:
            text: Single string or list of strings
            return_all_scores: Whether to return probabilities for all classes
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Handle single string input
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        start_time = time.time()
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        tokenization_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        inference_time = time.time() - start_time
        
        # Process results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "sentiment": self.id2label[pred.item()],
                "confidence": probs[pred].item(),
                "prediction_id": pred.item()
            }
            
            if return_all_scores:
                result["scores"] = {
                    label: probs[idx].item() 
                    for label, idx in self.label2id.items()
                }
            
            results.append(result)
        
        # Add timing information
        timing_info = {
            "tokenization_time_ms": tokenization_time * 1000,
            "inference_time_ms": inference_time * 1000,
            "total_time_ms": (tokenization_time + inference_time) * 1000,
            "texts_per_second": len(texts) / (tokenization_time + inference_time)
        }
        
        if single_input:
            return {**results[0], "timing": timing_info}
        else:
            return {"predictions": results, "timing": timing_info}
    
    def batch_predict_efficient(self, texts: List[str], batch_size=32, show_progress=True):
        """
        Efficient batch prediction for large datasets
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        
        Returns:
            List of predictions with timing information
        """
        all_results = []
        total_time = 0
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            results = self.predict(batch, return_all_scores=False)
            all_results.extend(results["predictions"])
            total_time += results["timing"]["total_time_ms"]
        
        return {
            "predictions": all_results,
            "total_time_ms": total_time,
            "average_time_per_text_ms": total_time / len(texts),
            "texts_per_second": len(texts) / (total_time / 1000)
        }
    
    def interactive_demo(self):
        """Interactive command-line demo"""
        print("\n=== Sentiment Analysis Demo ===")
        print("Enter text to analyze (or 'quit' to exit):\n")
        
        while True:
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("Please enter some text to analyze.")
                continue
            
            result = self.predict(text, return_all_scores=True)
            
            print(f"\nSentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if 'scores' in result:
                print("\nDetailed Scores:")
                for label, score in result['scores'].items():
                    print(f"  {label}: {score:.2%}")
            
            print(f"\nProcessing time: {result['timing']['total_time_ms']:.2f}ms")
            print("-" * 50)


def create_inference_pipeline(model_path):
    """Create and return an inference pipeline"""
    return SentimentAnalyzer(model_path)