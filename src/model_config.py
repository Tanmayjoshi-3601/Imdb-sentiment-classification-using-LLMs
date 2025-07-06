"""
Model configuration and selection module
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)
import pandas as pd
import time


class ModelConfigurator:
    """Handles model selection and configuration"""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        
    def get_model_info(self):
        """Get information about the selected model"""
        config = AutoConfig.from_pretrained(self.model_name)
        
        info = {
            "model_name": self.model_name,
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "num_parameters": None  # Will be filled after model instantiation
        }
        
        return info
    
    def setup_model(self, freeze_embeddings=False, freeze_layers=0):
        """Initialize and configure the model for fine-tuning"""
        print(f"Loading model: {self.model_name}")
        
        # Load model with classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Optionally freeze embeddings
        if freeze_embeddings:
            if hasattr(model, 'bert'):
                for param in model.bert.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'distilbert'):
                for param in model.distilbert.embeddings.parameters():
                    param.requires_grad = False
            print("Embeddings frozen")
        
        # Optionally freeze lower layers
        if freeze_layers > 0:
            if hasattr(model, 'bert'):
                for i in range(freeze_layers):
                    for param in model.bert.encoder.layer[i].parameters():
                        param.requires_grad = False
            print(f"Frozen first {freeze_layers} encoder layers")
        
        # Recount trainable parameters after freezing
        trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_after:,}")
        print(f"Frozen parameters: {total_params - trainable_params_after:,}")
        
        return model, {
            "total_params": total_params,
            "trainable_params": trainable_params_after,
            "frozen_params": total_params - trainable_params_after
        }
    
    def get_tokenizer(self):
        """Get the tokenizer for the model"""
        return AutoTokenizer.from_pretrained(self.model_name)


def compare_models():
    """Compare different model architectures"""
    models_to_compare = [
        {"name": "bert-base-uncased", "type": "BERT"},
        {"name": "distilbert-base-uncased", "type": "DistilBERT"},
        {"name": "albert-base-v2", "type": "ALBERT"}
    ]
    
    comparison = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_info in models_to_compare:
        configurator = ModelConfigurator(model_info["name"])
        info = configurator.get_model_info()
        model, param_info = configurator.setup_model()
        
        # Measure inference speed
        model.eval()
        model.to(device)
        tokenizer = configurator.get_tokenizer()
        
        # Create dummy input
        dummy_input = tokenizer(
            "This is a test sentence for speed measurement.",
            return_tensors="pt",
            padding="max_length",
            max_length=256
        ).to(device)
        
        # Remove token_type_ids for DistilBERT
        if 'distilbert' in model_info["name"].lower():
            dummy_input.pop('token_type_ids', None)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**dummy_input)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(**dummy_input)
        inference_time = (time.time() - start_time) / 100
        
        info.update(param_info)
        info["Model"] = model_info["type"]
        info["Inference Time (ms)"] = inference_time * 1000
        comparison.append(info)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return pd.DataFrame(comparison)