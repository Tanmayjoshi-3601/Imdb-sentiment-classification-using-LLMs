"""
Data preparation module for IMDB sentiment analysis
"""

import os
import json
import random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


class IMDBDataProcessor:
    """Handles IMDB dataset loading, preprocessing, and formatting"""
    
    def __init__(self, model_name="bert-base-uncased", max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_and_prepare_dataset(self, train_size=4000, val_size=500, test_size=1000):
        """Load IMDB dataset and prepare train/val/test splits"""
        print("Loading IMDB dataset...")
        
        # Load the official IMDB dataset from Hugging Face
        full_dataset = load_dataset("imdb")
        
        # Create balanced subsets
        train_val_data = self._create_balanced_subset(
            full_dataset['train'], 
            train_size + val_size
        )
        
        # Split into train and validation
        train_val_split = train_val_data.train_test_split(
            test_size=val_size / (train_size + val_size),
            seed=42
        )
        
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']
        test_dataset = self._create_balanced_subset(full_dataset['test'], test_size)
        
        # Print dataset statistics
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        return dataset_dict
    
    def _create_balanced_subset(self, dataset, size):
        """Create a balanced subset with equal positive/negative samples"""
        half_size = size // 2
        
        # Get indices for each label
        pos_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
        neg_indices = [i for i, label in enumerate(dataset['label']) if label == 0]
        
        # Sample balanced indices
        selected_pos = random.sample(pos_indices, min(half_size, len(pos_indices)))
        selected_neg = random.sample(neg_indices, min(half_size, len(neg_indices)))
        
        # Combine and shuffle
        selected_indices = selected_pos + selected_neg
        random.shuffle(selected_indices)
        
        return dataset.select(selected_indices)
    
    def preprocess_function(self, examples):
        """Tokenize and format the dataset"""
        # Clean text - remove HTML tags and excessive whitespace
        cleaned_texts = []
        for text in examples["text"]:
            # Remove basic HTML tags
            text = text.replace("<br />", " ").replace("<br/>", " ")
            text = text.replace("<BR />", " ").replace("<BR/>", " ")
            text = " ".join(text.split())  # Normalize whitespace
            
            # Ensure text is not empty
            if not text.strip():
                text = "empty review"
                
            cleaned_texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            cleaned_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove token_type_ids if using DistilBERT
        if 'distilbert' in self.model_name.lower():
            tokenized.pop('token_type_ids', None)
        
        # Add labels
        tokenized["labels"] = examples["label"]
        
        return tokenized
    
    def prepare_datasets(self, dataset_dict):
        """Apply preprocessing to all splits"""
        print("Tokenizing datasets...")
        
        tokenized_datasets = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing"
        )
        
        # Set format for PyTorch
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets
    
    def save_datasets(self, tokenized_datasets, output_dir="./data"):
        """Save processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        tokenized_datasets.save_to_disk(os.path.join(output_dir, "tokenized_imdb"))
        
        # Save dataset info
        info = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "train_size": len(tokenized_datasets["train"]),
            "val_size": len(tokenized_datasets["validation"]),
            "test_size": len(tokenized_datasets["test"]),
            "num_labels": 2,
            "label_names": ["negative", "positive"]
        }
        
        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        
        return info


def prepare_data(train_size=4000, val_size=500, test_size=1000):
    """Main function to prepare data"""
    processor = IMDBDataProcessor()
    dataset_dict = processor.load_and_prepare_dataset(train_size, val_size, test_size)
    tokenized_datasets = processor.prepare_datasets(dataset_dict)
    info = processor.save_datasets(tokenized_datasets)
    return tokenized_datasets, info