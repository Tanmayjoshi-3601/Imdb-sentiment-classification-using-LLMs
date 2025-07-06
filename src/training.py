"""
Training module with hyperparameter optimization
"""

import os
import json
import time
import torch
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


class CustomLoggingCallback(TrainerCallback):
    """Custom callback for detailed logging"""
    
    def __init__(self):
        self.training_logs = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.training_logs.append(log_entry)
            
    def get_logs(self):
        return pd.DataFrame(self.training_logs)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Training configurations
TRAINING_CONFIGS = {
    "conservative": {
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 8,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "description": "Lower learning rate, smaller batches, more stable"
    },
    "balanced": {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 2,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "description": "Standard BERT fine-tuning parameters"
    },
    "aggressive": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 32,
        "num_train_epochs": 1,
        "warmup_ratio": 0.05,
        "weight_decay": 0.001,
        "gradient_accumulation_steps": 1,
        "description": "Higher learning rate, faster training"
    }
}


class ModelTrainer:
    """Handles model training with hyperparameter optimization"""
    
    def __init__(self, model, tokenizer, dataset_path="./data/tokenized_imdb"):
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = load_from_disk(dataset_path)
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    def get_training_args(self, output_dir, config):
        """Create training arguments from config"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get("num_epochs", config.get("num_train_epochs", 2)),
            per_device_train_batch_size=config.get("batch_size", config.get("per_device_train_batch_size", 16)),
            per_device_eval_batch_size=32,
            learning_rate=config.get("learning_rate", 2e-5),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=400,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            report_to="none",
            seed=42
        )
    
    def train_with_config(self, config, output_dir="./outputs"):
        """Train model with specific configuration"""
        training_args = self.get_training_args(output_dir, config)
        
        logging_callback = CustomLoggingCallback()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                logging_callback
            ],
        )
        
        # Train
        train_start = time.time()
        train_result = trainer.train()
        train_time = time.time() - train_start
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        return trainer, train_result, eval_result, logging_callback.get_logs(), train_time
    
    def hyperparameter_search(self, n_trials=3):
        """Perform hyperparameter optimization using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            config = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                "num_epochs": 1,  # Fixed for quick search
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
            }
            
            # Train with this configuration
            output_dir = f"./hp_search/trial_{trial.number}"
            os.makedirs(output_dir, exist_ok=True)
            
            trainer, _, eval_result, _, _ = self.train_with_config(config, output_dir)
            
            # Clean up
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return eval_result['eval_f1']
        
        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Save results
        best_params = study.best_params
        os.makedirs("./hp_search", exist_ok=True)
        with open("./hp_search/best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        return study, best_params


def train_model(model, tokenizer, dataset_path="./data/tokenized_imdb", config_name="balanced"):
    """Main training function"""
    trainer = ModelTrainer(model, tokenizer, dataset_path)
    config = TRAINING_CONFIGS[config_name]
    output_dir = f"./outputs/{config_name}"
    
    return trainer.train_with_config(config, output_dir)