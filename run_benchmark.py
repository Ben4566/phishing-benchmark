import sys
import os
import pandas as pd
import json
import numpy as np
import torch
from typing import List, Dict, Any

# --- IMPORTS ---
import src.train as train 
import src.visualize as viz

# --- CONFIGURATION & CONSTANTS ---
DATASETS = [
    "data/raw/combined_urls.csv",                 
    "data/processed/feature_all.csv",             
    "data/processed/PhiUSIIL_Phishing_URL_Dataset.csv" 
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_benchmark_results.csv")
BENCHMARK_JSON = "benchmark_results.json"

# --- SENIOR-LEVEL UPDATE: CENTRALIZED HYPERPARAMETER CONFIGURATION ---
# Senior-Level Comment:
# We externalize hyperparameters into a structured dictionary acting as a 'Single Source of Truth'.
# This eliminates 'magic numbers' buried within execution loops and allows for 
# rapid experimentation (e.g., changing CNN epochs) without risking logic regression in the pipeline.
MODEL_CONFIG = {
    "cnn": {
        "epochs": 1,       # <--- HIER EINFACH ÄNDERN (z.B. von 5 auf 20)
        "lr": 0.001,
        "batch_size": 64
    },
    "svm": {
        "epochs": 1,        # SVM in sklearn doesn't iterate epochs the same way, but keeping structure consistent
        "lr": 0.0,          # Not used for SVM (handled by sklearn internally)
        "batch_size": 0     # Not used
    },
    "xgb": {
        "epochs": 100,      # Represents n_estimators
        "lr": 0.1,
        "batch_size": 0     # Not used
    },
    "lr": {
        "epochs": 50,
        "lr": 0.01,
        "batch_size": 4096  # Larger batch size for GPU Logistic Regression
    }
}

class ExperimentArgs:
    def __init__(self, model, file, epochs=5, batch_size=64, lr=0.001, imbalance_ratio=0, seed=42):
        self.model = model
        self.file = file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.imbalance_ratio = imbalance_ratio
        self.seed = seed 

def load_benchmark_history(json_path: str = BENCHMARK_JSON) -> List[Dict]:
    if not os.path.exists(json_path): return []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception: return []

def run_full_benchmark():
    """
    Orchestrates the complete ablation study with centralized configuration.
    """
    results = []
    
    # Define the search space
    models = ["cnn", "svm", "xgb", "lr"] 
    seeds = [42, 1337, 2024] #, 1337, 2024]  # Reduced for quicker testing
    ratios = [1000]

    print(f"--- STARTING ORCHESTRATION: Benchmark Suite ---")
    
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"[WARNING] Dataset artifact missing: {dataset}. Skipping.")
            continue
            
        dataset_name = os.path.basename(dataset)
        print(f"\n>>> Context: {dataset_name} <<<")

        # --- MODALITY GUARDRAILS ---
        is_hybrid  = "PhiUSIIL" in dataset_name      
        is_numeric = "feature_all" in dataset_name   
        is_text    = "combined_urls" in dataset_name 

        for model in models:
            # Check constraints (Text vs Numeric vs Hybrid)
            if is_text and model in ["xgb", "lr"]: continue 
            if is_numeric and model in ["cnn", "svm"]: continue 
            
            # Senior-Level Comment:
            # Retrieve model-specific configuration. 
            # This implements the Strategy Pattern implicitly: the execution logic adapts 
            # based on the config state rather than hardcoded if-else blocks.
            config = MODEL_CONFIG.get(model, {"epochs": 5, "lr": 0.001, "batch_size": 64})

            for ratio in ratios:
                for seed in seeds:
                    print(f"   -> Executing: {model.upper()} | Prior: {ratio}:1 | Seed: {seed}")
                    print(f"      [Config] Epochs: {config['epochs']} | LR: {config['lr']}")

                    args = ExperimentArgs(
                        model=model,
                        file=dataset,
                        epochs=config["epochs"],        # Injected from Config
                        lr=config["lr"],                # Injected from Config
                        batch_size=config["batch_size"],# Injected from Config
                        imbalance_ratio=ratio,
                        seed=seed
                    )
                    
                    history_before = load_benchmark_history()
                    
                    try:
                        # ... (Modell-Training Aufrufe bleiben gleich) ...
                        if model == "cnn":
                            train.run_cnn(dataset, args)
                        elif model == "svm":
                            train.run_svm_tfidf(dataset, args)
                        elif model in ["lr", "xgb"]:
                            train.run_numeric_model(model, dataset, args)
                        
                        # --- SENIOR-LEVEL FIX: CAPTURE ALL NEW ENTRIES ---
                        # Wir berechnen das Delta, um sicherzustellen, dass sowohl 
                        # 'Training' als auch 'Inference' Einträge erfasst werden.
                        history_after = load_benchmark_history()
                        new_items_count = len(history_after) - len(history_before)
                        
                        if new_items_count > 0:
                            # Hole alle neuen Einträge (Slicing von hinten)
                            new_entries = history_after[-new_items_count:]
                            
                            for entry in new_entries:
                                entry["dataset_name"] = dataset_name
                                entry["imbalance_ratio"] = ratio
                                entry["seed"] = seed
                                results.append(entry)
                            
                            print(f"      [SUCCESS] Captured {new_items_count} new result(s) for {model}.")
                        else:
                            print(f"      [WARNING] Pipeline finished but no results were persisted for {model}.")
                            
                    except Exception as e:
                        print(f"      [ERROR] Pipeline failed: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n--- Benchmark Complete. Aggregated results saved to {OUTPUT_CSV} ---")
        return df
    else:
        print("No results generated.")
        return pd.DataFrame()
  
if __name__ == "__main__":
    df_results = run_full_benchmark()
    
    if not df_results.empty:
        print("Generating visualizations...")
        viz.generate_plots(df_results, OUTPUT_DIR)
    elif os.path.exists(OUTPUT_CSV):
         print("Loading existing results for visualization...")
         df_existing = pd.read_csv(OUTPUT_CSV)
         viz.generate_plots(df_existing, OUTPUT_DIR)