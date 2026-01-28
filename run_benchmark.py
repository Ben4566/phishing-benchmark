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
from src.logger import setup_logger

# Logger initialisieren (Singleton-Pattern via logging modul)
# Das schreibt nun in 'logs/benchmark_run_DATUM.log' UND in die Konsole
logger = setup_logger("BenchmarkRunner")

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

# --- CENTRALIZED CONFIGURATION ---
MODEL_CONFIG = {
    "cnn": {
        "epochs": 1,        
        "lr": 0.001,
        "batch_size": 64
    },
    "svm": {
        "epochs": 1,        
        "lr": 0.0,          
        "batch_size": 0     
    },
    "xgb": {
        "epochs": 100,      
        "lr": 0.1,
        "batch_size": 0     
    },
    "lr": {
        "epochs": 50,
        "lr": 0.01,
        "batch_size": 4096  
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
    seeds = [42] #, 1337, 2024
    ratios = [1000]

    logger.info("--- STARTING ORCHESTRATION: Benchmark Suite ---")
    
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            logger.warning(f"Dataset artifact missing: {dataset}. Skipping.")
            continue
            
        dataset_name = os.path.basename(dataset)
        logger.info(f">>> Context: {dataset_name} <<<")

        # --- MODALITY GUARDRAILS ---
        is_hybrid  = "PhiUSIIL" in dataset_name      
        is_numeric = "feature_all" in dataset_name   
        is_text    = "combined_urls" in dataset_name 

        for model in models:
            # Check constraints
            if is_text and model in ["xgb", "lr"]: continue 
            if is_numeric and model in ["cnn", "svm"]: continue 
            
            # Get Config
            config = MODEL_CONFIG.get(model, {"epochs": 5, "lr": 0.001, "batch_size": 64})

            for ratio in ratios:
                for seed in seeds:
                    # Logging statt Print für saubere Protokollierung
                    logger.info(f"   -> Executing: {model.upper()} | Prior: {ratio}:1 | Seed: {seed}")
                    logger.debug(f"      [Config] Epochs: {config['epochs']} | LR: {config['lr']}")

                    args = ExperimentArgs(
                        model=model,
                        file=dataset,
                        epochs=config["epochs"],
                        lr=config["lr"],
                        batch_size=config["batch_size"],
                        imbalance_ratio=ratio,
                        seed=seed
                    )
                    
                    history_before = load_benchmark_history()
                    
                    try:
                        # --- Execution Delegate ---
                        if model == "cnn":
                            train.run_cnn(dataset, args)
                        elif model == "svm":
                            train.run_svm_tfidf(dataset, args)
                        elif model in ["lr", "xgb"]:
                            train.run_numeric_model(model, dataset, args)
                        
                        # --- Result Capture ---
                        history_after = load_benchmark_history()
                        new_items_count = len(history_after) - len(history_before)
                        
                        if new_items_count > 0:
                            new_entries = history_after[-new_items_count:]
                            for entry in new_entries:
                                entry["dataset_name"] = dataset_name
                                entry["imbalance_ratio"] = ratio
                                entry["seed"] = seed
                                results.append(entry)
                            
                            logger.info(f"[SUCCESS] Captured {new_items_count} new result(s) for {model}.")
                        else:
                            # Warnung, wenn zwar Code lief, aber keine JSON-Daten geschrieben wurden
                            logger.warning(f"Pipeline finished for {model} but no results were persisted.")
                            
                    except Exception as e:
                        # Senior-Level: exc_info=True speichert den vollen Stacktrace im Logfile
                        logger.error(f"      [CRITICAL] Pipeline failed for {model}: {e}", exc_info=True)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"--- Benchmark Complete. Aggregated results saved to {OUTPUT_CSV} ---")
        return df
    else:
        logger.warning("No results generated during this run.")
        return pd.DataFrame()
  
if __name__ == "__main__":
    df_results = run_full_benchmark()
    
    if not df_results.empty:
        logger.info("Generating visualizations...")
        try:
            viz.generate_plots(df_results, OUTPUT_DIR)
            logger.info(f"Plots saved to {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            
    elif os.path.exists(OUTPUT_CSV):
         logger.info("Loading existing results for visualization...")
         try:
             df_existing = pd.read_csv(OUTPUT_CSV)
             viz.generate_plots(df_existing, OUTPUT_DIR)
             logger.info(f"Plots saved to {OUTPUT_DIR}")
         except Exception as e:
             logger.error(f"Visualization of existing data failed: {e}", exc_info=True)