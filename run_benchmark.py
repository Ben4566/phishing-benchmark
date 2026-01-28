import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import torch
from typing import List, Dict, Any

# Import local training modules
import src.train as train 

# --- CONFIGURATION & CONSTANTS ---
# Define dataset paths relative to project root.
# We explicitly categorize datasets by their modality (Text vs. Tabular Features).
DATASETS = [
    "data/raw/combined_urls.csv",                 # Modality: Raw Text (URLs) -> Suitable for CNN, SVM
    "data/processed/feature_all.csv",             # Modality: Hand-crafted Features -> Suitable for XGB, LR
    "data/processed/PhiUSIIL_Phishing_URL_Dataset.csv" # Hybrid/Comprehensive -> Supports all architectures
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_benchmark_results.csv")
BENCHMARK_JSON = "benchmark_results.json"

class ExperimentArgs:
    """
    Mock object to mimic argparse namespace, allowing programmatic execution 
    of training scripts without command-line calls.
    """
    def __init__(self, model, file, epochs=5, batch_size=64, lr=0.001, imbalance_ratio=0, seed=42):
        self.model = model
        self.file = file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.imbalance_ratio = imbalance_ratio
        self.seed = seed 

def load_benchmark_history(json_path: str = BENCHMARK_JSON) -> List[Dict]:
    """
    Loads existing benchmark telemetry to append new runs.
    Ensures idempotency and data persistence.
    """
    if not os.path.exists(json_path): return []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception: return []

def run_full_benchmark():
    """
    Orchestrates the complete ablation study.
    Iterates over Models x Datasets x Imbalance Ratios x Random Seeds.
    
    Implements 'Modality Guardrails' to prevent incompatible model-data pairs
    (e.g., trying to run a CNN on pre-extracted numerical features).
    """
    results = []
    
    # Define the search space for the experiment
    models = ["cnn", "svm", "xgb", "lr"] 
    seeds = [42, 1337, 2024]        # Multiple seeds to quantify stability (Standard Deviation)
    ratios = [0, 1000]              # 0 = Balanced, 1000 = Realistic Phishing Prior (1:1000)

    print(f"--- STARTING ORCHESTRATION: Benchmark Suite ---")
    
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"[WARNING] Dataset artifact missing: {dataset}. Skipping.")
            continue
            
        dataset_name = os.path.basename(dataset)
        print(f"\n>>> Context: {dataset_name} <<<")

        # --- MODALITY GUARDRAILS ---
        # Determine data capabilities based on nomenclature
        is_hybrid  = "PhiUSIIL" in dataset_name      # Contains both URL text and features
        is_numeric = "feature_all" in dataset_name   # Contains only numerical features
        is_text    = "combined_urls" in dataset_name # Contains only raw URL strings

        for model in models:
            # Constraint 1: Text-Only Datasets -> NLP Models only
            if is_text and model in ["xgb", "lr"]:
                continue # Skip numeric models on text data
            
            # Constraint 2: Feature-Only Datasets -> Numeric Models only
            if is_numeric and model in ["cnn", "svm"]:
                continue # Skip NLP models on numeric data
            
            # Constraint 3: Hybrid Datasets -> All models allowed

            for ratio in ratios:
                for seed in seeds:
                    print(f"   -> Executing: {model.upper()} | Prior: {ratio}:1 | Seed: {seed}")
                    
                    # Dynamic Hyperparameter Adjustment based on Architecture
                    curr_epochs = 100 if model == "xgb" else 5
                    curr_lr = 0.1 if model == "xgb" else 0.001
                    
                    args = ExperimentArgs(
                        model=model,
                        file=dataset,
                        epochs=curr_epochs,
                        lr=curr_lr,
                        imbalance_ratio=ratio,
                        seed=seed
                    )
                    
                    history_before = load_benchmark_history()
                    
                    try:
                        # Dispatcher Logic: Route to correct pipeline
                        if model == "cnn":
                            train.run_cnn(dataset, args)
                        elif model == "svm":
                            # Enforce TF-IDF vectorization for SVM
                            train.run_svm_tfidf(dataset, args)
                        elif model in ["lr", "xgb"]:
                            train.run_numeric_model(model, dataset, args)
                        
                        # Verify Result Generation
                        history_after = load_benchmark_history()
                        if len(history_after) > len(history_before):
                            entry = history_after[-1]
                            # Enrich metadata for analysis
                            entry["dataset_name"] = dataset_name
                            entry["imbalance_ratio"] = ratio
                            entry["seed"] = seed
                            results.append(entry)
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

def generate_plots(df):
    """
    Generates publication-ready visualizations with statistical error bars.
    Focuses on Trade-offs: Performance vs. Resource Utilization.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Metrics mapping for the 4x3 grid
    metrics_config = [
        ("time_sec", "Training Time (s) - Lower is better"),
        ("time_sec", "Inference Time (s) - Lower is better"), # Placeholder logic if inference time logged separately
        ("gpu_util_percent_avg", "GPU Utilization (%)"),
        ("cpu_percent_avg", "CPU Utilization (%)"),
        ("vram_system_peak_mb", "VRAM Usage (MB)"),
        ("ram_mb", "RAM Usage (MB)"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
        ("fpr", "False Positive Rate"),
        ("auc", "AUC")
    ]

    plt.figure(figsize=(20, 24))
    plt.suptitle("AI Model Benchmark: Deep Learning vs. Classical ML", fontsize=20, weight='bold', y=1.02)

    # Filter for Inference task for most metrics (except training time)
    # Note: Logic assumes 'task' column exists. If strictly splitting plots, logic adapts.
    
    unique_models = df['model'].unique()
    palette = sns.color_palette("coolwarm", len(unique_models))

    for i, (metric, title) in enumerate(metrics_config):
        plt.subplot(4, 3, i + 1)
        
        # Check if metric exists
        if metric not in df.columns:
            continue
            
        # Determine task context (Training metrics vs Inference metrics)
        plot_data = df
        if "time" in metric and "Inference" in title:
            plot_data = df[df['task'] == 'Inference']
        elif "time" in metric and "Training" in title:
            plot_data = df[df['task'] == 'Training']
        elif metric in ["accuracy", "precision", "recall", "f1_score", "fpr", "auc"]:
            plot_data = df[df['task'] == 'Inference'] # Quality metrics usually come from Test set (Inference)

        if plot_data.empty: continue

        # Barplot with Error Bars (Standard Deviation across Seeds)
        sns.barplot(
            data=plot_data, 
            x="model", 
            y=metric, 
            hue="model", 
            palette=palette, 
            errorbar="sd", # Draws the black line for standard deviation
            capsize=0.1,
            dodge=False
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate bars with mean values
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "benchmark_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    # 1. Execute Benchmark
    df_results = run_full_benchmark()
    
    # 2. Visualize Results
    if not df_results.empty:
        # Optional: Load from CSV if you want to replot without running
        # df_results = pd.read_csv(OUTPUT_CSV) 
        generate_plots(df_results)