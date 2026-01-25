import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import torch

# HINWEIS: sys.path Hack entfernt. Das Projekt sollte über 'pip install -e .' installiert sein
# oder von Root ausgeführt werden. [cite: 30]

import src.train as train 

DATASETS = [
    "data/raw/combined_urls.csv",
    "data/processed/PhiUSIIL_Phishing_URL_Dataset.csv",
    "data/processed/feature_all.csv"
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_benchmark_results.csv")

class ExperimentArgs:
    def __init__(self, model, file, epochs=5, batch_size=64, lr=0.001, imbalance_ratio=0, seed=42):
        self.model = model
        self.file = file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.imbalance_ratio = imbalance_ratio
        self.seed = seed 

def get_latest_benchmark_entry(json_path="benchmark_results.json"):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data[-1] if data else None
    except: return None

def run_full_benchmark():
    results = []
    models = ["cnn", "xgb", "lr", "svm"] 
    seeds = [42, 1337, 2024]        
    ratios = [0, 1000]              

    print(f"--- STARTING AUTOMATION FROM ROOT ---")
    
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"WARNING: File {dataset} not found. Skipping.")
            continue
            
        dataset_name = os.path.basename(dataset)
        print(f"\n>>> Processing Dataset: {dataset_name} <<<")

        # Warm-up (optional)
        try:
            print("   -> Warm-up run...")
            train.run_cnn(dataset, ExperimentArgs("cnn", dataset, epochs=1, seed=42))
        except Exception:
            pass 

        for model in models:
            if model == "svm" and dataset_name != "combined_urls.csv":
                continue

            for ratio in ratios:
                for i, seed in enumerate(seeds):
                    print(f"   -> {model.upper()} | Ratio {ratio}:1 | Seed {seed}")
                    
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
                    
                    # Globale Seeds in train.py setzen (Sauberkeit)
                    try:
                        # Aufruf der entsprechenden Trainingsfunktion
                        if model == "cnn":
                            train.run_cnn(dataset, args)
                        elif model == "svm":
                            train.run_svm_tfidf(dataset, args)
                        elif model in ["lr", "xgb"]:
                            train.run_numeric_model(model, dataset, args)
                        
                        entry = get_latest_benchmark_entry()
                        if entry:
                            entry["dataset_name"] = dataset_name
                            entry["imbalance_ratio"] = ratio
                            entry["seed"] = seed
                            results.append(entry)
                            
                    except Exception as e:
                        print(f"      [SKIP] Error: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n--- Done. Results saved to {OUTPUT_CSV} ---")
    return df

def generate_plots(df):
    if df.empty:
        print("No results to plot.")
        return

    sns.set_theme(style="whitegrid")
    unique_datasets = df["dataset_name"].unique()
    
    for ds in unique_datasets:
        ds_data = df[df["dataset_name"] == ds]
        filename_suffix = ds.replace(".", "_")

        subset_1000 = ds_data[ds_data["imbalance_ratio"] == 1000]
        if not subset_1000.empty:
            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=subset_1000, x="model", y="f1_score", 
                hue="model", errorbar="sd", palette="viridis", capsize=.1
            )
            plt.title(f"F1-Score at 1000:1 Imbalance\n({ds})")
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"plot_f1_1000_{filename_suffix}.png"))
            plt.close()

        plt.figure(figsize=(8, 5))
        sns.pointplot(
            data=ds_data, x="imbalance_ratio", y="f1_score", 
            hue="model", dodge=True, errorbar=None
        )
        plt.title(f"Imbalance Impact\n({ds})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"plot_robustness_{filename_suffix}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="time_sec", y="vram_system_peak_mb", 
        hue="model", style="dataset_name", s=100, alpha=0.7
    )
    plt.title("Efficiency: Time vs. VRAM")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_efficiency_global.png"))
    plt.close()

if __name__ == "__main__":
    df_results = run_full_benchmark()
    generate_plots(df_results)