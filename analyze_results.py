import pandas as pd
import json
import os
import sys

# Import domain-specific visualization and logging modules
from src.visualize import BenchmarkVisualizer
from src.logger import setup_logger

logger = setup_logger("Analyzer")

INPUT_JSON = "benchmark_results.json"
OUTPUT_DIR = "results"

# --- Configuration & Constants ---
# Maps internal model identifiers to presentation-ready display names for reporting.
MODEL_DISPLAY_NAMES = {
    'cnn': 'CNN (PyTorch)',
    'xgb': 'XGBoost',
    'lr': 'Logistic Regression',
    'svm': 'Linear SVC',
    'rf': 'Random Forest' 
}

def load_and_prep_data(filepath):
    """
    Ingests raw benchmark results from JSON and performs data cleaning/enrichment 
    to prepare for visualization.

    Args:
        filepath (str): Path to the source JSON file.

    Returns:
        pd.DataFrame: A processed DataFrame ready for plotting, or an empty DF on failure.
    """
    if not os.path.exists(filepath):
        logger.error(f"Input artifact not found: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error("JSON file is corrupted or empty.")
            return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty: return df

    # 1. Standardize model identifiers for consistent reporting
    df['model'] = df['model'].map(lambda x: MODEL_DISPLAY_NAMES.get(x.lower(), x))

    # 2. Differentiate model variants based on the underlying dataset
    # If the dataset is 'PhiUSIIL', we append a suffix to distinguish it 
    # from the baseline runs in combined visualizations.
    def refine_name(row):
        ds = row.get('dataset', '')
        if 'PhiUSIIL' in str(ds):
            return f"{row['model']} (PhiUSIIL)"
        return row['model']
    
    df['model'] = df.apply(refine_name, axis=1)

    return df

def main():
    logger.info("Starting analysis pipeline...")
    
    # 1. Data Ingestion
    df = load_and_prep_data(INPUT_JSON)
    if df.empty:
        logger.warning("No data found in benchmark_results.json. Did you execute 'run_benchmark.py -m'?")
        sys.exit(0)

    # 2. Data Persistence (CSV Export)
    # Dump the raw summary for external auditing or further analysis.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary exported to: {csv_path}")

    # 3. Visualization Generation
    viz = BenchmarkVisualizer(OUTPUT_DIR)
    
    # --- Visualization Strategy: Segmented Reporting ---

    # A) Baseline Performance Overview (Standard Dataset only)
    # Filter out PhiUSIIL variants to provide a clean comparison of base architectures.
    logger.info("Generating plot: Baseline Model Comparison (Standard Data)...")
    df_base = df[~df['model'].str.contains("PhiUSIIL")].copy()
    
    if not df_base.empty:
        viz.plot_overview(df_base, "Base Model Comparison (Standard Data)", "benchmark_base_models.png")
    else:
        logger.warning("No standard dataset results found (only PhiUSIIL runs detected?).")

    # B) Dataset Impact Analysis (A/B Testing View)
    # Generate pairwise comparisons for each architecture to isolate the impact of the dataset.
    base_models = ['CNN (PyTorch)', 'XGBoost', 'Logistic Regression', 'Linear SVC']
    
    for model_name in base_models:
        # Filter logic: Select rows where the model name matches the base string.
        # This captures both "XGBoost" (Base) and "XGBoost (PhiUSIIL)".
        df_subset = df[df['model'].str.contains(model_name, regex=False)].copy()
        
        if not df_subset.empty:
            # Sanitize filename for compatibility
            safe_filename = model_name.replace(" ", "").replace("(", "").replace(")", "").lower()
            output_filename = f"benchmark_comparison_{safe_filename}.png"
            
            logger.info(f"Generating impact analysis for {model_name}...")
            viz.plot_overview(df_subset, f"Dataset Impact: {model_name}", output_filename)

    logger.info("Analysis complete. Please review artifacts in the 'results' directory.")

if __name__ == "__main__":
    main()