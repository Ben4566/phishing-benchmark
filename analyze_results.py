import pandas as pd
import json
import os
import sys

# Importiert den Visualizer
from src.visualize import BenchmarkVisualizer
from src.logger import setup_logger

logger = setup_logger("Analyzer")

INPUT_JSON = "benchmark_results.json"
OUTPUT_DIR = "results"

# --- ZENTRALE KONFIGURATION ---
MODEL_DISPLAY_NAMES = {
    'cnn': 'CNN (PyTorch)',
    'xgb': 'XGBoost',
    'lr': 'Logistic Regression',
    'svm': 'Linear SVC',
    'rf': 'Random Forest' 
}

def load_and_prep_data(filepath):
    """Lädt JSON und bereitet Namen für die Plots vor."""
    if not os.path.exists(filepath):
        logger.error(f"Datei nicht gefunden: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error("JSON Datei ist defekt oder leer.")
            return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty: return df

    # 1. Namen normalisieren (mapping)
    df['model'] = df['model'].map(lambda x: MODEL_DISPLAY_NAMES.get(x.lower(), x))

    # 2. Datasets unterscheiden
    # Wir hängen "(PhiUSIIL)" an den Namen, wenn es das PhiUSIIL Dataset ist.
    def refine_name(row):
        ds = row.get('dataset', '')
        if 'PhiUSIIL' in str(ds):
            return f"{row['model']} (PhiUSIIL)"
        return row['model']
    
    df['model'] = df.apply(refine_name, axis=1)

    return df

def main():
    logger.info("Starte Analyse...")
    
    # 1. Daten laden
    df = load_and_prep_data(INPUT_JSON)
    if df.empty:
        logger.warning("Keine Daten in benchmark_results.json gefunden. Haben Sie 'run_benchmark.py -m' ausgeführt?")
        sys.exit(0)

    # 2. CSV Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV exportiert: {csv_path}")

    # 3. Visualisierung
    viz = BenchmarkVisualizer(OUTPUT_DIR)
    
    # --- NEUE LOGIK: Gezielte Plots ---

    # A) Plot: Basismodelle Vergleich (Nur Standard-Datensatz)
    # Logik: Wir nehmen alles, was NICHT "PhiUSIIL" im Namen hat.
    logger.info("Erstelle Plot: Basismodelle (Standard Datensatz)...")
    df_base = df[~df['model'].str.contains("PhiUSIIL")].copy()
    
    if not df_base.empty:
        viz.plot_overview(df_base, "Base Model Comparison (Standard Data)", "benchmark_base_models.png")
    else:
        logger.warning("Keine Standard-Daten gefunden (evtl. nur PhiUSIIL Runs vorhanden?).")

    # B) Plots: Einzelvergleiche (Dataset Impact)
    # Wir erstellen für jedes Basismodell einen eigenen Plot: Modell vs. Modell (PhiUSIIL)
    base_models = ['CNN (PyTorch)', 'XGBoost', 'Logistic Regression', 'Linear SVC']
    
    for model_name in base_models:
        # Filter: Der Name muss den Basis-String enthalten.
        # "XGBoost" matcht auf "XGBoost" UND "XGBoost (PhiUSIIL)"
        df_subset = df[df['model'].str.contains(model_name, regex=False)].copy()
        
        if not df_subset.empty:
            # Dateinamen säubern (Leerzeichen weg, lowercase)
            safe_filename = model_name.replace(" ", "").replace("(", "").replace(")", "").lower()
            output_filename = f"benchmark_comparison_{safe_filename}.png"
            
            logger.info(f"Erstelle Plot: Dataset Vergleich für {model_name}...")
            viz.plot_overview(df_subset, f"Dataset Impact: {model_name}", output_filename)

    logger.info("Analyse abgeschlossen. Prüfen Sie den 'results' Ordner.")

if __name__ == "__main__":
    main()