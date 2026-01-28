import pandas as pd
import json
import os
import sys
from src.visualize import generate_plots
from src.logger import setup_logger

logger = setup_logger("Analyzer")

# --- KONFIGURATION ---
# Eingabedatei (liegt meist im Hauptverzeichnis, da benchmark.py dorthin schreibt)
INPUT_JSON = "benchmark_results.json"

# Ausgabeverzeichnis (Hierhin sollen CSV und Bilder)
OUTPUT_DIR = "results"
# ---------------------

def main():
    # 1. Prüfen, ob Ergebnisse existieren
    if not os.path.exists(INPUT_JSON):
        logger.error(f"Datei '{INPUT_JSON}' nicht gefunden.")
        logger.error("Bitte führe zuerst den Benchmark aus: python run_benchmark.py -m ...")
        sys.exit(1)

    logger.info(f"Lade Ergebnisse aus {INPUT_JSON}...")
    
    try:
        # Ordner 'results' erstellen, falls er nicht existiert
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # JSON laden
        with open(INPUT_JSON, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("Die Ergebnisdatei ist leer.")
            return

        # 2. CSV Export in den results-Ordner speichern
        # Wir bauen den Pfad: results/benchmark_summary.csv
        csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV gespeichert unter: {csv_path}")

        # 3. Visualisierung starten
        # Wir übergeben 'results' als Zielordner an deine visualize.py
        logger.info(f"Erstelle Grafiken im Ordner '{OUTPUT_DIR}'...")
        generate_plots(df, OUTPUT_DIR)
        
        logger.info("Analyse abgeschlossen.")

    except json.JSONDecodeError:
        logger.error("Fehler beim Lesen der JSON-Datei. Format defekt.")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)

if __name__ == "__main__":
    main()