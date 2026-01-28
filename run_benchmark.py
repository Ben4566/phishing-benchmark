import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Deine Module importieren
import src.train as train
from src.logger import setup_logger

# Logger setup
logger = setup_logger("HydraRunner")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra Entry Point.
    Jeder Aufruf dieser Funktion entspricht EINEM Experiment (ein Modell, ein Dataset, ein Seed).
    """
    
    # 1. Logging der aktiven Konfiguration (wichtig für Reproduzierbarkeit)
    # Hydra erstellt automatisch einen Ordner für diesen Run und speichert dort Logs.
    logger.info(f"--- Starting Run [Model: {cfg.model.name} | Dataset: {cfg.dataset.name}] ---")
    logger.debug(f"Full Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Global Seed setzen
    train.seed_everything(cfg.seed)

    # 3. Pfad- und Kompatibilitäts-Checks (Guardrails)
    # Wir müssen prüfen, ob das gewählte Modell zum gewählten Dataset passt.
    dataset_path = str(cfg.dataset.path)
    model_name = cfg.model.name

    is_text_data = "url" in dataset_path.lower() or "combined" in dataset_path.lower()
    is_numeric_data = "feature" in dataset_path.lower() or "phiusiil" in dataset_path.lower()

    # Regel: Numerische Modelle (XGB, LR) nicht auf Raw-URL-Text loslassen
    if is_text_data and model_name in ["xgb", "lr"]:
        logger.warning(f"SKIPPING: Model '{model_name}' is not designed for Text/URL data ({dataset_path}).")
        return # Beendet diesen Run sauber ohne Fehler

    # Regel: Deep Learning / NLP Modelle (CNN, SVM-TFIDF) nicht auf reinen Zahlen-Features
    # (Hinweis: CNN könnte angepasst werden, aber im aktuellen Code erwartet es Strings)
    if is_numeric_data and model_name in ["cnn", "svm"]:
        logger.warning(f"SKIPPING: Model '{model_name}' is not designed for Pre-extracted Numeric Features ({dataset_path}).")
        return

    # 4. Training starten
    try:
        # Hydra ändert das Working Directory. Wir nutzen hydra.utils.to_absolute_path in train.py
        # oder übergeben hier schon absolute Pfade, aber train.py wurde im letzten Schritt
        # darauf vorbereitet, das 'cfg' Objekt zu nehmen.
        
        if model_name == "cnn":
            train.run_cnn(cfg)
        elif model_name == "svm":
            train.run_svm_tfidf(cfg)
        elif model_name in ["lr", "xgb"]:
            train.run_numeric_model(model_name, cfg)
        else:
            logger.error(f"Unknown model architecture in config: {model_name}")
            
    except Exception as e:
        logger.error(f"Pipeline failed for {model_name} on {cfg.dataset.name}: {e}", exc_info=True)
        # Wir raisen den Fehler, damit Hydra den Run als 'FAILED' markiert
        raise e

if __name__ == "__main__":
    main()