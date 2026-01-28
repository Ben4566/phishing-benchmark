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
    Jeder Aufruf dieser Funktion entspricht EINEM Experiment.
    Inkompatible Kombinationen werden jetzt bereits in config.yaml via 'exclude' gefiltert.
    """
    
    # 1. Logging
    logger.info(f"--- Starting Run [Model: {cfg.model.name} | Dataset: {cfg.dataset.name}] ---")
    logger.debug(f"Full Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Global Seed setzen
    train.seed_everything(cfg.seed)

    # 3. Parameter vorbereiten
    model_name = cfg.model.name

    # 4. Training starten
    try:
        # Die Checks "if is_text_pure and model_name == xgb..." sind nicht mehr nötig!
        
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
        raise e

if __name__ == "__main__":
    # AUTOMATISIERUNG:
    # Wenn keine Argumente übergeben wurden (len=1, da Skriptname selbst arg[0] ist),
    # starte automatisch den vollen Benchmark (-m).
    if len(sys.argv) == 1:
        logger.info("--- Keine Argumente gefunden. Starte automatischen Full-Benchmark (-m) ---")
        sys.argv.append("-m")

    main()