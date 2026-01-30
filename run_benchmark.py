import hydra
import os
import re
import sys
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Eigene Module
from src.factories import create_model_adapter
from src.engine import Trainer
from src.benchmark import PerformanceMonitor
from src.logger import setup_logger
from src.data_loader import load_url_data, load_and_standardize_data, encode_urls 
from src.utils import seed_everything, enforce_imbalance

logger = setup_logger("Main")

def validate_config_combination(cfg):
    """
    Prüft, ob die Kombination aus Modell und Datensatz gültig ist.
    Gibt False zurück, wenn der Run übersprungen werden soll.
    """
    # Wir holen die echten Namen aus den YAMLs (nicht die Dateinamen)
    model_name = cfg.model.name.lower()
    dataset_name = cfg.dataset.name  # z.B. "raw_urls" oder "numeric_features"

    # Regel 1: Numerische Modelle (XGB, LR) brauchen Features, keine Raw URLs
    if model_name in ["xgb", "lr"] and dataset_name == "raw_urls":
        logger.warning(f"SKIP: Modell '{model_name}' kann nicht auf '{dataset_name}' trainiert werden.")
        return False
    
    # Regel 2: Text-Modelle (CNN, SVM) brauchen Text, keine rein numerischen Features
    # Annahme: "numeric_features" ist der Name in features.yaml
    if model_name in ["cnn", "svm"] and dataset_name == "numeric_features":
        logger.warning(f"SKIP: Modell '{model_name}' erwartet Text, aber '{dataset_name}' sind Zahlen.")
        return False

    return True

def prepare_data(cfg):
    """
    Lädt Daten, splittet sie, wendet Imbalance an und bereitet Features vor.
    """
    path = hydra.utils.to_absolute_path(cfg.dataset.path)
    
    # Unterscheidung: Text (CNN, SVM) vs. Numerisch (XGB, LR)
    is_text_model = cfg.model.name in ["cnn", "svm"]
    
    if is_text_model:
        urls, labels = load_url_data(path)
        # Split ZUERST
        X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=cfg.seed)
    else:
        # Numerischer Pfad
        X, y = load_and_standardize_data(path, "label")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg.seed)

    # Imbalance Handling
    if cfg.imbalance_ratio > 0:
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=cfg.imbalance_ratio)

    # Feature Engineering
    vocab_size = 0
    input_dim = 0

    if cfg.model.name == "cnn":
        chars = sorted(list(set("".join(X_train))))
        char_to_int = {c: i+2 for i, c in enumerate(chars)}
        vocab_size = len(char_to_int) + 2
        
        X_train = encode_urls(X_train, char_to_int)
        X_test = encode_urls(X_test, char_to_int)
        
    elif cfg.model.name == "svm":
        X_train = [re.sub(r'\W+', ' ', str(u)) for u in X_train]
        X_test = [re.sub(r'\W+', ' ', str(u)) for u in X_test]
        
    elif not is_text_model:
        input_dim = X_train.shape[1]

    return X_train, X_test, y_train, y_test, vocab_size, input_dim

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Validierung VOR allem anderen
    if not validate_config_combination(cfg):
        return # Beendet diesen Run sauber, Hydra macht mit dem nächsten weiter

    # 2. Reproduzierbarkeit
    seed_everything(cfg.seed)
    
    logger.info(f"--- Starte Run: {cfg.model.name} | Dataset: {cfg.dataset.name} | Seed: {cfg.seed} ---")
    
    try:
        # 3. Daten laden
        X_train, X_test, y_train, y_test, vocab_size, input_dim = prepare_data(cfg)

        # 4. Modell erstellen
        model_adapter = create_model_adapter(cfg, input_dim=input_dim, vocab_size=vocab_size)
        
        # 5. Trainer Setup
        monitor = PerformanceMonitor(cfg.model.name, cfg.dataset.name)
        trainer = Trainer(model=model_adapter, monitor=monitor)
        
        # 6. Run
        trainer.run(X_train, y_train, X_test, y_test)
        
        # 7. Speichern
        save_path = os.path.join("results", f"{cfg.model.name}_model.bin")
        trainer.save_model(save_path)
        
    except Exception as e:
        logger.error(f"Fehler im Run {cfg.model.name} auf {cfg.dataset.name}: {e}", exc_info=True)
        # Wir werfen den Fehler nicht weiter, damit Hydra den Multirun nicht komplett abbricht,
        # sondern nur diesen einen Job als 'Failed' markiert.

if __name__ == "__main__":
    main()