import hydra
import os
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Domain-specific modules
from src.factories import create_model_adapter
from src.engine import Trainer
from src.benchmark import PerformanceMonitor
from src.logger import setup_logger
from src.data_loader import load_url_data, load_and_standardize_data, encode_urls 
from src.utils import seed_everything, enforce_imbalance

logger = setup_logger("Main")

def validate_config_combination(cfg):
    """
    Validates architectural compatibility between the selected model and the input dataset.
    Prevents execution of invalid pairings (e.g., CNNs on tabular features).

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    # Extract canonical names from configuration (ignoring filenames)
    model_name = cfg.model.name.lower()
    dataset_name = cfg.dataset.name  # e.g., "raw_urls" or "numeric_features"

    # Rule 1: Gradient Boosting & Linear Regressors require numerical feature vectors, not raw text strings.
    if model_name in ["xgb", "lr"] and dataset_name == "raw_urls":
        logger.warning(f"Configuration Mismatch: Model '{model_name}' cannot process raw text data ('{dataset_name}'). Skipping run.")
        return False
    
    # Rule 2: Deep Learning text models (CNN) & SVMs imply raw text or sparse inputs, not pre-extracted dense features.
    if model_name in ["cnn", "svm"] and dataset_name == "numeric_features":
        logger.warning(f"Configuration Mismatch: Model '{model_name}' expects text input, but dataset '{dataset_name}' contains numerical features. Skipping run.")
        return False

    return True

def prepare_data(cfg):
    """
    Orchestrates the data pipeline: Ingestion, Splitting, Imbalance Injection, and Preprocessing.
    
    Returns:
        tuple: Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test, vocab_size, input_dim)
    """
    path = hydra.utils.to_absolute_path(cfg.dataset.path)
    is_text_model = cfg.model.name in ["cnn", "svm"]
    
    # 1. Data Ingestion
    if is_text_model:
        X_raw, y_raw = load_url_data(path)
    else:
        X_raw, y_raw = load_and_standardize_data(path, "label")

    # 2. Stratified Partitioning (Train / Validation / Test)
    # We perform a 3-way split to ensure a dedicated validation set for early stopping.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=cfg.seed, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=cfg.seed, stratify=y_temp
    )

    # 3. Synthetic Anomaly Injection (Test Set Only)
    # Simulate real-world conditions by enforcing class imbalance in the evaluation set.
    if cfg.imbalance_ratio > 0:
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=cfg.imbalance_ratio)

    # 4. Feature Engineering & Preprocessing Pipeline
    vocab_size = 0
    input_dim = 0

    if cfg.model.name == "cnn":
        # Character-level Encoding for Deep Learning
        import string
        chars = sorted(list(string.printable))
        char_to_int = {c: i+2 for i, c in enumerate(chars)}
        vocab_size = len(char_to_int) + 2
        
        X_train = encode_urls(X_train, char_to_int)
        X_val   = encode_urls(X_val, char_to_int)
        X_test  = encode_urls(X_test, char_to_int)
        
    elif cfg.model.name == "svm":
        # Sparse Vectorization (TF-IDF)
        # Note: Vectorizer is fitted ONLY on the training set to prevent data leakage.
        logger.info("Applying TF-IDF vectorization for SVM...")
        vec = TfidfVectorizer(max_features=10000)
        
        X_train = vec.fit_transform(X_train)
        X_val   = vec.transform(X_val)
        X_test  = vec.transform(X_test)
        
    elif not is_text_model:
        # Dense Numerical Features (XGBoost & Logistic Regression)
        input_dim = X_train.shape[1]
        
        # Standardization (Z-Score Normalization)
        # Strictly required for Logistic Regression; beneficial for SVMs on dense features.
        if cfg.model.name in ["lr", "svm"]: 
            logger.info("Standardizing numerical features (Z-Score normalization)...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, vocab_size, input_dim

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Pre-flight Configuration Validation
    if not validate_config_combination(cfg):
        return 

    seed_everything(cfg.seed)
    
    logger.info(f"--- Initiating Benchmark Run: Model={cfg.model.name} | Dataset={cfg.dataset.name} | Seed={cfg.seed} ---")
    
    try:
        # 3. Data Pipeline Execution
        # Unpacking the complete 3-way split and metadata
        X_train, X_val, X_test, y_train, y_val, y_test, vocab_size, input_dim = prepare_data(cfg)

        # 4. Model Factory Instantiation
        model_adapter = create_model_adapter(cfg, input_dim=input_dim, vocab_size=vocab_size)
        
        # 5. Training Engine Initialization
        monitor = PerformanceMonitor(cfg.model.name, cfg.dataset.name)
        trainer = Trainer(model=model_adapter, monitor=monitor)
        
        # 6. Execution Phase (Training & Evaluation)
        # Validation set is explicitly passed to enable Early Stopping logic.
        trainer.run(X_train, y_train, X_test, y_test, X_val=X_val, y_val=y_val)
        
        # 7. Artifact Persistence
        orig_cwd = hydra.utils.get_original_cwd()
        output_dir = os.path.join(orig_cwd, "results", "models")
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{cfg.model.name}_{cfg.dataset.name}_{cfg.seed}.bin")
        trainer.save_model(save_path)
        logger.info(f"Model artifact successfully saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Critical failure during execution of {cfg.model.name} on {cfg.dataset.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()