import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

def load_and_standardize_data(filepath: str, target_col_name: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Robust ETL (Extract, Transform, Load) pipeline for tabular CSV data.
    Includes automated sanitization and data leakage prevention.
    """
    logger.info(f"--- [Loader] Ingesting: {os.path.basename(filepath)} ---")
    try:
        # Strategy: Strict parsing first, permissive fallback on failure
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f"Standard loading failed ({e}). Attempting permissive fallback...")
        df = pd.read_csv(filepath, on_bad_lines='warn')

    if target_col_name not in df.columns:
        error_msg = f"Schema Validation Error: Target column '{target_col_name}' missing in {filepath}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    y_raw = df[target_col_name]
    X_raw = df.drop(columns=[target_col_name])

    # --- Feature Sanitization Strategy ---
    # 1. Removal of Identifiers (non-predictive, high cardinality)
    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']

    # 2. Prevention of Data Leakage (Target Proxies)
    # These features often contain information that wouldn't be available at inference time
    # or strongly imply the label, artificially inflating metrics.
    leakage_features = [
        'URLSimilarityIndex',  # Primary source of leakage (direct correlation to target)
        'CharContinuationRate', # Highly correlated heuristic
        'TLDLegitimateProb',    # Probabilistic leakage (pre-calculated likelihood)
        'URLCharProb',          # Probabilistic leakage
        'Domain'                # String feature; high cardinality causes overfitting in tree models
    ]
    
    cols_to_drop = drop_candidates + leakage_features
    X_raw = X_raw.drop(columns=cols_to_drop, errors='ignore')

    # Type Coercion & Imputation
    # Force numeric types and handle missing values (Zero Imputation)
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Label Encoding
    # Ensure labels are strict integers (0/1) for binary classification
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    logger.info(f"--- [Loader] Complete. Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

def load_url_data(csv_file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Specialized ingestion pipeline for NLP/URL tasks.
    Handles raw text extraction and label normalization.
    """
    logger.info(f"Loading raw URL data from: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        
        # Column Discovery
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label'
        
        if url_col not in df.columns or label_col not in df.columns:
            logger.error(f"Schema Error: Required columns '{url_col}' or '{label_col}' missing.")
            return [], np.array([])

        df = df.dropna(subset=[url_col, label_col])
        urls = df[url_col].astype(str).tolist()
        
        # Label Normalization Logic
        if pd.api.types.is_numeric_dtype(df[label_col]):
            labels = df[label_col].values
        else:
            # Map string labels to binary format
            label_mapping = {
                'phishing': 1, 'bad': 1, 'malicious': 1, 
                'benign': 0, 'good': 0, 'legitimate': 0
            }
            labels = df[label_col].map(label_mapping)
            
            # Clean up unmappable labels
            if labels.isna().any():
                mask = labels.notna()
                urls = [u for u, m in zip(urls, mask) if m]
                labels = labels.dropna()
            labels = labels.values.astype(int)
            
        logger.info(f"Successfully ingested: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        logger.critical(f"Critical failure during CSV ingestion: {e}", exc_info=True)
        return [], np.array([])

def encode_urls(urls: List[str], char_to_int: Dict[str, int], max_len: int = 200) -> np.ndarray:
    """
    Vectorizes a list of URL strings into integer sequences (Character-Level Encoding).
    Truncates or pads sequences to fixed 'max_len'.
    """
    # Note: Logging is intentionally omitted here to prevent log flooding during high-volume loops.
    encoded_batch = []
    for url in urls:
        vec = [char_to_int.get(c, 1) for c in url]
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded_batch.append(vec)
    return np.array(encoded_batch)