import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

def load_and_standardize_data(filepath: str, target_col_name: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Robust CSV loader for tabular data.
    """
    logger.info(f"--- [Loader] Loading: {os.path.basename(filepath)} ---")
    try:
        # Robust parsing: Skip malformed lines
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f"Fallback loading triggered due to: {e}")
        # Fallback: Warn but attempt to proceed
        df = pd.read_csv(filepath, on_bad_lines='warn')

    if target_col_name not in df.columns:
        error_msg = f"Error: Target column '{target_col_name}' missing in {filepath}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    y_raw = df[target_col_name]
    X_raw = df.drop(columns=[target_col_name])

    # Feature Sanitization
    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']
    X_raw = X_raw.drop(columns=drop_candidates, errors='ignore')

    # Type Coercion & Imputation
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Label Encoding
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    logger.info(f"--- [Loader] Loaded Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

def load_url_data(csv_file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Specialized ingestion pipeline for NLP/URL tasks.
    """
    logger.info(f"Loading URL data from: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label'
        
        if url_col not in df.columns or label_col not in df.columns:
            logger.error(f"Error: Required columns '{url_col}' or '{label_col}' missing.")
            return [], np.array([])

        df = df.dropna(subset=[url_col, label_col])
        urls = df[url_col].astype(str).tolist()
        
        # Label Normalization Logic
        if pd.api.types.is_numeric_dtype(df[label_col]):
            labels = df[label_col].values
        else:
            label_mapping = {'phishing': 1, 'bad': 1, 'malicious': 1, 
                             'benign': 0, 'good': 0, 'legitimate': 0}
            labels = df[label_col].map(label_mapping)
            
            if labels.isna().any():
                mask = labels.notna()
                urls = [u for u, m in zip(urls, mask) if m]
                labels = labels.dropna()
            labels = labels.values.astype(int)
            
        logger.info(f"Successfully ingested: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        logger.critical(f"Critical Error loading CSV: {e}", exc_info=True)
        return [], np.array([])

def encode_urls(urls: List[str], char_to_int: Dict[str, int], max_len: int = 200) -> np.ndarray:
    # Hier ist kein Logging nötig, da dies eine Hilfsfunktion in einer Schleife ist
    # (Logging hier würde die Log-Dateien fluten).
    encoded_batch = []
    for url in urls:
        vec = [char_to_int.get(c, 1) for c in url]
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded_batch.append(vec)
    return np.array(encoded_batch)