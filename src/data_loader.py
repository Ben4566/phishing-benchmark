import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict

# --- ETL & PREPROCESSING MODULE ---
# This module handles the extraction, transformation, and loading (ETL) of raw data.
# It ensures robustness against malformed inputs and standardizes formats for downstream models.

def load_and_standardize_data(filepath: str, target_col_name: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Robust CSV loader for tabular data.
    Performs type coercion, missing value imputation, and metadata removal.
    
    Args:
        filepath (str): Path to the dataset.
        target_col_name (str): Name of the target variable column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned feature matrix (X) and target vector (y).
    """
    print(f"--- [Loader] Loading: {os.path.basename(filepath)} ---")
    try:
        # Robust parsing: Skip malformed lines (common in scraped datasets) rather than crashing
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        print(f"Fallback loading triggered due to: {e}")
        # Fallback: Warn but attempt to proceed if strictly skipping fails
        df = pd.read_csv(filepath, on_bad_lines='warn')

    if target_col_name not in df.columns:
        raise ValueError(f"Error: Target column '{target_col_name}' missing in {filepath}.")

    y_raw = df[target_col_name]
    X_raw = df.drop(columns=[target_col_name])

    # Feature Sanitization:
    # Remove high-cardinality metadata (IDs, filenames) that cause data leakage 
    # or have no predictive power for the model.
    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']
    X_raw = X_raw.drop(columns=drop_candidates, errors='ignore')

    # Type Coercion & Imputation:
    # Force numeric types (coercing errors to NaN) and fill missing values with 0.
    # Note: In production, mean/median imputation might be preferred over 0-filling,
    # but 0-filling is standard for sparse feature sets.
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Label Encoding: Handle categorical targets if necessary
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    print(f"--- [Loader] Loaded Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

def load_url_data(csv_file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Specialized ingestion pipeline for NLP/URL tasks.
    Extracts raw URL strings and normalizes semantic labels (e.g., 'phishing' -> 1).
    """
    print(f"Loading URL data from: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        
        # Normalize column names to handle dataset heterogeneity
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label'
        
        if url_col not in df.columns or label_col not in df.columns:
            print(f"Error: Required columns '{url_col}' or '{label_col}' missing.")
            return [], np.array([])

        # Data Integrity: Drop rows where critical data is missing
        df = df.dropna(subset=[url_col, label_col])
        
        # Enforce string typing for URLs (prevent float interpretation of "NaN" strings)
        urls = df[url_col].astype(str).tolist()
        
        # Label Normalization Logic
        if pd.api.types.is_numeric_dtype(df[label_col]):
            labels = df[label_col].values
        else:
            # Semantic mapping for heterogeneous sources
            label_mapping = {'phishing': 1, 'bad': 1, 'malicious': 1, 
                             'benign': 0, 'good': 0, 'legitimate': 0}
            labels = df[label_col].map(label_mapping)
            
            # Clean up unmapped labels
            if labels.isna().any():
                mask = labels.notna()
                urls = [u for u, m in zip(urls, mask) if m]
                labels = labels.dropna()
            labels = labels.values.astype(int)
            
        print(f"Successfully ingested: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        print(f"Critical Error loading CSV: {e}")
        return [], np.array([])

def encode_urls(urls: List[str], char_to_int: Dict[str, int], max_len: int = 200) -> np.ndarray:
    """
    Tokenization & Sequence Padding.
    Converts raw URL strings into integer sequences for Embedding layers.
    
    Args:
        urls: List of URL strings.
        char_to_int: Vocabulary mapping characters to integers.
        max_len: Fixed sequence length (truncation/padding limit).
        
    Returns:
        np.ndarray: Matrix of shape (num_samples, max_len).
    """
    encoded_batch = []
    for url in urls:
        # Map chars to ints, default to 1 (<UNK>) if unknown
        vec = [char_to_int.get(c, 1) for c in url]
        
        # Sequence Standardization:
        # Apply Post-Padding (appending zeros) or Truncation to ensure fixed tensor size.
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec)) # Zero-padding
        else:
            vec = vec[:max_len] # Truncation
            
        encoded_batch.append(vec)
        
    return np.array(encoded_batch)