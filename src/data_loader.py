import pandas as pd
import numpy as np
import os
import torch

# --- Bestehende Funktion für numerische Features (LogReg, XGBoost) ---
def load_and_standardize_data(filepath, target_col_name, delimiter=","):
    print(f"--- [Loader] Starte Laden von: {os.path.basename(filepath)} ---")
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(filepath, delimiter=delimiter, error_bad_lines=False)

    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']
    df = df.drop(columns=drop_candidates, errors='ignore')
    
    real_target_col = None
    if target_col_name in df.columns:
        real_target_col = target_col_name
    else:
        alternatives = ['label', 'class', 'Phishing?', 'phishing', 'target']
        for alt in alternatives:
            if alt in df.columns:
                real_target_col = alt
                break
        if real_target_col is None:
            real_target_col = df.columns[-1]

    y_raw = df[real_target_col]
    df_numeric = df.select_dtypes(include=[np.number])
    
    if real_target_col in df_numeric.columns:
        X = df_numeric.drop(columns=[real_target_col])
    else:
        X = df_numeric 
        
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    print(f"--- [Loader] Fertig. Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

# --- NEU: Funktion für Raw URLs (CNN, SVM) aus PhiUSIIL Notebook ---
def load_url_data(csv_file_path):
    """
    Lädt URLs und Labels speziell für Text-Modelle (CNN/SVM).
    """
    print(f"Lade URL-Daten aus: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
        
        # Spalten identifizieren
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label' if 'label' in df.columns else 'Phishing?'
        
        if url_col not in df.columns:
            return [], np.array([])

        df = df.dropna(subset=[url_col, label_col])
        
        # URLs extrahieren
        urls = df[url_col].astype(str).tolist()
        
        # Labels verarbeiten
        if pd.api.types.is_numeric_dtype(df[label_col]):
            labels = df[label_col].values
        else:
            # Mapping für PhiUSIIL Text Labels
            label_mapping = {'phishing': 1, 'bad': 1, 'malicious': 1, 
                             'benign': 0, 'good': 0, 'legitimate': 0}
            labels = df[label_col].map(label_mapping)
            
            # Fehlerhafte Labels entfernen
            if labels.isna().any():
                mask = labels.notna()
                urls = [u for u, m in zip(urls, mask) if m]
                labels = labels.dropna()
            
            labels = labels.values.astype(int)
            
        print(f"Erfolgreich geladen: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        print(f"Fehler beim Laden der CSV: {e}")
        return [], np.array([])

# --- NEU: Helper für CNN Vectorization ---
def encode_urls(urls, char_to_int, max_len=200):
    encoded_batch = []
    for url in urls:
        vec = [char_to_int.get(c, 1) for c in url] # 1 = Unknown
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded_batch.append(vec)
    return np.array(encoded_batch)