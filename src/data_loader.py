import pandas as pd
import numpy as np
import os

def load_and_standardize_data(filepath, target_col_name="label"):
    print(f"--- [Loader] Loading: {os.path.basename(filepath)} ---")
    try:
        # Modernes Pandas handling für defekte Zeilen
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        print(f"Fallback loading due to: {e}")
        df = pd.read_csv(filepath, on_bad_lines='warn')

    if target_col_name not in df.columns:
        raise ValueError(f"Error: Column '{target_col_name}' missing in {filepath}.")

    y_raw = df[target_col_name]
    X_raw = df.drop(columns=[target_col_name])

    # Entfernen von Metadaten
    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']
    X_raw = X_raw.drop(columns=drop_candidates, errors='ignore')

    # Numerische Konvertierung erzwingen (NaN -> 0)
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    print(f"--- [Loader] Loaded Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

def load_url_data(csv_file_path):
    print(f"Lade URL-Daten aus: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label'
        
        if url_col not in df.columns or label_col not in df.columns:
            print(f"Fehler: Spalten '{url_col}' oder '{label_col}' fehlen.")
            return [], np.array([])

        df = df.dropna(subset=[url_col, label_col])
        # Sicherstellen, dass es wirklich Strings sind
        urls = df[url_col].astype(str).tolist()
        
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
            
        print(f"Erfolgreich geladen: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        print(f"Fehler beim Laden der CSV: {e}")
        return [], np.array([])

def encode_urls(urls, char_to_int, max_len=200):
    encoded_batch = []
    for url in urls:
        vec = [char_to_int.get(c, 1) for c in url]
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded_batch.append(vec)
    return np.array(encoded_batch)